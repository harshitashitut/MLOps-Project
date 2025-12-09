"""
FastAPI Backend for PitchQuest Video Analysis
Clean HTTP layer - all business logic delegated to AnalysisService
Async processing with BackgroundTasks for real-time progress tracking
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import uuid
from datetime import datetime
import logging
import time
from dotenv import load_dotenv
from typing import Optional
import json

load_dotenv()

from utils.db_helper import (
    create_video_record,
    update_video_status,
    get_results_from_db,
    get_video_by_id,
    get_or_create_user,
    test_connection,
    get_videos_by_user,
    get_admin_analytics,
    check_user_is_admin
)
from services.analysis_service import AnalysisService, PipelineError

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== APP INITIALIZATION ====================
app = FastAPI(
    title="PitchQuest Video Analysis API",
    version="1.0.0",
    description="AI-powered pitch video analysis with real-time progress tracking"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURATION ====================
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

GUEST_USER_ID = "00000000-0000-0000-0000-000000000000"
MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Initialize service
analysis_service = AnalysisService()

# ==================== REQUEST LOGGING MIDDLEWARE ====================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"📥 [{request_id}] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"📤 [{request_id}] Status: {response.status_code} - {duration:.2f}s")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ [{request_id}] Error after {duration:.2f}s: {str(e)}")
        raise

# ==================== STARTUP & HEALTH ====================
@app.on_event("startup")
async def startup_event():
    """Initialize API and verify connections"""
    logger.info("🚀 Starting PitchQuest Video Analysis API v1.0.0")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    # Check if running in CI/test mode
    CI_MODE = os.getenv('CI_MODE', 'false').lower() == 'true'
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    
    if CI_MODE or 'mock' in SUPABASE_URL.lower():
        logger.warning("🧪 Running in CI/TEST mode - skipping database initialization")
        logger.info("✅ CI mode active - backend will start without database")
        return
    
    # Normal production mode - test database connection
    if test_connection():
        logger.info("✅ Database connection verified")
        try:
            get_or_create_user(GUEST_USER_ID, "guest@pitchquest.com", "Guest User")
            logger.info("✅ Guest user initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not create guest user: {e}")
    else:
        logger.error("❌ Database connection failed!")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "PitchQuest Video Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze-video",
            "status": "/api/status/{video_id}",
            "results": "/api/results/{analysis_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    CI_MODE = os.getenv('CI_MODE', 'false').lower() == 'true'
    
    if CI_MODE:
        return {
            "status": "healthy",
            "mode": "ci_test",
            "database": "skipped",
            "storage": {
                "upload_dir": str(UPLOAD_DIR.exists()),
                "output_dir": str(OUTPUT_DIR.exists())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    db_status = "healthy" if test_connection() else "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "storage": {
            "upload_dir": str(UPLOAD_DIR.exists()),
            "output_dir": str(OUTPUT_DIR.exists())
        },
        "timestamp": datetime.utcnow().isoformat()
    }
# ==================== BACKGROUND TASK FUNCTION ====================
def process_video_background(video_path: str, video_id: str, user_id: str):
    """
    Background task to process video
    Runs asynchronously so API can return immediately
    """
    try:
        # Run the pipeline
        results = analysis_service.process_video(
            video_path=video_path,
            video_id=video_id,
            user_id=user_id
        )
        
        # Save JSON backup
        output_file = OUTPUT_DIR / f"{video_id}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 JSON backup saved: {output_file.name}")
        
    except PipelineError as e:
        logger.error(f"Pipeline error in background task: {e}")
        update_video_status(video_id, "failed")
        
        # Cleanup uploaded file on error
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
    
    except Exception as e:
        logger.error(f"Unexpected error in background task: {e}", exc_info=True)
        update_video_status(video_id, "failed")
        
        # Cleanup uploaded file on error
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

# ==================== MAIN ANALYSIS ENDPOINT ====================
@app.post("/api/analyze-video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Upload and analyze a pitch video (ASYNC)
    
    Returns immediately with video_id
    Poll /api/status/{video_id} for real-time progress
    """
    analysis_id = str(uuid.uuid4())
    user_id = user_id or GUEST_USER_ID
    video_path = None
    
    try:
        # ==================== VALIDATION ====================
        logger.info(f"🎬 New video upload: {video.filename}")
        
        # Validate file type
        file_extension = Path(video.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{file_extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # ==================== UPLOAD ====================
        video_filename = f"{analysis_id}_{video.filename}"
        video_path = UPLOAD_DIR / video_filename
        
        # Save file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        # Validate file size
        if file_size_mb > MAX_FILE_SIZE_MB:
            os.remove(video_path)
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"✅ Upload complete: {file_size_mb:.2f}MB")
        
        # ==================== CREATE DATABASE RECORD ====================
        create_video_record(
            video_id=analysis_id,
            user_id=user_id,
            filename=video.filename,
            file_path=str(video_path),
            file_size_mb=file_size_mb
        )
        
        update_video_status(analysis_id, "processing")
        
        # ==================== START BACKGROUND PROCESSING ====================
        background_tasks.add_task(
            process_video_background,
            str(video_path),
            analysis_id,
            user_id
        )
        
        logger.info(f"🚀 Background processing started for {analysis_id[:8]}")
        
        # ==================== RETURN IMMEDIATELY ====================
        return JSONResponse(content={
            "status": "processing",
            "video_id": analysis_id,
            "message": "Video uploaded successfully. Analysis in progress.",
            "poll_url": f"/api/status/{analysis_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    
    except Exception as e:
        # Unexpected errors during upload
        logger.error(f"Upload error: {e}", exc_info=True)
        
        # Cleanup uploaded file
        if video_path and video_path.exists():
            try:
                os.remove(video_path)
            except Exception:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

# ==================== STATUS & RESULTS ENDPOINTS ====================
@app.get("/api/status/{video_id}")
async def get_video_status(video_id: str):
    """
    Get real-time processing status and progress
    
    Frontend should poll this endpoint during processing
    """
    video = get_video_by_id(video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = video.get("status", "unknown")
    progress = video.get("progress_percentage", 0)
    current_step = video.get("current_step", "")
    
    response = {
        "video_id": video_id,
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "uploaded_at": video.get("uploaded_at"),
        "processing_started_at": video.get("processing_started_at"),
        "processing_completed_at": video.get("processing_completed_at")
    }
    
    # If completed, include summary scores
    if status == "completed":
        results = get_results_from_db(video_id)
        if results:
            response["overall_score"] = results.get("overall_score")
            response["performance_level"] = results.get("performance_level")
    
    return JSONResponse(content=response)

@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """
    Retrieve complete analysis results
    
    Returns full results with all scores, feedback, and recommendations
    """
    # Try database first
    db_results = get_results_from_db(analysis_id)
    if db_results:
        return JSONResponse(content=db_results)
    
    # Fallback to JSON file
    output_file = OUTPUT_DIR / f"{analysis_id}_results.json"
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
        return JSONResponse(content=results)
    
    raise HTTPException(status_code=404, detail="Analysis results not found")

@app.get("/api/video/{analysis_id}")
async def get_video_info(analysis_id: str):
    """Get video metadata and processing information"""
    video = get_video_by_id(analysis_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return JSONResponse(content=video)


@app.get("/api/videos")
async def get_user_videos(user_id: Optional[str] = Header(None, alias="X-User-ID")):
    """
    Get all videos for a user with their scores
    
    Returns list of videos with overall scores for dashboard
    """
    user_id = user_id or GUEST_USER_ID
    
    from utils.db_helper import get_videos_by_user
    videos = get_videos_by_user(user_id)
    
    return JSONResponse(content={
        "user_id": user_id,
        "count": len(videos),
        "videos": videos
    })



@app.get("/api/admin/analytics")
async def admin_analytics(user_id: Optional[str] = Header(None, alias="X-User-ID")):
    """
    Get platform-wide analytics (admin only)
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_user_is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    analytics = get_admin_analytics()
    return JSONResponse(content=analytics)


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )