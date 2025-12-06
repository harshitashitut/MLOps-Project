# Backend Implementation Guide (FastAPI)

**Owner:** Person 1 (Backend Lead)  
**Timeline:** Day 1-3  
**Tech Stack:** FastAPI, Python 3.10, Uvicorn, SQLAlchemy, Requests

---

## Overview

You're building the orchestration layer that connects the frontend to Airflow and Supabase. Think of it as the "traffic controller" that:
- Accepts video uploads from users
- Triggers Airflow DAGs
- Polls Airflow for status
- Retrieves results from Supabase
- Handles all errors gracefully

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Environment variables
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # All API endpoints
│   │   └── models.py           # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── airflow_client.py   # Airflow REST API wrapper
│   │   ├── storage.py          # File upload handling
│   │   └── monitoring.py       # Drift metrics aggregation
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── supabase_client.py  # Supabase connection
│   │   └── crud.py             # Database operations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Structured logging
│       └── exceptions.py       # Custom exceptions
│
├── tests/
│   ├── test_routes.py
│   └── test_airflow_client.py
│
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Day 1: Foundation (4-6 hours)

### Step 1: Project Setup (30 min)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-multipart supabase requests pydantic-settings python-dotenv
pip freeze > requirements.txt
```

### Step 2: Create Main Application (30 min)

**File: `app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="PitchQuest API",
    description="Backend API for AI-powered pitch analysis",
    version="1.0.0"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 PitchQuest Backend API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 PitchQuest Backend API shutting down...")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "pitchquest-backend",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Step 3: Configuration Management (15 min)

**File: `app/config.py`**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str
    
    # Airflow
    AIRFLOW_URL: str = "http://localhost:8080"
    AIRFLOW_USERNAME: str = "admin"
    AIRFLOW_PASSWORD: str = "admin"
    
    # API Keys
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    
    # App Config
    UPLOAD_DIR: str = "/tmp/pitchquest_uploads"
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: list = [".mp4", ".mov", ".avi"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

**File: `.env.example`**

```env
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-anon-key-here
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres

# Airflow
AIRFLOW_URL=http://localhost:8080
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin

# API Keys
OPENAI_API_KEY=your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here

# App Config
UPLOAD_DIR=/tmp/pitchquest_uploads
MAX_UPLOAD_SIZE_MB=100
```

### Step 4: Logger Setup (15 min)

**File: `app/utils/logger.py`**

```python
import logging
import sys
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """Configure structured logger"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Format: [2025-12-06 10:30:45] INFO [module_name] Message
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
```

### Step 5: Custom Exceptions (15 min)

**File: `app/utils/exceptions.py`**

```python
from fastapi import HTTPException, status

class PitchQuestException(HTTPException):
    """Base exception for PitchQuest API"""
    pass

class VideoUploadError(PitchQuestException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video upload failed: {detail}"
        )

class AirflowConnectionError(PitchQuestException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to connect to Airflow: {detail}"
        )

class VideoNotFoundError(PitchQuestException):
    def __init__(self, video_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {video_id}"
        )

class AnalysisNotReadyError(PitchQuestException):
    def __init__(self, video_id: str):
        super().__init__(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Analysis not ready for video: {video_id}"
        )
```

### Step 6: Pydantic Models (30 min)

**File: `app/api/models.py`**

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AnalysisStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    status: AnalysisStatus
    message: str
    uploaded_at: datetime

class AnalysisRequest(BaseModel):
    video_id: str

class AnalysisResponse(BaseModel):
    job_id: str
    video_id: str
    status: AnalysisStatus
    message: str
    estimated_time_seconds: Optional[int] = 120

class StatusResponse(BaseModel):
    status: AnalysisStatus
    progress: int = Field(ge=0, le=100)
    current_task: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    
    @validator('progress')
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Progress must be between 0 and 100')
        return v

class ResultsResponse(BaseModel):
    video_id: str
    overall_score: int
    results: Dict[str, Any]
    feedback: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time_seconds: float
    created_at: datetime

class MonitoringResponse(BaseModel):
    total_analyses: int
    avg_score: float
    system_health: str  # "healthy" | "warning" | "critical"
    drift_detected: bool
    score_history: list
    last_updated: datetime
```

### Step 7: Supabase Client (30 min)

**File: `app/database/supabase_client.py`**

```python
from supabase import create_client, Client
from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    try:
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("✅ Supabase client connected")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        raise

supabase: Client = get_supabase_client()
```

**File: `app/database/crud.py`**

```python
from app.database.supabase_client import supabase
from app.utils.logger import setup_logger
from app.utils.exceptions import VideoNotFoundError
from datetime import datetime
from typing import Optional, Dict, Any

logger = setup_logger(__name__)

async def create_video_record(
    video_id: str,
    filename: str,
    file_path: str,
    file_size_mb: float,
    duration_seconds: Optional[float] = None
) -> Dict:
    """Create video record in database"""
    try:
        data = {
            "video_id": video_id,
            "filename": filename,
            "file_path": file_path,
            "file_size_mb": file_size_mb,
            "duration_seconds": duration_seconds,
            "status": "uploaded",
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("videos").insert(data).execute()
        logger.info(f"✅ Created video record: {video_id}")
        return result.data[0]
    except Exception as e:
        logger.error(f"❌ Failed to create video record: {e}")
        raise

async def update_video_status(video_id: str, status: str, job_id: Optional[str] = None):
    """Update video processing status"""
    try:
        data = {"status": status}
        if job_id:
            data["job_id"] = job_id
        
        result = supabase.table("videos").update(data).eq("video_id", video_id).execute()
        logger.info(f"✅ Updated video status: {video_id} -> {status}")
        return result.data[0]
    except Exception as e:
        logger.error(f"❌ Failed to update video status: {e}")
        raise

async def get_video_by_id(video_id: str) -> Dict:
    """Fetch video record by ID"""
    try:
        result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
        if not result.data:
            raise VideoNotFoundError(video_id)
        return result.data[0]
    except VideoNotFoundError:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch video: {e}")
        raise

async def get_analysis_results(video_id: str) -> Optional[Dict]:
    """Fetch analysis results for video"""
    try:
        result = supabase.table("analysis_results").select("*").eq("video_id", video_id).execute()
        if not result.data:
            return None
        return result.data[0]
    except Exception as e:
        logger.error(f"❌ Failed to fetch results: {e}")
        raise

async def get_monitoring_metrics() -> Dict:
    """Fetch monitoring metrics"""
    try:
        # Get total count
        videos = supabase.table("videos").select("*", count="exact").execute()
        total_analyses = videos.count
        
        # Get average score
        results = supabase.table("analysis_results").select("overall_score").execute()
        scores = [r["overall_score"] for r in results.data]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Get score history (last 30 days)
        score_history = supabase.table("analysis_results")\
            .select("overall_score, created_at")\
            .order("created_at", desc=False)\
            .limit(100)\
            .execute()
        
        return {
            "total_analyses": total_analyses,
            "avg_score": round(avg_score, 2),
            "score_history": score_history.data
        }
    except Exception as e:
        logger.error(f"❌ Failed to fetch metrics: {e}")
        raise
```

### Step 8: File Upload Service (30 min)

**File: `app/services/storage.py`**

```python
import os
import shutil
from pathlib import Path
from fastapi import UploadFile
from app.config import get_settings
from app.utils.logger import setup_logger
from app.utils.exceptions import VideoUploadError

logger = setup_logger(__name__)
settings = get_settings()

class StorageService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_video(self, file: UploadFile, video_id: str) -> str:
        """Save uploaded video to disk"""
        try:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.ALLOWED_EXTENSIONS:
                raise VideoUploadError(
                    f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
                )
            
            # Create file path
            file_path = self.upload_dir / f"{video_id}{file_ext}"
            
            # Save file
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
                file_path.unlink()  # Delete file
                raise VideoUploadError(
                    f"File too large ({file_size_mb:.1f}MB). Max: {settings.MAX_UPLOAD_SIZE_MB}MB"
                )
            
            logger.info(f"✅ Saved video: {file_path} ({file_size_mb:.1f}MB)")
            return str(file_path)
        
        except VideoUploadError:
            raise
        except Exception as e:
            logger.error(f"❌ Video save failed: {e}")
            raise VideoUploadError(str(e))
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        return Path(file_path).stat().st_size / (1024 * 1024)

storage_service = StorageService()
```

### Step 9: Airflow Client (1 hour) - CRITICAL

**File: `app/services/airflow_client.py`**

```python
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Optional
from app.config import get_settings
from app.utils.logger import setup_logger
from app.utils.exceptions import AirflowConnectionError

logger = setup_logger(__name__)
settings = get_settings()

class AirflowClient:
    def __init__(self):
        self.base_url = settings.AIRFLOW_URL
        self.auth = HTTPBasicAuth(settings.AIRFLOW_USERNAME, settings.AIRFLOW_PASSWORD)
        self.dag_id = "pitchquest_video_analysis"
    
    def trigger_dag(self, video_id: str, file_path: str) -> Dict:
        """Trigger Airflow DAG for video analysis"""
        try:
            url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns"
            
            payload = {
                "conf": {
                    "video_id": video_id,
                    "video_path": file_path
                }
            }
            
            response = requests.post(
                url,
                json=payload,
                auth=self.auth,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data["dag_run_id"]
                logger.info(f"✅ Triggered DAG: {job_id} for video {video_id}")
                return {
                    "job_id": job_id,
                    "status": "running",
                    "dag_id": self.dag_id
                }
            else:
                logger.error(f"❌ Airflow trigger failed: {response.text}")
                raise AirflowConnectionError(response.text)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Airflow connection error: {e}")
            raise AirflowConnectionError(str(e))
    
    def get_dag_run_status(self, job_id: str) -> Dict:
        """Get status of DAG run"""
        try:
            url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns/{job_id}"
            
            response = requests.get(url, auth=self.auth, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                state = data["state"]  # "running", "success", "failed"
                
                # Map Airflow states to our status
                status_map = {
                    "running": "processing",
                    "success": "completed",
                    "failed": "failed",
                    "queued": "processing"
                }
                
                return {
                    "status": status_map.get(state, "processing"),
                    "airflow_state": state,
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date")
                }
            else:
                raise AirflowConnectionError(f"Failed to get status: {response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get DAG status: {e}")
            raise AirflowConnectionError(str(e))
    
    def get_task_instances(self, job_id: str) -> list:
        """Get individual task statuses"""
        try:
            url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns/{job_id}/taskInstances"
            
            response = requests.get(url, auth=self.auth, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("task_instances", [])
            else:
                return []
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to get task instances: {e}")
            return []

airflow_client = AirflowClient()
```

### Step 10: API Routes - Upload Endpoint (30 min)

**File: `app/api/routes.py`**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.api.models import *
from app.services.storage import storage_service
from app.services.airflow_client import airflow_client
from app.database.crud import *
from app.utils.logger import setup_logger
from datetime import datetime
import uuid

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload video file and create database record"""
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        logger.info(f"📤 Uploading video: {file.filename}")
        
        # Save video file
        file_path = await storage_service.save_video(file, video_id)
        file_size_mb = storage_service.get_file_size_mb(file_path)
        
        # Create database record
        await create_video_record(
            video_id=video_id,
            filename=file.filename,
            file_path=file_path,
            file_size_mb=file_size_mb
        )
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            status=AnalysisStatus.UPLOADED,
            message="Video uploaded successfully. Ready for analysis.",
            uploaded_at=datetime.utcnow()
        )
    
    except VideoUploadError as e:
        logger.error(f"❌ Upload error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")

# TODO: Add remaining endpoints (analyze, status, results) in Day 2
```

### Step 11: Test the Backend (30 min)

```bash
# Start the server
cd backend
python -m app.main

# In another terminal, test upload
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_video.mp4"

# Should return:
# {
#   "video_id": "...",
#   "filename": "test_video.mp4",
#   "status": "uploaded",
#   ...
# }
```

### End of Day 1 Deliverable
- [ ] Backend runs on :8000
- [ ] `/health` returns 200
- [ ] `/api/upload` accepts video and saves to disk
- [ ] Video record created in Supabase
- [ ] Airflow client can connect (test manually)

---

## Day 2: Core Endpoints (6-8 hours)

### Step 1: Analyze Endpoint (1 hour)

Add to `app/api/routes.py`:

```python
@router.post("/analyze/{video_id}", response_model=AnalysisResponse)
async def trigger_analysis(video_id: str):
    """Trigger Airflow DAG to analyze video"""
    try:
        # Get video from database
        video = await get_video_by_id(video_id)
        
        if video["status"] == "processing":
            raise HTTPException(400, "Video is already being processed")
        
        if video["status"] == "completed":
            raise HTTPException(400, "Video has already been analyzed")
        
        # Trigger Airflow DAG
        logger.info(f"🚀 Triggering analysis for video: {video_id}")
        result = airflow_client.trigger_dag(video_id, video["file_path"])
        
        # Update database status
        await update_video_status(video_id, "processing", result["job_id"])
        
        return AnalysisResponse(
            job_id=result["job_id"],
            video_id=video_id,
            status=AnalysisStatus.PROCESSING,
            message="Analysis started successfully",
            estimated_time_seconds=120
        )
    
    except VideoNotFoundError as e:
        logger.error(f"❌ Video not found: {video_id}")
        raise
    except AirflowConnectionError as e:
        logger.error(f"❌ Airflow connection failed: {e}")
        raise HTTPException(503, "Analysis service unavailable. Please try again later.")
    except Exception as e:
        logger.error(f"❌ Analysis trigger failed: {e}")
        raise HTTPException(500, "Failed to start analysis. Please try again.")
```

### Step 2: Status Endpoint (1 hour)

```python
@router.get("/status/{video_id}", response_model=StatusResponse)
async def get_analysis_status(video_id: str):
    """Get current status of video analysis"""
    try:
        video = await get_video_by_id(video_id)
        
        if video["status"] == "uploaded":
            return StatusResponse(
                status=AnalysisStatus.UPLOADED,
                progress=0,
                current_task="Waiting to start",
                estimated_time_remaining=None
            )
        
        if video["status"] == "completed":
            return StatusResponse(
                status=AnalysisStatus.COMPLETED,
                progress=100,
                current_task="Completed",
                estimated_time_remaining=0
            )
        
        if video["status"] == "failed":
            return StatusResponse(
                status=AnalysisStatus.FAILED,
                progress=0,
                current_task="Failed",
                estimated_time_remaining=None
            )
        
        # For "processing" status, query Airflow
        job_id = video.get("job_id")
        if not job_id:
            raise HTTPException(500, "Job ID not found")
        
        airflow_status = airflow_client.get_dag_run_status(job_id)
        tasks = airflow_client.get_task_instances(job_id)
        
        # Calculate progress based on completed tasks
        if tasks:
            total_tasks = len(tasks)
            completed_tasks = sum(1 for t in tasks if t["state"] == "success")
            progress = int((completed_tasks / total_tasks) * 100)
            
            # Find current running task
            current_task = next(
                (t["task_id"] for t in tasks if t["state"] == "running"),
                "Processing..."
            )
        else:
            progress = 50
            current_task = "Processing video..."
        
        return StatusResponse(
            status=AnalysisStatus(airflow_status["status"]),
            progress=progress,
            current_task=current_task,
            estimated_time_remaining=max(0, 120 - (progress * 1.2))
        )
    
    except VideoNotFoundError:
        raise
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(500, "Failed to get status")
```

### Step 3: Results Endpoint (1 hour)

```python
@router.get("/results/{video_id}", response_model=ResultsResponse)
async def get_analysis_results(video_id: str):
    """Get analysis results for completed video"""
    try:
        # Check if video exists
        video = await get_video_by_id(video_id)
        
        if video["status"] != "completed":
            raise AnalysisNotReadyError(video_id)
        
        # Fetch results from database
        results = await get_analysis_results(video_id)
        
        if not results:
            raise HTTPException(404, "Results not found")
        
        return ResultsResponse(
            video_id=video_id,
            overall_score=results["overall_score"],
            results=results["content_analysis"],  # Contains all analysis sections
            feedback=results["feedback"],
            metadata=results["metadata"],
            processing_time_seconds=results["processing_time_seconds"],
            created_at=results["created_at"]
        )
    
    except VideoNotFoundError:
        raise
    except AnalysisNotReadyError:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch results: {e}")
        raise HTTPException(500, "Failed to retrieve results")
```

### Step 4: Monitoring Endpoint (1 hour)

Add monitoring service:

**File: `app/services/monitoring.py`**

```python
from app.database.crud import get_monitoring_metrics
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

async def calculate_system_health(avg_score: float, total_analyses: int) -> str:
    """Calculate overall system health"""
    if total_analyses < 5:
        return "healthy"  # Not enough data
    
    if avg_score < 30:
        return "critical"
    elif avg_score < 50:
        return "warning"
    else:
        return "healthy"

async def detect_drift(score_history: list) -> bool:
    """Simple drift detection based on recent scores"""
    if len(score_history) < 10:
        return False
    
    recent_scores = [s["overall_score"] for s in score_history[-10:]]
    avg_recent = sum(recent_scores) / len(recent_scores)
    
    older_scores = [s["overall_score"] for s in score_history[-30:-10]]
    if not older_scores:
        return False
    
    avg_older = sum(older_scores) / len(older_scores)
    
    # Drift if recent average differs by >15 points
    return abs(avg_recent - avg_older) > 15
```

Add to routes:

```python
from app.services.monitoring import calculate_system_health, detect_drift

@router.get("/monitoring", response_model=MonitoringResponse)
async def get_monitoring_data():
    """Get system monitoring metrics"""
    try:
        metrics = await get_monitoring_metrics()
        
        system_health = await calculate_system_health(
            metrics["avg_score"],
            metrics["total_analyses"]
        )
        
        drift_detected = await detect_drift(metrics["score_history"])
        
        if drift_detected:
            logger.warning("⚠️ DRIFT DETECTED in analysis scores!")
        
        return MonitoringResponse(
            total_analyses=metrics["total_analyses"],
            avg_score=metrics["avg_score"],
            system_health=system_health,
            drift_detected=drift_detected,
            score_history=metrics["score_history"],
            last_updated=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to fetch monitoring data: {e}")
        raise HTTPException(500, "Failed to retrieve monitoring data")
```

### Step 5: Error Handler Middleware (30 min)

Add to `app/main.py`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from app.utils.exceptions import PitchQuestException

@app.exception_handler(PitchQuestException)
async def pitchquest_exception_handler(request: Request, exc: PitchQuestException):
    """Handle custom exceptions gracefully"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )
```

### Step 6: Testing (2 hours)

**File: `tests/test_routes.py`**

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_video():
    # Create mock video file
    with open("test_video.mp4", "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("test_video.mp4", f, "video/mp4")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert data["status"] == "uploaded"

def test_upload_invalid_file():
    # Test with non-video file
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

# Add more tests...
```

Run tests:
```bash
pytest tests/ -v
```

### End of Day 2 Deliverable
- [ ] All 5 endpoints working (/upload, /analyze, /status, /results, /monitoring)
- [ ] Error handling everywhere
- [ ] Airflow integration tested
- [ ] API tests passing

---

## Day 3: Production Polish (4-6 hours)

### Step 1: Add Rate Limiting (1 hour)

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/upload")
@limiter.limit("5/minute")  # 5 uploads per minute
async def upload_video(request: Request, file: UploadFile = File(...)):
    # ... existing code
```

### Step 2: Request Validation (1 hour)

Add validators to all routes:

```python
from pydantic import validator

class AnalysisRequest(BaseModel):
    video_id: str
    
    @validator('video_id')
    def validate_video_id(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('Invalid video_id format')
        return v
```

### Step 3: API Documentation (1 hour)

Enhance OpenAPI docs in `main.py`:

```python
app = FastAPI(
    title="PitchQuest API",
    description="""
    ## PitchQuest - AI Pitch Analysis API
    
    Upload startup pitch videos and receive comprehensive AI-powered analysis.
    
    ### Features:
    - **Video Upload**: Secure video file uploads
    - **Analysis**: Multi-modal AI analysis (visual, audio, content)
    - **Status Tracking**: Real-time progress updates
    - **Results**: Detailed feedback and scoring
    - **Monitoring**: System health and drift detection
    
    ### Authentication:
    Currently no authentication required (add in production)
    
    ### Rate Limits:
    - Upload: 5 requests per minute
    - Other endpoints: 100 requests per minute
    """,
    version="1.0.0",
    contact={
        "name": "PitchQuest Team",
        "email": "team@pitchquest.com"
    }
)
```

### Step 4: Health Checks (1 hour)

Enhanced health check:

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health = {
        "status": "healthy",
        "service": "pitchquest-backend",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Check Supabase connection
    try:
        supabase.table("videos").select("id").limit(1).execute()
        health["checks"]["database"] = "healthy"
    except Exception as e:
        health["checks"]["database"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    # Check Airflow connection
    try:
        response = requests.get(f"{settings.AIRFLOW_URL}/health", timeout=5)
        health["checks"]["airflow"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        health["checks"]["airflow"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    return health
```

### Step 5: Dockerfile (30 min)

**File: `Dockerfile`**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create upload directory
RUN mkdir -p /tmp/pitchquest_uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6: Final Testing (2 hours)

Test all edge cases:
- [ ] Large file upload (>100MB)
- [ ] Corrupted video file
- [ ] Concurrent uploads
- [ ] Airflow connection failure
- [ ] Database connection failure
- [ ] Invalid video_id
- [ ] Rate limit exceeded

### End of Day 3 Deliverable
- [ ] Rate limiting works
- [ ] All validations in place
- [ ] Swagger docs complete
- [ ] Docker image builds
- [ ] All edge cases handled

---

## Quick Reference

### Start Backend
```bash
cd backend
source venv/bin/activate
python -m app.main
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Upload video
curl -X POST http://localhost:8000/api/upload \
  -F "file=@video.mp4"

# Trigger analysis
curl -X POST http://localhost:8000/api/analyze/VIDEO_ID

# Check status
curl http://localhost:8000/api/status/VIDEO_ID

# Get results
curl http://localhost:8000/api/results/VIDEO_ID

# Monitoring
curl http://localhost:8000/api/monitoring
```

### Common Issues

**Issue:** CORS errors from frontend  
**Fix:** Add frontend URL to CORS origins

**Issue:** Airflow connection refused  
**Fix:** Check AIRFLOW_URL in .env, ensure Airflow is running

**Issue:** Supabase auth error  
**Fix:** Verify SUPABASE_KEY is correct (anon key, not service role key)

**Issue:** File upload fails  
**Fix:** Check UPLOAD_DIR exists and has write permissions

---

## Success Criteria

- [ ] All endpoints return expected responses
- [ ] Error handling covers all failure modes
- [ ] API documentation complete
- [ ] Tests passing (>80% coverage)
- [ ] Docker image builds successfully
- [ ] Integration with Airflow working
- [ ] Integration with Supabase working

**You're now ready for frontend integration! 🚀**