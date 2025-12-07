from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import uuid
from datetime import datetime
import logging

# Import your pipeline functions
from pipeline.preprocessing import extract_frames, extract_audio, cleanup_temp_files
from pipeline.visual_analysis import analyze_pose_mediapipe, analyze_visual_gemini, combine_visual_analyses
from pipeline.audio_analysis import transcribe_with_whisper, analyze_emotion_wav2vec, compute_vocal_metrics, combine_audio_analyses
from pipeline.content_analysis import analyze_content_gemini
from pipeline.aggregation import aggregate_all_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Analysis API")

# Configure CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Video Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Endpoint to upload and analyze a video
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_extension = Path(video.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    video_filename = f"{analysis_id}_{video.filename}"
    video_path = UPLOAD_DIR / video_filename
    
    frames_dir = None
    audio_path = None
    
    try:
        # Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"Video saved: {video_path}")
        
        # ============================================
        # STEP 1: PREPROCESSING
        # ============================================
        logger.info("=== Step 1: Preprocessing ===")
        
        # Extract frames
        frame_data = extract_frames(str(video_path), video_id=analysis_id)
        frames_dir = frame_data['frames_dir']
        frame_paths = frame_data['frame_paths']
        
        # Extract audio
        audio_data = extract_audio(str(video_path), video_id=analysis_id)
        audio_path = audio_data['audio_path']
        
        # ============================================
        # STEP 2: VISUAL ANALYSIS
        # ============================================
        logger.info("=== Step 2: Visual Analysis ===")
        
        # MediaPipe pose analysis
        mediapipe_results = analyze_pose_mediapipe(frame_paths)
        
        # Gemini visual analysis
        gemini_results = analyze_visual_gemini(frame_paths)
        
        # Combine visual analyses
        visual_results = combine_visual_analyses(mediapipe_results, gemini_results)
        
        # ============================================
        # STEP 3: AUDIO ANALYSIS
        # ============================================
        logger.info("=== Step 3: Audio Analysis ===")
        
        # Transcribe with Whisper
        transcription = transcribe_with_whisper(audio_path)
        
        # Analyze emotion
        emotion = analyze_emotion_wav2vec(audio_path)
        
        # Compute vocal metrics
        vocal_metrics = compute_vocal_metrics(
            transcription['transcript'], 
            transcription['duration']
        )
        
        # Combine audio analyses
        audio_results = combine_audio_analyses(transcription, emotion, vocal_metrics)
        
        # ============================================
        # STEP 4: CONTENT ANALYSIS
        # ============================================
        logger.info("=== Step 4: Content Analysis ===")
        
        content_results = analyze_content_gemini(transcription['transcript'])
        
        # ============================================
        # STEP 5: AGGREGATE RESULTS
        # ============================================
        logger.info("=== Step 5: Aggregation ===")
        
        final_results = aggregate_all_results(
            visual_results,
            audio_results,
            content_results
        )
        
        # ============================================
        # STEP 6: SAVE AND CLEANUP
        # ============================================
        
        # Save results to file
        output_file = OUTPUT_DIR / f"{analysis_id}_results.json"
        import json
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        
        # Cleanup temporary files
        cleanup_temp_files(frames_dir, audio_path)
        
        # Clean up uploaded video (optional - comment out to keep)
        # os.remove(video_path)
        
        logger.info(f"✅ Analysis complete for {analysis_id}")
        
        return JSONResponse(content={
            "status": "success",
            "analysis_id": analysis_id,
            "results": final_results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        # Clean up on error
        if frames_dir:
            cleanup_temp_files(frames_dir, audio_path)
        if video_path.exists():
            os.remove(video_path)
        
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """
    Retrieve results for a specific analysis
    """
    output_file = OUTPUT_DIR / f"{analysis_id}_results.json"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    import json
    with open(output_file, "r") as f:
        results = json.load(f)
    
    return JSONResponse(content=results)

@app.delete("/api/results/{analysis_id}")
async def delete_results(analysis_id: str):
    """
    Delete analysis results
    """
    output_file = OUTPUT_DIR / f"{analysis_id}_results.json"
    video_files = list(UPLOAD_DIR.glob(f"{analysis_id}_*"))
    
    deleted = []
    
    if output_file.exists():
        os.remove(output_file)
        deleted.append("results")
    
    for video_file in video_files:
        os.remove(video_file)
        deleted.append("video")
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {"status": "deleted", "items": deleted}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)