from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.models import UploadResponse
from app.services.storage import storage
from app.database.supabase_client import supabase
import uuid
from datetime import datetime
from app.services.airflow_client import airflow_client

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    try:
        video_id = str(uuid.uuid4())
        
        # Save file
        file_path = await storage.save_video(file, video_id)
        
        # Save to Supabase
        supabase.table("videos").insert({
            "video_id": video_id,
            "filename": file.filename,
            "file_path": file_path,
            "status": "uploaded",
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
        
        return UploadResponse(
            video_id=video_id,
            filename=file.filename,
            status="uploaded",
            message="Upload successful"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze/{video_id}")
async def analyze_video(video_id: str):
    try:
        # Get video from DB
        result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video = result.data[0]
        
        # Trigger Airflow
        airflow_result = airflow_client.trigger_dag(video_id, video["file_path"])
        
        # Update status in DB
        supabase.table("videos").update({
            "status": "processing",
            "job_id": airflow_result["job_id"]
        }).eq("video_id", video_id).execute()
        
        return {
            "video_id": video_id,
            "job_id": airflow_result["job_id"],
            "status": "processing",
            "message": "Analysis started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/status/{video_id}")
async def get_status(video_id: str):
    # Get video from DB
    result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video = result.data[0]
    status = video["status"]
    
    if status != "processing":
        return {"status": status, "progress": 100 if status == "completed" else 0}
    
    job_id = video.get("job_id")
    airflow_status = airflow_client.get_dag_run_status(job_id)
    tasks = airflow_client.get_task_instances(job_id)
    
    if tasks:
        total = len(tasks)
        completed = sum(1 for t in tasks if t["state"] == "success")
        progress = int((completed / total) * 100)
        current = next((t["task_id"] for t in tasks if t["state"] == "running"), "Processing...")
    else:
        progress = 50
        current = "Processing..."
    
    return {
        "status": airflow_status["status"],
        "progress": progress,
        "current_task": current,
        "airflow_state": airflow_status["airflow_state"]
    }


@router.get("/results/{video_id}")
async def get_results(video_id: str):
    """Get completed analysis results"""
    # Check video status
    video_result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
    
    if not video_result.data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video = video_result.data[0]
    
    if video["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Analysis not ready. Status: {video['status']}")
    
    # Fetch results from analysis_results table
    results = supabase.table("analysis_results").select("*").eq("video_id", video_id).execute()
    
    if not results.data:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return results.data[0]


@router.get("/monitoring")
async def get_monitoring():
    """Get system monitoring metrics"""
    try:
        # Count total analyses
        videos = supabase.table("videos").select("*", count="exact").execute()
        total = videos.count
        
        # Get all results for average score
        results = supabase.table("analysis_results").select("overall_score").execute()
        scores = [r["overall_score"] for r in results.data]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Simple health check
        health = "healthy" if avg_score > 50 else "warning" if avg_score > 30 else "critical"
        
        return {
            "total_analyses": total,
            "avg_score": round(avg_score, 2),
            "system_health": health,
            "drift_detected": False,  # Placeholder
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))