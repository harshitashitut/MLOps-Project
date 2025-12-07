from supabase import create_client, Client
import os
import json
from datetime import datetime

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_results_to_db(video_id: str, results: dict):
    """Save analysis results to Supabase"""
    try:
        # Prepare database record
        db_record = {
            "video_id": video_id,
            "overall_score": results["results"]["overall_score"],
            "performance_level": results["results"]["performance_level"],
            "results": results["results"],  # Store entire results object as JSONB
            "timestamp": results["timestamp"]
        }
        
        # Insert into database
        response = supabase.table("analysis_results").insert(db_record).execute()
        
        print(f"✅ Results saved to database for video: {video_id}")

        update_video_status(video_id, "completed")
        return response
        
    except Exception as e:
        print(f"❌ Failed to save to database: {e}")
        raise

def update_video_status(video_id: str, status: str, job_id: str = None):
    """Update video processing status"""
    try:
        data = {"status": status}
        if job_id:
            data["job_id"] = job_id
            
        supabase.table("videos").update(data).eq("video_id", video_id).execute()
        print(f"✅ Updated video status: {video_id} -> {status}")
        
    except Exception as e:
        print(f"❌ Failed to update status: {e}")