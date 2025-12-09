# from supabase import create_client, Client
# from datetime import datetime
# import json
# import os
# from typing import Dict, Any, Optional
# import logging

# logger = logging.getLogger(__name__)

# # Initialize Supabase client
# def get_supabase_client() -> Client:
#     """Initialize Supabase client"""
#     try:
#         supabase_url = os.getenv("SUPABASE_URL")
#         supabase_key = os.getenv("SUPABASE_KEY")
        
#         if not supabase_url or not supabase_key:
#             raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
#         client = create_client(supabase_url, supabase_key)
#         logger.info("✅ Supabase client initialized")
#         return client
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize Supabase: {e}")
#         raise

# supabase: Client = get_supabase_client()

# # ==================== USER OPERATIONS ====================

# def get_or_create_user(user_id: str, email: str, full_name: str = None) -> Dict:
#     """Get existing user or create new one"""
#     try:
#         # Check if user exists
#         result = supabase.table("users").select("*").eq("id", user_id).execute()
        
#         if result.data:
#             return result.data[0]
        
#         # Create new user
#         user_data = {
#             "id": user_id,
#             "email": email,
#             "full_name": full_name or "User"
#         }
#         result = supabase.table("users").insert(user_data).execute()
#         logger.info(f"✅ Created new user: {email}")
#         return result.data[0]
#     except Exception as e:
#         logger.error(f"❌ Failed to get/create user: {e}")
#         raise

# # ==================== VIDEO OPERATIONS ====================

# def create_video_record(video_id: str, user_id: str, filename: str, 
#                         file_path: str, file_size_mb: float) -> Dict:
#     """Create video record in database"""
#     try:
#         data = {
#             "video_id": video_id,
#             "user_id": user_id,
#             "filename": filename,
#             "file_path": file_path,
#             "file_size_mb": file_size_mb,
#             "status": "uploaded",
#             "uploaded_at": datetime.utcnow().isoformat()
#         }
#         result = supabase.table("videos").insert(data).execute()
#         logger.info(f"✅ Created video record: {video_id}")
#         return result.data[0]
#     except Exception as e:
#         logger.error(f"❌ Failed to create video record: {e}")
#         raise


# def update_video_status(video_id: str, status: str):
#     """Update video processing status"""
#     try:
#         data = {
#             "status": status,
#             "updated_at": datetime.utcnow().isoformat()
#         }
        
#         # Add timestamps based on status
#         if status == "processing":
#             data["processing_started_at"] = datetime.utcnow().isoformat()
#         elif status == "completed":
#             data["processing_completed_at"] = datetime.utcnow().isoformat()
        
#         supabase.table("videos").update(data).eq("video_id", video_id).execute()
#         logger.info(f"📊 Status updated: {status}")
#     except Exception as e:
#         logger.error(f"❌ Failed to update status: {e}")

# def update_video_progress(video_id: str, progress: int, current_step: str):
#     """Update video processing progress"""
#     try:
#         data = {
#             "progress_percentage": progress,
#             "current_step": current_step,
#             "updated_at": datetime.utcnow().isoformat()
#         }
#         supabase.table("videos").update(data).eq("video_id", video_id).execute()
#         logger.info(f"📊 Progress: {progress}% - {current_step}")
#     except Exception as e:
#         logger.error(f"❌ Failed to update progress: {e}")

# def get_video_by_id(video_id: str) -> Optional[Dict]:
#     """Fetch video record from database"""
#     try:
#         result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
#         if result.data:
#             return result.data[0]
#         return None
#     except Exception as e:
#         logger.error(f"❌ Failed to fetch video: {e}")
#         return None

# # ==================== RESULTS OPERATIONS ====================

# def save_results_to_db(video_id: str, user_id: str, results: Dict[str, Any]) -> None:
#     """Save analysis results with denormalized scores"""
#     try:
#         logger.info(f"💾 Saving results to database for video: {video_id}")
        
#         # Navigate nested JSON safely
#         results_obj = results.get("results", {})
#         category_scores = results_obj.get("category_scores", {})
#         key_metrics = results_obj.get("key_metrics", {})
        
#         # Helper for safe extraction
#         def safe_get(data, *keys, default=0):
#             for key in keys:
#                 if isinstance(data, dict):
#                     data = data.get(key, {})
#                 else:
#                     return default
#             return data if data != {} else default
        
#         # Build database record
#         db_record = {
#             "video_id": video_id,
#             "user_id": user_id,
#             "overall_score": safe_get(results_obj, "overall_score", default=0),
#             "performance_level": safe_get(results_obj, "performance_level", default="beginner"),
#             "content_score": safe_get(category_scores, "content", "score", default=0),
#             "vocal_delivery_score": safe_get(category_scores, "vocal_delivery", "score", default=0),
#             "visual_presentation_score": safe_get(category_scores, "visual_presentation", "score", default=0),
#             "tone_emotion_score": safe_get(category_scores, "tone_emotion", "score", default=0),
#             "speech_duration_seconds": safe_get(key_metrics, "speech_duration_seconds", default=0.0),
#             "words_per_minute": safe_get(key_metrics, "words_per_minute", default=0.0),
#             "filler_words_count": safe_get(key_metrics, "filler_words_count", default=0),
#             "dominant_emotion": safe_get(category_scores, "tone_emotion", "dominant_emotion", default="neutral"),
#             "results": json.dumps(results),
#             "model_version": safe_get(results_obj, "model_used", default="unknown"),
#             "created_at": datetime.utcnow().isoformat()
#         }
        
#         result = supabase.table("analysis_results").insert(db_record).execute()
        
#         if result.data:
#             logger.info(f"✅ Results saved - Overall: {db_record['overall_score']}/100")
#             update_video_status(video_id, "completed")
            
#             # Save emotion timeline
#             emotion_timeline = results_obj.get("emotion_timeline", [])
#             for emotion_point in emotion_timeline:
#                 supabase.table("emotion_timeline").insert({
#                     "video_id": video_id,
#                     "user_id": user_id,
#                     "timestamp_seconds": emotion_point.get("timestamp_seconds", 0),
#                     "emotion": emotion_point.get("emotion", "neutral"),
#                     "confidence": emotion_point.get("confidence", 0)
#                 }).execute()
#         else:
#             raise Exception("Database insert failed")
            
#     except Exception as e:
#         logger.error(f"❌ Failed to save results: {e}", exc_info=True)
#         update_video_status(video_id, "failed")
#         raise

# def get_results_from_db(video_id: str) -> Optional[Dict]:
#     """Retrieve analysis results from database"""
#     try:
#         result = supabase.table("analysis_results").select("*").eq("video_id", video_id).execute()
#         if result.data:
#             record = result.data[0]
#             # Parse JSON string back to dict
#             if isinstance(record["results"], str):
#                 record["results"] = json.loads(record["results"])
#             return record
#         return None
#     except Exception as e:
#         logger.error(f"❌ Failed to fetch results: {e}")
#         return None



# def get_videos_by_user(user_id: str) -> list:
#     """Fetch all videos for a user with their analysis scores"""
#     try:
#         # Get videos
#         videos_result = supabase.table("videos")\
#             .select("video_id, filename, status, uploaded_at, processing_completed_at")\
#             .eq("user_id", user_id)\
#             .order("uploaded_at", desc=True)\
#             .execute()
        
#         if not videos_result.data:
#             return []
        
#         # Get scores for each video
#         videos_with_scores = []
#         for video in videos_result.data:
#             video_data = {
#                 "video_id": video["video_id"],
#                 "filename": video["filename"],
#                 "status": video["status"],
#                 "uploaded_at": video["uploaded_at"],
#                 "processing_completed_at": video.get("processing_completed_at"),
#             }
            
#             # Get scores if completed
#             if video["status"] == "completed":
#                 results = supabase.table("analysis_results")\
#                     .select("overall_score, performance_level, content_score, vocal_delivery_score, visual_presentation_score, tone_emotion_score")\
#                     .eq("video_id", video["video_id"])\
#                     .execute()
                
#                 if results.data:
#                     video_data.update(results.data[0])
            
#             videos_with_scores.append(video_data)
        
#         logger.info(f"📋 Found {len(videos_with_scores)} videos for user {user_id[:8]}")
#         return videos_with_scores
#     except Exception as e:
#         logger.error(f"❌ Failed to fetch user videos: {e}")
#         return []



# def get_admin_analytics() -> dict:
#     """Get overall platform analytics for admin dashboard"""
#     try:
#         # Total users
#         users_result = supabase.table("users").select("id, email, full_name, created_at", count="exact").execute()
#         total_users = users_result.count or len(users_result.data)
        
#         # Total videos
#         videos_result = supabase.table("videos").select("id, status", count="exact").execute()
#         total_videos = videos_result.count or len(videos_result.data)
#         completed_videos = len([v for v in videos_result.data if v["status"] == "completed"])
        
#         # Average scores
#         results = supabase.table("analysis_results").select(
#             "overall_score, content_score, vocal_delivery_score, visual_presentation_score, tone_emotion_score"
#         ).execute()
        
#         if results.data:
#             avg_score = sum(r["overall_score"] or 0 for r in results.data) / len(results.data)
#             avg_content = sum(r["content_score"] or 0 for r in results.data) / len(results.data)
#             avg_vocal = sum(r["vocal_delivery_score"] or 0 for r in results.data) / len(results.data)
#             avg_visual = sum(r["visual_presentation_score"] or 0 for r in results.data) / len(results.data)
#             avg_emotion = sum(r["tone_emotion_score"] or 0 for r in results.data) / len(results.data)
            
#             # Score distribution
#             distribution = [
#                 {"range": "0-30", "count": len([r for r in results.data if (r["overall_score"] or 0) <= 30])},
#                 {"range": "31-50", "count": len([r for r in results.data if 30 < (r["overall_score"] or 0) <= 50])},
#                 {"range": "51-70", "count": len([r for r in results.data if 50 < (r["overall_score"] or 0) <= 70])},
#                 {"range": "71-100", "count": len([r for r in results.data if (r["overall_score"] or 0) > 70])},
#             ]
#         else:
#             avg_score = avg_content = avg_vocal = avg_visual = avg_emotion = 0
#             distribution = []
        
#         # Recent users with video counts
#         recent_users = []
#         for user in users_result.data[:10]:
#             user_videos = supabase.table("videos").select("id", count="exact").eq("user_id", user["id"]).execute()
#             recent_users.append({
#                 "email": user["email"],
#                 "full_name": user.get("full_name"),
#                 "created_at": user["created_at"],
#                 "video_count": user_videos.count or len(user_videos.data)
#             })
        
#         return {
#             "total_users": total_users,
#             "total_videos": total_videos,
#             "completed_videos": completed_videos,
#             "average_score": avg_score,
#             "avg_content": avg_content,
#             "avg_vocal": avg_vocal,
#             "avg_visual": avg_visual,
#             "avg_emotion": avg_emotion,
#             "score_distribution": distribution,
#             "recent_users": recent_users
#         }
#     except Exception as e:
#         logger.error(f"❌ Failed to get admin analytics: {e}")
#         return {}


# def check_user_is_admin(user_id: str) -> bool:
#     """Check if user has admin role"""
#     try:
#         result = supabase.table("users").select("role").eq("id", user_id).single().execute()
#         return result.data and result.data.get("role") == "admin"
#     except Exception as e:
#         logger.error(f"❌ Failed to check admin status: {e}")
#         return False

# # ==================== TEST CONNECTION ====================

# def test_connection() -> bool:
#     """Test database connection"""
#     try:
#         result = supabase.table("videos").select("id").limit(1).execute()
#         logger.info("✅ Database connection successful")
#         return True
#     except Exception as e:
#         logger.error(f"❌ Database connection failed: {e}")
#         return False


from supabase import create_client, Client
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ==================== SUPABASE CLIENT INITIALIZATION ====================

def get_supabase_client() -> Optional[Client]:
    """
    Initialize Supabase client (or return None in CI mode)
    
    CI Mode is enabled when:
    - CI_MODE environment variable is set to 'true'
    - SUPABASE_URL contains 'mock'
    
    This allows the backend to start without a real database for testing
    """
    try:
        # Check if running in CI/test mode
        CI_MODE = os.getenv('CI_MODE', 'false').lower() == 'true'
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        
        # Skip Supabase in CI mode
        if CI_MODE or 'mock' in supabase_url.lower():
            logger.warning("🧪 CI/TEST mode detected - Supabase client disabled")
            logger.info("✅ Backend will run without database for testing")
            return None
        
        # Normal production mode
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized")
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase: {e}")
        raise

# Initialize client (will be None in CI mode)
supabase: Optional[Client] = get_supabase_client()

# ==================== USER OPERATIONS ====================

def get_or_create_user(user_id: str, email: str, full_name: str = None) -> Dict:
    """Get existing user or create new one"""
    # CI mode - return mock user
    if supabase is None:
        logger.info("🧪 CI mode - returning mock user")
        return {
            "id": user_id,
            "email": email,
            "full_name": full_name or "CI Test User"
        }
    
    try:
        # Check if user exists
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        
        if result.data:
            return result.data[0]
        
        # Create new user
        user_data = {
            "id": user_id,
            "email": email,
            "full_name": full_name or "User"
        }
        result = supabase.table("users").insert(user_data).execute()
        logger.info(f"✅ Created new user: {email}")
        return result.data[0]
    except Exception as e:
        logger.error(f"❌ Failed to get/create user: {e}")
        raise

# ==================== VIDEO OPERATIONS ====================

def create_video_record(video_id: str, user_id: str, filename: str, 
                        file_path: str, file_size_mb: float) -> Dict:
    """Create video record in database"""
    # CI mode - skip database insert
    if supabase is None:
        logger.info("🧪 CI mode - skipping video record creation")
        return {
            "video_id": video_id,
            "user_id": user_id,
            "filename": filename,
            "status": "uploaded"
        }
    
    try:
        data = {
            "video_id": video_id,
            "user_id": user_id,
            "filename": filename,
            "file_path": file_path,
            "file_size_mb": file_size_mb,
            "status": "uploaded",
            "uploaded_at": datetime.utcnow().isoformat()
        }
        result = supabase.table("videos").insert(data).execute()
        logger.info(f"✅ Created video record: {video_id}")
        return result.data[0]
    except Exception as e:
        logger.error(f"❌ Failed to create video record: {e}")
        raise


def update_video_status(video_id: str, status: str):
    """Update video processing status"""
    # CI mode - skip database update
    if supabase is None:
        logger.info(f"🧪 CI mode - would update status to: {status}")
        return
    
    try:
        data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add timestamps based on status
        if status == "processing":
            data["processing_started_at"] = datetime.utcnow().isoformat()
        elif status == "completed":
            data["processing_completed_at"] = datetime.utcnow().isoformat()
        
        supabase.table("videos").update(data).eq("video_id", video_id).execute()
        logger.info(f"📊 Status updated: {status}")
    except Exception as e:
        logger.error(f"❌ Failed to update status: {e}")


def update_video_progress(video_id: str, progress: int, current_step: str):
    """Update video processing progress"""
    # CI mode - skip database update
    if supabase is None:
        logger.info(f"🧪 CI mode - would update progress: {progress}% - {current_step}")
        return
    
    try:
        data = {
            "progress_percentage": progress,
            "current_step": current_step,
            "updated_at": datetime.utcnow().isoformat()
        }
        supabase.table("videos").update(data).eq("video_id", video_id).execute()
        logger.info(f"📊 Progress: {progress}% - {current_step}")
    except Exception as e:
        logger.error(f"❌ Failed to update progress: {e}")


def get_video_by_id(video_id: str) -> Optional[Dict]:
    """Fetch video record from database"""
    # CI mode - return mock video
    if supabase is None:
        logger.info("🧪 CI mode - returning mock video")
        return {
            "video_id": video_id,
            "status": "completed",
            "uploaded_at": datetime.utcnow().isoformat()
        }
    
    try:
        result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch video: {e}")
        return None

# ==================== RESULTS OPERATIONS ====================

def save_results_to_db(video_id: str, user_id: str, results: Dict[str, Any]) -> None:
    """Save analysis results with denormalized scores"""
    # CI mode - skip database save
    if supabase is None:
        logger.info("🧪 CI mode - skipping results save to database")
        return
    
    try:
        logger.info(f"💾 Saving results to database for video: {video_id}")
        
        # Navigate nested JSON safely
        results_obj = results.get("results", {})
        category_scores = results_obj.get("category_scores", {})
        key_metrics = results_obj.get("key_metrics", {})
        
        # Helper for safe extraction
        def safe_get(data, *keys, default=0):
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key, {})
                else:
                    return default
            return data if data != {} else default
        
        # Build database record
        db_record = {
            "video_id": video_id,
            "user_id": user_id,
            "overall_score": safe_get(results_obj, "overall_score", default=0),
            "performance_level": safe_get(results_obj, "performance_level", default="beginner"),
            "content_score": safe_get(category_scores, "content", "score", default=0),
            "vocal_delivery_score": safe_get(category_scores, "vocal_delivery", "score", default=0),
            "visual_presentation_score": safe_get(category_scores, "visual_presentation", "score", default=0),
            "tone_emotion_score": safe_get(category_scores, "tone_emotion", "score", default=0),
            "speech_duration_seconds": safe_get(key_metrics, "speech_duration_seconds", default=0.0),
            "words_per_minute": safe_get(key_metrics, "words_per_minute", default=0.0),
            "filler_words_count": safe_get(key_metrics, "filler_words_count", default=0),
            "dominant_emotion": safe_get(category_scores, "tone_emotion", "dominant_emotion", default="neutral"),
            "results": json.dumps(results),
            "model_version": safe_get(results_obj, "model_used", default="unknown"),
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("analysis_results").insert(db_record).execute()
        
        if result.data:
            logger.info(f"✅ Results saved - Overall: {db_record['overall_score']}/100")
            update_video_status(video_id, "completed")
            
            # Save emotion timeline
            emotion_timeline = results_obj.get("emotion_timeline", [])
            for emotion_point in emotion_timeline:
                supabase.table("emotion_timeline").insert({
                    "video_id": video_id,
                    "user_id": user_id,
                    "timestamp_seconds": emotion_point.get("timestamp_seconds", 0),
                    "emotion": emotion_point.get("emotion", "neutral"),
                    "confidence": emotion_point.get("confidence", 0)
                }).execute()
        else:
            raise Exception("Database insert failed")
            
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}", exc_info=True)
        update_video_status(video_id, "failed")
        raise


def get_results_from_db(video_id: str) -> Optional[Dict]:
    """Retrieve analysis results from database"""
    # CI mode - return None
    if supabase is None:
        logger.info("🧪 CI mode - no results in database")
        return None
    
    try:
        result = supabase.table("analysis_results").select("*").eq("video_id", video_id).execute()
        if result.data:
            record = result.data[0]
            # Parse JSON string back to dict
            if isinstance(record["results"], str):
                record["results"] = json.loads(record["results"])
            return record
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch results: {e}")
        return None


def get_videos_by_user(user_id: str) -> list:
    """Fetch all videos for a user with their analysis scores"""
    # CI mode - return empty list
    if supabase is None:
        logger.info("🧪 CI mode - returning empty video list")
        return []
    
    try:
        # Get videos
        videos_result = supabase.table("videos")\
            .select("video_id, filename, status, uploaded_at, processing_completed_at")\
            .eq("user_id", user_id)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        if not videos_result.data:
            return []
        
        # Get scores for each video
        videos_with_scores = []
        for video in videos_result.data:
            video_data = {
                "video_id": video["video_id"],
                "filename": video["filename"],
                "status": video["status"],
                "uploaded_at": video["uploaded_at"],
                "processing_completed_at": video.get("processing_completed_at"),
            }
            
            # Get scores if completed
            if video["status"] == "completed":
                results = supabase.table("analysis_results")\
                    .select("overall_score, performance_level, content_score, vocal_delivery_score, visual_presentation_score, tone_emotion_score")\
                    .eq("video_id", video["video_id"])\
                    .execute()
                
                if results.data:
                    video_data.update(results.data[0])
            
            videos_with_scores.append(video_data)
        
        logger.info(f"📋 Found {len(videos_with_scores)} videos for user {user_id[:8]}")
        return videos_with_scores
    except Exception as e:
        logger.error(f"❌ Failed to fetch user videos: {e}")
        return []


def get_admin_analytics() -> dict:
    """Get overall platform analytics for admin dashboard"""
    # CI mode - return empty analytics
    if supabase is None:
        logger.info("🧪 CI mode - returning empty analytics")
        return {
            "total_users": 0,
            "total_videos": 0,
            "completed_videos": 0,
            "average_score": 0
        }
    
    try:
        # Total users
        users_result = supabase.table("users").select("id, email, full_name, created_at", count="exact").execute()
        total_users = users_result.count or len(users_result.data)
        
        # Total videos
        videos_result = supabase.table("videos").select("id, status", count="exact").execute()
        total_videos = videos_result.count or len(videos_result.data)
        completed_videos = len([v for v in videos_result.data if v["status"] == "completed"])
        
        # Average scores
        results = supabase.table("analysis_results").select(
            "overall_score, content_score, vocal_delivery_score, visual_presentation_score, tone_emotion_score"
        ).execute()
        
        if results.data:
            avg_score = sum(r["overall_score"] or 0 for r in results.data) / len(results.data)
            avg_content = sum(r["content_score"] or 0 for r in results.data) / len(results.data)
            avg_vocal = sum(r["vocal_delivery_score"] or 0 for r in results.data) / len(results.data)
            avg_visual = sum(r["visual_presentation_score"] or 0 for r in results.data) / len(results.data)
            avg_emotion = sum(r["tone_emotion_score"] or 0 for r in results.data) / len(results.data)
            
            # Score distribution
            distribution = [
                {"range": "0-30", "count": len([r for r in results.data if (r["overall_score"] or 0) <= 30])},
                {"range": "31-50", "count": len([r for r in results.data if 30 < (r["overall_score"] or 0) <= 50])},
                {"range": "51-70", "count": len([r for r in results.data if 50 < (r["overall_score"] or 0) <= 70])},
                {"range": "71-100", "count": len([r for r in results.data if (r["overall_score"] or 0) > 70])},
            ]
        else:
            avg_score = avg_content = avg_vocal = avg_visual = avg_emotion = 0
            distribution = []
        
        # Recent users with video counts
        recent_users = []
        for user in users_result.data[:10]:
            user_videos = supabase.table("videos").select("id", count="exact").eq("user_id", user["id"]).execute()
            recent_users.append({
                "email": user["email"],
                "full_name": user.get("full_name"),
                "created_at": user["created_at"],
                "video_count": user_videos.count or len(user_videos.data)
            })
        
        return {
            "total_users": total_users,
            "total_videos": total_videos,
            "completed_videos": completed_videos,
            "average_score": avg_score,
            "avg_content": avg_content,
            "avg_vocal": avg_vocal,
            "avg_visual": avg_visual,
            "avg_emotion": avg_emotion,
            "score_distribution": distribution,
            "recent_users": recent_users
        }
    except Exception as e:
        logger.error(f"❌ Failed to get admin analytics: {e}")
        return {}


def check_user_is_admin(user_id: str) -> bool:
    """Check if user has admin role"""
    # CI mode - return False
    if supabase is None:
        logger.info("🧪 CI mode - admin check returns False")
        return False
    
    try:
        result = supabase.table("users").select("role").eq("id", user_id).single().execute()
        return result.data and result.data.get("role") == "admin"
    except Exception as e:
        logger.error(f"❌ Failed to check admin status: {e}")
        return False

# ==================== TEST CONNECTION ====================

def test_connection() -> bool:
    """Test database connection"""
    # CI mode - always return True (pretend healthy)
    if supabase is None:
        logger.info("🧪 CI mode - database test skipped (returns True)")
        return True
    
    try:
        result = supabase.table("videos").select("id").limit(1).execute()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False