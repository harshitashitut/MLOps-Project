"""
Result Mapper: Transform aggregation output to database schema
Handles data normalization and validation before DB save
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultMapper:
    """Maps aggregation results to database schema format"""
    
    @staticmethod
    def normalize_aggregation_output(raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize aggregation output to consistent format
        Handles both wrapped {"results": {...}} and unwrapped formats
        
        Args:
            raw_results: Output from aggregation.py
            
        Returns:
            Normalized format with "results" wrapper
        """
        # If already wrapped, use as-is
        if "results" in raw_results and isinstance(raw_results["results"], dict):
            return raw_results
        
        # If unwrapped, wrap it
        # This handles the case where aggregation returns flat structure
        return {"results": raw_results}
    
    @staticmethod
    def map_to_db_schema(
        video_id: str,
        user_id: str,
        aggregation_output: Dict[str, Any],
        processing_time_seconds: float = None
    ) -> Dict[str, Any]:
        """
        Map aggregation output to database schema
        
        DB Schema:
        - Denormalized columns for fast queries (overall_score, content_score, etc.)
        - Full JSON in 'results' column for flexibility
        
        Args:
            video_id: UUID of video
            user_id: UUID of user
            aggregation_output: Normalized aggregation results
            processing_time_seconds: Total processing time
            
        Returns:
            Dict ready for database insertion
        """
        try:
            # Normalize first (handles wrapped/unwrapped)
            normalized = ResultMapper.normalize_aggregation_output(aggregation_output)
            results_obj = normalized.get("results", {})
            
            # Extract category scores safely
            category_scores = results_obj.get("category_scores", {})
            key_metrics = results_obj.get("key_metrics", {})
            
            # Helper function for safe extraction
            def safe_extract(data: Dict, *keys, default=None):
                """Safely navigate nested dict"""
                for key in keys:
                    if isinstance(data, dict):
                        data = data.get(key, {})
                    else:
                        return default
                return data if data != {} else default
            
            # Build database record
            db_record = {
                # Primary identifiers
                "video_id": video_id,
                "user_id": user_id,
                
                # Denormalized scores (for fast queries/analytics)
                "overall_score": int(safe_extract(results_obj, "overall_score", default=0)),
                "performance_level": safe_extract(results_obj, "performance_level", default="beginner"),
                
                # Category scores
                "content_score": int(safe_extract(category_scores, "content", "score", default=0)),
                "vocal_delivery_score": int(safe_extract(category_scores, "vocal_delivery", "score", default=0)),
                "visual_presentation_score": int(safe_extract(category_scores, "visual_presentation", "score", default=0)),
                "tone_emotion_score": int(safe_extract(category_scores, "tone_emotion", "score", default=0)),
                
                # Key metrics
                "speech_duration_seconds": float(safe_extract(key_metrics, "speech_duration_seconds", default=0.0)),
                "words_per_minute": float(safe_extract(key_metrics, "words_per_minute", default=0.0)),
                "filler_words_count": int(safe_extract(key_metrics, "filler_words_count", default=0)),
                "dominant_emotion": safe_extract(key_metrics, "dominant_emotion", default="neutral"),
                
                # Full results as JSONB (for flexibility)
                "results": results_obj,
                
                # Metadata
                "model_version": safe_extract(results_obj, "model_used", default="unknown"),
                "processing_time_seconds": processing_time_seconds,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Validate required fields
            ResultMapper._validate_db_record(db_record)
            
            logger.info(f"✅ Mapped results for {video_id}: Overall={db_record['overall_score']}/100")
            
            return db_record
            
        except Exception as e:
            logger.error(f"❌ Failed to map results: {e}", exc_info=True)
            raise ValueError(f"Result mapping failed: {e}")
    
    @staticmethod
    def _validate_db_record(record: Dict[str, Any]) -> None:
        """
        Validate DB record has required fields and correct types
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["video_id", "user_id", "overall_score", "results"]
        
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate score ranges (0-100)
        score_fields = [
            "overall_score", "content_score", "vocal_delivery_score",
            "visual_presentation_score", "tone_emotion_score"
        ]
        
        for field in score_fields:
            score = record.get(field, 0)
            if not (0 <= score <= 100):
                logger.warning(f"⚠️ Score out of range: {field}={score}")
                # Clamp to valid range
                record[field] = max(0, min(100, score))
    
    @staticmethod
    def extract_emotion_timeline(
        video_id: str,
        user_id: str,
        aggregation_output: Dict[str, Any]
    ) -> list:
        """
        Extract emotion timeline data for separate table
        
        Args:
            video_id: UUID of video
            user_id: UUID of user
            aggregation_output: Normalized aggregation results
            
        Returns:
            List of emotion timeline records
        """
        try:
            normalized = ResultMapper.normalize_aggregation_output(aggregation_output)
            results_obj = normalized.get("results", {})
            emotion_timeline = results_obj.get("emotion_timeline", [])
            
            if not emotion_timeline:
                logger.info(f"No emotion timeline data for {video_id}")
                return []
            
            # Map to DB format
            timeline_records = []
            for point in emotion_timeline:
                timeline_records.append({
                    "video_id": video_id,
                    "user_id": user_id,
                    "timestamp_seconds": float(point.get("timestamp_seconds", 0)),
                    "emotion": point.get("emotion", "neutral"),
                    "confidence": float(point.get("confidence", 0))
                })
            
            logger.info(f"✅ Extracted {len(timeline_records)} emotion timeline points")
            return timeline_records
            
        except Exception as e:
            logger.error(f"❌ Failed to extract emotion timeline: {e}")
            return []
    
    @staticmethod
    def create_summary(db_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create quick summary for API responses (without full JSONB)
        
        Args:
            db_record: Full database record
            
        Returns:
            Lightweight summary dict
        """
        return {
            "video_id": db_record["video_id"],
            "overall_score": db_record["overall_score"],
            "performance_level": db_record["performance_level"],
            "category_scores": {
                "content": db_record["content_score"],
                "vocal_delivery": db_record["vocal_delivery_score"],
                "visual_presentation": db_record["visual_presentation_score"],
                "tone_emotion": db_record["tone_emotion_score"]
            },
            "key_metrics": {
                "words_per_minute": db_record["words_per_minute"],
                "filler_words_count": db_record["filler_words_count"],
                "dominant_emotion": db_record["dominant_emotion"]
            }
        }


# Example usage for testing
if __name__ == "__main__":
    # Test with sample aggregation output
    sample_output = {
        "overall_score": 7.0,
        "performance_level": "good",
        "category_scores": {
            "visual_presentation": {"score": 3, "rating": "good"},
            "vocal_delivery": {"score": 6.3, "rating": "fair"},
            "content": {"score": 4, "rating": "good"},
            "tone_emotion": {"score": 7.0, "rating": "good"}
        },
        "key_metrics": {
            "words_per_minute": 155.1,
            "filler_words_count": 1,
            "speaking_pace": "optimal",
            "dominant_emotion": "surprised"
        },
        "timestamp": "2025-12-07T00:17:14.374105Z",
        "model_used": "gemini-2.5-pro"
    }
    
    mapper = ResultMapper()
    db_record = mapper.map_to_db_schema(
        video_id="test-123",
        user_id="user-456",
        aggregation_output=sample_output,
        processing_time_seconds=45.2
    )
    
    print("DB Record:")
    print(f"Overall Score: {db_record['overall_score']}")
    print(f"Content Score: {db_record['content_score']}")
    print(f"Vocal Score: {db_record['vocal_delivery_score']}")


    