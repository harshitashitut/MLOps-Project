"""
Aggregation: Combine all analyses into comprehensive dashboard report
Uses Gemini Pro for complex reasoning and detailed feedback generation
"""
import logging
import json
from typing import Dict
from datetime import datetime
from utils.config import config, load_prompt
from utils.error_handlers import PipelineError
from utils.gemini_client import gemini_client

logger = logging.getLogger(__name__)


def aggregate_all_results(
    visual_analysis: Dict,
    audio_analysis: Dict,
    content_analysis: Dict,
    **context
) -> Dict:
    """
    Combine all analyses using Gemini Pro for comprehensive evaluation
    
    Args:
        visual_analysis: Combined visual + pose data
        audio_analysis: Combined transcription + emotion + vocal metrics
        content_analysis: Speech content analysis
        
    Returns:
        Comprehensive dashboard-ready report with scores, feedback, recommendations
    """
    try:
        logger.info("Starting final aggregation with Gemini Pro...")
        
        # Extract data from each analysis
        visual_data = _extract_visual_data(visual_analysis)
        audio_data = _extract_audio_data(audio_analysis)
        content_data = _extract_content_data(content_analysis)
        
        # Build comprehensive input for Gemini
        aggregation_prompt = load_prompt('aggregation')
        
        input_data = {
            "visual_analysis": visual_data,
            "audio_analysis": audio_data,
            "content_analysis": content_data
        }
        
        full_prompt = f"""{aggregation_prompt}

=== INPUT DATA ===
{json.dumps(input_data, indent=2)}

=== END INPUT DATA ===

Analyze this data and create a comprehensive dashboard report following the exact JSON structure specified."""
        
        # Call Gemini Pro for aggregation (uses config.AGGREGATION_MODEL)
        logger.info(f"Calling Gemini ({config.AGGREGATION_MODEL}) for aggregation...")
        aggregated_result = gemini_client.generate_content(full_prompt)
        
        # Add metadata
        aggregated_result['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        aggregated_result['model_used'] = config.AGGREGATION_MODEL
        aggregated_result['processing_complete'] = True
        
        logger.info(f"✅ Aggregation complete. Overall score: {aggregated_result.get('overall_score', 0)}")
        
        return aggregated_result
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise PipelineError(f"Final aggregation failed: {e}")


def _extract_visual_data(visual_analysis: Dict) -> Dict:
    """Extract relevant visual metrics for aggregation"""
    visual_pres = visual_analysis.get('visual_presentation', {})
    
    return {
        "posture_score": visual_pres.get('posture_score', 0),
        "gesture_score": visual_pres.get('gesture_score', 0),
        "eye_contact_score": visual_pres.get('eye_contact_score', 0),
        "facial_expressiveness_score": visual_pres.get('facial_expressiveness_score', 0),
        "confidence_level": visual_pres.get('confidence_level', 'unknown'),
        "nervous_habits_detected": visual_pres.get('nervous_habits_detected', []),
        "overall_visual_score": visual_pres.get('overall_visual_score', 0),
        "posture_score_ai": visual_pres.get('posture_score_ai', 0),
        "strengths": visual_pres.get('strengths', []),
        "improvements": visual_pres.get('improvements', [])
    }

def _extract_audio_data(audio_analysis: Dict) -> Dict:
    """Extract relevant audio metrics for aggregation"""
    transcript_data = audio_analysis.get('transcript_data', {})
    emotion_data = audio_analysis.get('emotion_data', {})
    vocal_data = audio_analysis.get('vocal_data', {})
    
    return {
        "full_transcript": transcript_data.get('full_transcript', ''),
        "word_count": transcript_data.get('word_count', 0),
        "speech_duration_seconds": transcript_data.get('duration', 0),
        "dominant_emotion": emotion_data.get('dominant_emotion', 'neutral'),
        "emotion_distribution": emotion_data.get('emotion_scores', {}),
        "vocal_confidence": emotion_data.get('vocal_confidence', 'moderate'),
        "words_per_minute": vocal_data.get('wpm', 0),
        "filler_words_count": vocal_data.get('filler_count', 0),
        "filler_words": vocal_data.get('filler_words', []),
        "speaking_pace": vocal_data.get('speaking_pace', 'unknown'),
        "articulation_score": vocal_data.get('articulation_score', 0)
    }


def _extract_content_data(content_analysis: Dict) -> Dict:
    """Extract relevant content metrics for aggregation"""
    content_result = content_analysis.get('content_analysis', {})
    
    return {
        "structure_score": content_result.get('structure_score', 0),
        "clarity_score": content_result.get('clarity_score', 0),
        "content_depth_score": content_result.get('content_depth_score', 0),
        "engagement_score": content_result.get('engagement_score', 0),
        "language_quality_score": content_result.get('language_quality_score', 0),
        "audience_awareness_score": content_result.get('audience_awareness_score', 0),
        "overall_content_score": content_result.get('overall_content_score', 0),
        "content_rating": content_result.get('content_rating', 'needs_improvement'),
        "opening_strength": content_result.get('opening_strength', 'weak'),
        "closing_strength": content_result.get('closing_strength', 'weak'),
        "message_focus": content_result.get('message_focus', 'unfocused'),
        "key_themes": content_result.get('key_themes', []),
        "strongest_elements": content_result.get('strongest_content_elements', []),
        "weaknesses": content_result.get('content_weaknesses', [])
    }


def save_results(aggregated_results: Dict, video_id: str, **context) -> str:
    """
    Save final results to database and JSON backup
    
    Args:
        aggregated_results: Complete dashboard report
        video_id: Unique video identifier
        
    Returns:
        Path to saved results file
    """
    try:
        import os
        from utils.db_helper import save_results_to_db
        
        # Ensure output directory exists
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build final output structure
        final_output = {
            "video_id": video_id,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "status": "success",
            "results": aggregated_results
        }
        
        # Save to database (PRIMARY)
        logger.info(f"Saving results to database for video: {video_id}")
        save_results_to_db(video_id, final_output)
        
        # Save to JSON (BACKUP)
        output_path = f"{output_dir}/results_{video_id}.json"
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        logger.info(f"✅ Results saved to database AND {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise PipelineError(f"Result saving failed: {e}")

# def save_results(aggregated_results: Dict, video_id: str, **context) -> str:
#     """
#     Save final results to JSON file (placeholder for future DB integration)
    
#     Args:
#         aggregated_results: Complete dashboard report
#         video_id: Unique video identifier
        
#     Returns:
#         Path to saved results file
#     """
#     try:
#         import os
        
#         # Ensure output directory exists
#         output_dir = "data/output"
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Build final output structure
#         final_output = {
#             "video_id": video_id,
#             "timestamp": datetime.utcnow().isoformat() + 'Z',
#             "status": "success",
#             "results": aggregated_results
#         }
        
#         # Save to JSON
#         output_path = f"{output_dir}/results_{video_id}.json"
#         with open(output_path, 'w') as f:
#             json.dump(final_output, f, indent=2)
        
#         logger.info(f"✅ Results saved to: {output_path}")
        
#         # TODO: Future integration
#         # - Upload to GCS: upload_to_gcs(output_path, bucket_name, destination)
#         # - Save to database: save_results_to_db(video_id, final_output)
        
#         return output_path
        
#     except Exception as e:
#         logger.error(f"Failed to save results: {e}")
#         raise PipelineError(f"Result saving failed: {e}")


def generate_summary_stats(aggregated_results: Dict, **context) -> Dict:
    """
    Generate quick summary statistics for logging/monitoring
    
    Returns:
        Dict with key metrics for monitoring
    """
    try:
        category_scores = aggregated_results.get('category_scores', {})
        key_metrics = aggregated_results.get('key_metrics', {})
        
        return {
            'overall_score': aggregated_results.get('overall_score', 0),
            'performance_level': aggregated_results.get('performance_level', 'unknown'),
            'content_score': category_scores.get('content', {}).get('score', 0),
            'delivery_score': category_scores.get('vocal_delivery', {}).get('score', 0),
            'visual_score': category_scores.get('visual_presentation', {}).get('score', 0),
            'tone_score': category_scores.get('tone_emotion', {}).get('score', 0),
            'wpm': key_metrics.get('words_per_minute', 0),
            'filler_count': key_metrics.get('filler_words_count', 0),
            'improvement_count': len(aggregated_results.get('improvements', [])),
            'strength_count': len(aggregated_results.get('strengths', []))
        }
        
    except Exception as e:
        logger.error(f"Failed to generate summary stats: {e}")
        return {}