"""
Content analysis: Analyze pitch content using Gemini LLM
"""
import logging
from typing import Dict
from utils.config import load_prompt
from utils.error_handlers import PipelineError
from utils.gemini_client import gemini_client

logger = logging.getLogger(__name__)


def analyze_content_gemini(transcript: str, **context) -> Dict:
    """
    Analyze pitch content (problem, solution, market, etc.) using Gemini LLM
    
    Args:
        transcript: Full text transcript from Whisper
        
    Returns:
        Dict with business analysis scores and feedback
    """
    try:
        logger.info(f"Analyzing content ({len(transcript)} chars)...")
        
        # Load content analysis prompt
        prompt_template = load_prompt('content_analysis')
        
        # Insert transcript into prompt
        full_prompt = f"{prompt_template}\n\n=== TRANSCRIPT ===\n{transcript}\n\n=== END TRANSCRIPT ==="
        
        # Call Gemini (text-only, no images)
        content_data = gemini_client.generate_content(full_prompt)
        
        logger.info("✅ Content analysis complete")
        
        return {
            'content_analysis': content_data,
            'transcript_length': len(transcript),
            'model': 'gemini-2.0-flash'
        }
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise PipelineError(f"Gemini content analysis failed: {e}")


def extract_content_summary(analysis_result: Dict, **context) -> Dict:
    """
    Extract key metrics from content analysis for easy access
    
    Returns:
        Dict with flattened content scores
    """
    try:
        content_data = analysis_result.get('content_analysis', {})
        
        return {
            'structure_score': content_data.get('structure_score', 0),
            'clarity_score': content_data.get('clarity_score', 0),
            'content_depth_score': content_data.get('content_depth_score', 0),
            'engagement_score': content_data.get('engagement_score', 0),
            'language_quality_score': content_data.get('language_quality_score', 0),
            'audience_awareness_score': content_data.get('audience_awareness_score', 0),
            'overall_content_score': content_data.get('overall_content_score', 0),
            'content_rating': content_data.get('content_rating', 'needs_improvement'),
            'opening_strength': content_data.get('opening_strength', 'weak'),
            'closing_strength': content_data.get('closing_strength', 'weak'),
            'message_focus': content_data.get('message_focus', 'unfocused'),
            'evidence_quality': content_data.get('evidence_quality', 'absent'),
            'emotional_connection': content_data.get('emotional_connection', 'weak'),
            'vocabulary_level': content_data.get('vocabulary_level', 'appropriate'),
            'key_themes': content_data.get('key_themes', []),
            'strengths': content_data.get('strongest_content_elements', []),
            'weaknesses': content_data.get('content_weaknesses', [])
        }
        
    except Exception as e:
        logger.error(f"Failed to extract content summary: {e}")
        return {
            'structure_score': 0,
            'clarity_score': 0,
            'content_depth_score': 0,
            'engagement_score': 0,
            'language_quality_score': 0,
            'audience_awareness_score': 0,
            'overall_content_score': 0,
            'content_rating': 'needs_improvement',
            'opening_strength': 'weak',
            'closing_strength': 'weak',
            'message_focus': 'unfocused',
            'evidence_quality': 'absent',
            'emotional_connection': 'weak',
            'vocabulary_level': 'appropriate',
            'key_themes': [],
            'strengths': [],
            'weaknesses': []
        }