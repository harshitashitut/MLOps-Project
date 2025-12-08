"""
Audio analysis: Transcription (Whisper) + Emotion (Wav2Vec2) + Vocal metrics
"""
import logging
import re
from typing import Dict
from openai import OpenAI
from transformers import pipeline
import torch
from utils.config import config
from utils.error_handlers import PipelineError

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Initialize Wav2Vec2 emotion pipeline (lazy load)
emotion_pipeline = None

def get_emotion_pipeline():
    """Lazy load emotion detection model"""
    global emotion_pipeline
    if emotion_pipeline is None:
        logger.info("Loading Wav2Vec2 emotion model...")
        device = 0 if torch.cuda.is_available() else -1
        emotion_pipeline = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=device
        )
        logger.info(f"Emotion model loaded on: {'GPU' if device == 0 else 'CPU'}")
    return emotion_pipeline


def transcribe_with_whisper(audio_path: str, **context) -> Dict:
    """
    Transcribe audio using OpenAI Whisper API
    
    Returns:
        Dict with keys: transcript, word_count, duration, language
    """
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        
        with open(audio_path, 'rb') as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        
        transcript = response.text
        duration = response.duration if hasattr(response, 'duration') else 0
        language = response.language if hasattr(response, 'language') else 'en'
        
        word_count = len(transcript.split())
        
        logger.info(f"✅ Transcription complete: {word_count} words in {duration:.1f}s")
        
        return {
            'transcript': transcript,
            'word_count': word_count,
            'duration': duration,
            'language': language,
            'model': 'whisper-1'
        }
        
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise PipelineError(f"Whisper API failed: {e}")


def analyze_emotion_wav2vec(audio_path: str, **context) -> Dict:
    """
    Analyze emotion using Wav2Vec2 model
    
    Returns:
        Dict with keys: dominant_emotion, emotion_scores, confidence
    """
    try:
        logger.info(f"Analyzing emotion: {audio_path}")
        
        # Get emotion pipeline
        classifier = get_emotion_pipeline()
        
        # Run classification
        results = classifier(audio_path, top_k=5)
        
        # Extract dominant emotion
        dominant = results[0]
        dominant_emotion = dominant['label']
        confidence = dominant['score']
        
        # Build emotion distribution
        emotion_scores = {r['label']: round(r['score'], 3) for r in results}
        
        logger.info(f"✅ Emotion detected: {dominant_emotion} ({confidence:.2f})")
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': round(confidence, 3),
            'emotion_scores': emotion_scores,
            'model': 'wav2vec2-lg-xlsr-en-speech-emotion-recognition'
        }
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        # Non-blocking fallback
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'emotion_scores': {'neutral': 1.0},
            'error': str(e)
        }


def compute_vocal_metrics(transcript: str, duration: float, **context) -> Dict:
    """
    Calculate vocal delivery metrics from transcript
    
    Returns:
        Dict with keys: wpm, filler_count, filler_words, articulation_score, speaking_pace
    """
    try:
        logger.info("Computing vocal metrics...")
        
        if duration == 0:
            logger.warning("Duration is 0, using word count for estimation")
            duration = len(transcript.split()) / 150  # Assume 150 WPM
        
        # Words per minute
        word_count = len(transcript.split())
        wpm = round((word_count / duration) * 60, 1)
        
        # Filler words detection
        filler_patterns = [
            r'\buh+\b', r'\bum+\b', r'\blike\b', r'\byou know\b',
            r'\bbasically\b', r'\bactually\b', r'\bso\b', r'\bright\b',
            r'\bokay\b', r'\byeah\b', r'\bwell\b'
        ]
        
        filler_words = []
        filler_count = 0
        for pattern in filler_patterns:
            matches = re.findall(pattern, transcript.lower())
            filler_count += len(matches)
            filler_words.extend(matches)
        
        # Articulation score (0-10)
        # Based on WPM (optimal 120-150) and filler frequency
        wpm_score = 10 - abs(135 - wpm) / 10  # Penalty for deviation from 135 WPM
        wpm_score = max(0, min(10, wpm_score))
        
        filler_penalty = min(5, filler_count / word_count * 100)  # Max 5 point penalty
        articulation_score = max(0, round(wpm_score - filler_penalty, 1))
        
        # Speaking pace classification
        if wpm < 110:
            speaking_pace = "slow"
        elif wpm > 160:
            speaking_pace = "fast"
        else:
            speaking_pace = "optimal"
        
        logger.info(f"✅ Vocal metrics: {wpm} WPM, {filler_count} fillers, score {articulation_score}")
        
        return {
            'wpm': wpm,
            'filler_count': filler_count,
            'filler_words': list(set(filler_words)),  # Unique fillers
            'articulation_score': articulation_score,
            'speaking_pace': speaking_pace,
            'word_count': word_count,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"Vocal metrics computation failed: {e}")
        return {
            'wpm': 0,
            'filler_count': 0,
            'filler_words': [],
            'articulation_score': 0,
            'speaking_pace': 'unknown',
            'error': str(e)
        }


def combine_audio_analyses(
    transcription: Dict,
    emotion: Dict,
    vocal_metrics: Dict,
    **context
) -> Dict:
    """
    Combine all audio analysis results into unified structure
    
    Returns:
        Dict with keys: transcript_data, emotion_data, vocal_data, summary
    """
    try:
        logger.info("Combining audio analyses...")
        
        # Map emotion to confidence level
        emotion_name = emotion.get('dominant_emotion', 'neutral')
        emotion_confidence = emotion.get('confidence', 0)
        
        confidence_mapping = {
            'angry': 'aggressive',
            'happy': 'confident',
            'sad': 'low',
            'neutral': 'moderate',
            'fear': 'nervous'
        }
        vocal_confidence = confidence_mapping.get(emotion_name.lower(), 'moderate')
        
        # Build combined result
        result = {
            'transcript_data': {
                'full_transcript': transcription.get('transcript', ''),
                'word_count': transcription.get('word_count', 0),
                'duration': transcription.get('duration', 0),
                'language': transcription.get('language', 'en')
            },
            'emotion_data': {
                'dominant_emotion': emotion_name,
                'confidence': emotion_confidence,
                'emotion_scores': emotion.get('emotion_scores', {}),
                'vocal_confidence': vocal_confidence
            },
            'vocal_data': {
                'wpm': vocal_metrics.get('wpm', 0),
                'speaking_pace': vocal_metrics.get('speaking_pace', 'unknown'),
                'filler_count': vocal_metrics.get('filler_count', 0),
                'filler_words': vocal_metrics.get('filler_words', []),
                'articulation_score': vocal_metrics.get('articulation_score', 0)
            },
            'summary': {
                'speaking_pace_rating': vocal_metrics.get('speaking_pace', 'unknown'),
                'vocal_confidence': vocal_confidence,
                'filler_words_count': vocal_metrics.get('filler_count', 0),
                'emotional_tone': emotion_name,
                'articulation_score': vocal_metrics.get('articulation_score', 0)
            }
        }
        
        logger.info("✅ Audio analyses combined successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to combine audio analyses: {e}")
        raise PipelineError(f"Audio combination failed: {e}")