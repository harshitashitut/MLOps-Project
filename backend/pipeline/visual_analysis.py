"""
Visual analysis: Body language (MediaPipe) + Visual presentation (Gemini REST)
"""
import mediapipe as mp
import cv2
import logging
import json
from typing import Dict, List
from utils.config import config, load_prompt
from utils.error_handlers import PipelineError
from utils.gemini_client import gemini_client

logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
def analyze_pose_mediapipe(frame_paths: List[str], **context) -> Dict:
    """Stub: Skip MediaPipe (causing crashes), rely on Gemini Vision"""
    logger.info(f"Skipping MediaPipe (using Gemini Vision for all analysis)")
    return {'posture_score': 7.5}  # Placeholder

# def analyze_pose_mediapipe(frame_paths: List[str], **context) -> Dict:
#     """Analyze body language using MediaPipe Pose detection"""
#     try:
#         logger.info(f"Analyzing pose for {len(frame_paths)} frames")
        
#         # Initialize HERE, not at module level
#         mp_pose_local = mp.solutions.pose
#         pose_detector = mp_pose_local.Pose(
#             static_image_mode=True,
#             model_complexity=1,
#             enable_segmentation=False,
#             min_detection_confidence=0.5
#         )
        
#         pose_data = []
        
#         for frame_path in frame_paths:
#             image = cv2.imread(frame_path)
#             if image is None: 
#                 continue
                
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose_detector.process(image_rgb)
            
#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark
#                 shoulder_diff = abs(
#                     lm[mp_pose_local.PoseLandmark.LEFT_SHOULDER].y - 
#                     lm[mp_pose_local.PoseLandmark.RIGHT_SHOULDER].y
#                 )
#                 pose_data.append({'shoulder_alignment': shoulder_diff})
        
#         # Close detector
#         pose_detector.close()
        
#         if not pose_data: 
#             return {'posture_score': 5}
            
#         avg_diff = sum(d['shoulder_alignment'] for d in pose_data) / len(pose_data)
#         posture_score = max(0, 10 - (avg_diff * 50))
        
#         logger.info(f"✅ MediaPipe pose analysis complete: {posture_score}")
#         return {'posture_score': round(posture_score, 2)}
        
#     except Exception as e:
#         logger.error(f"MediaPipe failed: {e}", exc_info=True)
#         return {'posture_score': 5, 'error': str(e)}

def analyze_visual_gemini(frame_paths: List[str], **context) -> Dict:
    """Analyze visual presentation using Gemini Vision (REST)"""
    try:
        logger.info(f"Analyzing {len(frame_paths)} frames with Gemini (REST)")
        
        prompt = load_prompt('visual_analysis')
        selected_frames = frame_paths[:config.MAX_FRAMES_FOR_VISION]
        
        # Call REST Client
        visual_data = gemini_client.generate_content(prompt, selected_frames)
        
       
        
        logger.info("✅ Gemini analysis successful")
        return {
            'gemini_visual_analysis': visual_data,
            'model': config.GEMINI_MODEL
        }
        
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        raise PipelineError(f"Gemini analysis failed: {e}")

def combine_visual_analyses(mediapipe_data: Dict, gemini_data: Dict, **context) -> Dict:
    """Combine results"""
    gemini_result = gemini_data.get('gemini_visual_analysis', {})
    
    return {
        'visual_presentation': {
            'posture_score_ai': mediapipe_data.get('posture_score'),
            'overall_visual_score': gemini_result.get('overall_visual_score', 0),
            'posture_score': gemini_result.get('posture_score', 0),
            'gesture_score': gemini_result.get('gesture_score', 0),
            'eye_contact_score': gemini_result.get('eye_contact_score', 0),
            'confidence_level': gemini_result.get('confidence_level', 'unknown'),
            'visual_feedback': gemini_result.get('posture_notes', 'No feedback'),
            'strengths': gemini_result.get('strongest_visual_elements', []),
            'improvements': gemini_result.get('areas_for_improvement', [])
        }
    }