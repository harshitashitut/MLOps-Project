"""
Video preprocessing: Extract frames and audio from video file
"""
import cv2
import os
import tempfile
import ffmpeg
from pathlib import Path
from typing import Dict, List
import logging
from utils.config import config
from utils.validators import validate_video_file
from utils.error_handlers import PipelineError

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, sample_rate: int = None, video_id: str = None) -> Dict:
    """
    Extract frames from video at specified sample rate
    
    Args:
        video_path: Path to video file
        sample_rate: Extract 1 frame per N seconds (default from config)
        video_id: Optional video identifier for temp directory naming
        
    Returns:
        Dict with frames_dir, frame_paths, frame_count
        
    Raises:
        PipelineError: If extraction fails
    """
    if sample_rate is None:
        sample_rate = config.FRAME_SAMPLE_RATE
    
    if video_id is None:
        video_id = 'video'
    
    # Validate input
    validate_video_file(video_path)
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix=f"frames_{video_id}_")
    
    frame_paths = []
    
    try:
        logger.info(f"Extracting frames from {video_path} (1 frame per {sample_rate}s)")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise PipelineError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} total frames")
        
        # Calculate frame interval
        frame_interval = int(fps * sample_rate)
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(temp_dir, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                
                # Limit frames for vision API (cost control)
                if extracted_count >= config.MAX_FRAMES_FOR_VISION:
                    logger.info(f"Reached max frames limit ({config.MAX_FRAMES_FOR_VISION})")
                    break
            
            frame_count += 1
        
        cap.release()
        
        if not frame_paths:
            raise PipelineError("No frames extracted from video")
        
        logger.info(f"✅ Extracted {len(frame_paths)} frames to {temp_dir}")
        
        return {
            'frames_dir': temp_dir,
            'frame_paths': frame_paths,
            'frame_count': len(frame_paths),
            'video_duration': duration,
            'video_fps': fps
        }
        
    except Exception as e:
        # Cleanup on failure
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error(f"Frame extraction failed: {e}")
        raise PipelineError(f"Frame extraction failed: {e}")


def extract_audio(video_path: str, video_id: str = None) -> Dict:
    """
    Extract audio from video using FFmpeg
    
    Args:
        video_path: Path to video file
        video_id: Optional video identifier
        
    Returns:
        Dict with audio_path and audio_duration
    """
    import subprocess
    
    validate_video_file(video_path)
    
    if video_id is None:
        video_id = 'video'
    
    temp_audio = tempfile.NamedTemporaryFile(prefix=f"audio_{video_id}_", suffix=".mp3", delete=False)
    audio_path = temp_audio.name
    temp_audio.close()
    
    try:
        logger.info(f"Extracting audio from {video_path}")
        
        # Direct FFmpeg command
        command = [
            'ffmpeg', '-y',
            '-nostdin', 
            '-i', video_path,
            '-ac', '1',
            '-ar', '16000',
            '-acodec', 'libmp3lame',
            '-vn',
            audio_path
        ]
        
        # Run with timeout
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        
        # Verify + get duration
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise PipelineError("Audio file not created or empty")
        
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['streams'][0]['duration'])
        
        logger.info(f"✅ Extracted audio: {duration:.1f}s, saved to {audio_path}")
        return {'audio_path': audio_path, 'audio_duration': duration}
        
    except subprocess.TimeoutExpired:
        if os.path.exists(audio_path): 
            os.remove(audio_path)
        raise PipelineError("FFmpeg timed out")
    except subprocess.CalledProcessError as e:
        if os.path.exists(audio_path): 
            os.remove(audio_path)
        raise PipelineError(f"FFmpeg failed: {e.stderr.decode()}")
    except Exception as e:
        if os.path.exists(audio_path): 
            os.remove(audio_path)
        raise PipelineError(f"Audio extraction failed: {e}")


def cleanup_temp_files(frames_dir: str = None, audio_path: str = None) -> None:
    """
    Cleanup temporary files created during preprocessing
    
    Args:
        frames_dir: Directory containing frames
        audio_path: Path to audio file
    """
    import shutil
    
    cleaned = []
    
    if frames_dir and os.path.exists(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            cleaned.append(f"frames_dir: {frames_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup frames_dir {frames_dir}: {e}")
    
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            cleaned.append(f"audio: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio {audio_path}: {e}")
    
    if cleaned:
        logger.info(f"✅ Cleaned up: {', '.join(cleaned)}")