import os
from pathlib import Path
from utils.error_handlers import PipelineError

def validate_video_file(video_path: str) -> None:
    """Validate video file exists and has correct extension"""
    if not os.path.exists(video_path):
        raise PipelineError(f"Video file not found: {video_path}")
    
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if not any(video_path.endswith(ext) for ext in valid_extensions):
        raise PipelineError(f"Invalid video format. Supported: {valid_extensions}")
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise PipelineError(f"Video file is empty: {video_path}")

def validate_api_keys() -> None:
    """Validate required API keys are set"""
    from utils.config import config
    
    if not config.OPENAI_API_KEY:
        raise PipelineError("OPENAI_API_KEY not set in environment")
    
    if not config.GEMINI_API_KEY:
        raise PipelineError("GEMINI_API_KEY not set in environment")