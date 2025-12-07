import os
import shutil
from pathlib import Path
from fastapi import UploadFile
from app.config import settings

class StorageService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_video(self, file: UploadFile, video_id: str) -> str:
        """Save uploaded video, return file path"""
        # Get extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.mp4', '.mov', '.avi']:
            raise ValueError(f"Invalid file type: {ext}")
        
        # Save file
        file_path = self.upload_dir / f"{video_id}{ext}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            file_path.unlink()
            raise ValueError(f"File too large: {size_mb:.1f}MB")
        
        return str(file_path)

storage = StorageService()