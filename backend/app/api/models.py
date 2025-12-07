from pydantic import BaseModel
from datetime import datetime

class UploadResponse(BaseModel):
    video_id: str
    filename: str
    status: str
    message: str