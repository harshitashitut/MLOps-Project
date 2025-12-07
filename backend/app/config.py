from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Required - will fail if not in .env
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str
    AIRFLOW_URL: str = "http://localhost:8080"
    AIRFLOW_USERNAME: str = "admin"
    AIRFLOW_PASSWORD: str = "admin"
    
    # Optional with defaults
    UPLOAD_DIR: str = "/tmp/pitchquest_uploads"
    MAX_UPLOAD_SIZE_MB: int = 100
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()