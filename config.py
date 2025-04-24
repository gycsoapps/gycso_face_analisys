from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings that can be loaded from environment variables or .env file"""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = ""
    S3_BUCKET: str = ""
    DEFAULT_MODEL: str = ""
    DEFAULT_DETECTOR: str = ""
    LOG_LEVEL: str = ""
    
    # API settings
    API_TITLE: str = "Face Recognition API"
    API_DESCRIPTION: str = "API for comparing faces using DeepFace"
    API_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    """Get cached settings to avoid loading from environment variables multiple times"""
    return Settings() 