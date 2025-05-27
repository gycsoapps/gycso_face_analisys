from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings that can be loaded from environment variables or .env file"""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = ""
    S3_BUCKET: str = ""
    DEFAULT_MODEL: str = "Facenet512"
    DEFAULT_DETECTOR: str = "ssd"
    LOG_LEVEL: str = "INFO"
    
    # API settings
    API_TITLE: str = "Face Recognition API"
    API_DESCRIPTION: str = "API for comparing faces using DeepFace"
    API_VERSION: str = "1.0.0"
    
    # Database settings for face embeddings
    EMBEDDING_DB_PATH: str = "face_embeddings.json"
    DEFAULT_RECOGNITION_THRESHOLD: float = 0.6
    PRELOAD_DATABASE_ON_STARTUP: bool = True
    
    # Performance settings
    MAX_PARALLEL_COMPARISONS: int = 10
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_SIZE_LIMIT: int = 1000
    
    # High-scale optimization settings
    USE_FAISS_INDEX: bool = True
    FAISS_INDEX_TYPE: str = "IVF"  # Options: "Flat", "IVF", "HNSW"
    FAISS_NLIST: int = 100  # Number of clusters for IVF
    FAISS_NPROBE: int = 10  # Number of clusters to search
    FAISS_INDEX_PATH: str = "faiss_index.bin"
    
    # Redis cache settings (optional for distributed systems)
    USE_REDIS_CACHE: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_EXPIRE_TIME: int = 3600  # 1 hour
    
    # Batch processing settings
    BATCH_SIZE: int = 100
    MAX_WORKERS: int = 4
    
    # Memory optimization
    ENABLE_MEMORY_OPTIMIZATION: bool = True
    MAX_MEMORY_USAGE_MB: int = 2048
    AUTO_CLEANUP_INTERVAL: int = 300  # 5 minutes
    
    # HDF5 storage for large datasets
    USE_HDF5_STORAGE: bool = False
    HDF5_FILE_PATH: str = "embeddings.h5"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    """Get cached settings to avoid loading from environment variables multiple times"""
    return Settings() 