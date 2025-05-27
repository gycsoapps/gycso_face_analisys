from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from config import get_settings

settings = get_settings()

class EventBridgeRequest(BaseModel):
    body: str = Field(..., description="Base64 encoded image string")
 

# Request models
class Base64ComparisonRequest(BaseModel):
    image1: str = Field(..., description="Base64 encoded image string")
    image2: str = Field(..., description="Base64 encoded image string")
    model_name: str = Field(default=settings.DEFAULT_MODEL, description="Face recognition model to use")
    enforce_detection: bool = Field(default=False, description="Whether to enforce face detection")
    detector_backend: str = Field(default=settings.DEFAULT_DETECTOR, description="Face detector backend")
    
class S3ComparisonRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string")
    s3_key: str = Field(..., description="Path to image in S3 bucket")
    model_name: str = Field(default=settings.DEFAULT_MODEL, description="Face recognition model to use")
    enforce_detection: bool = Field(default=False, description="Whether to enforce face detection")
    detector_backend: str = Field(default=settings.DEFAULT_DETECTOR, description="Face detector backend")

class DatabaseComparisonRequest(BaseModel):
    """Request model for comparing an image against a database of face embeddings"""
    image: str = Field(..., description="Base64 encoded image to verify")
    user_id: Optional[str] = Field(None, description="Optional user ID for logging purposes")
    threshold: Optional[float] = Field(0.6, description="Similarity threshold (0.0-1.0)")
    model_name: Optional[str] = Field("Facenet", description="Face recognition model to use")
    enforce_detection: Optional[bool] = Field(False, description="Whether to enforce face detection")
    detector_backend: Optional[str] = Field("ssd", description="Face detection backend")

# Response models
class FaceComparisonResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    facial_areas: Optional[Dict[str, Any]] = None
    time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# HTTP Error Response model
class HTTPError(BaseModel):
    detail: str 

class DatabaseMatch(BaseModel):
    """Model for a face database match result"""
    user_id: str = Field(..., description="ID of the matched user")
    similarity_score: float = Field(..., description="Similarity score (0.0-1.0)")
    distance: float = Field(..., description="Distance value from comparison")

class DatabaseComparisonResponse(BaseModel):
    """Response model for database face comparison"""
    access_granted: bool = Field(..., description="Whether access should be granted")
    match_found: bool = Field(..., description="Whether a face match was found")
    best_match: Optional[DatabaseMatch] = Field(None, description="Best matching user if found")
    all_matches: List[DatabaseMatch] = Field([], description="All matches above threshold")
    processing_time: float = Field(..., description="Processing time in seconds")
    total_faces_compared: int = Field(..., description="Total number of faces in database")

class EmbeddingData(BaseModel):
    """Model for storing face embedding data"""
    user_id: str = Field(..., description="User identifier")
    embedding: List[float] = Field(..., description="Face embedding vector")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata") 