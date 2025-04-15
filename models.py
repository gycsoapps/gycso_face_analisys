from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

# Request models
class Base64ComparisonRequest(BaseModel):
    image1: str = Field(..., description="Base64 encoded image string")
    image2: str = Field(..., description="Base64 encoded image string")
    model_name: str = Field(default="ArcFace", description="Face recognition model to use")
    enforce_detection: bool = Field(default=False, description="Whether to enforce face detection")
    detector_backend: str = Field(default="retinaface", description="Face detector backend")
    
class S3ComparisonRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string")
    s3_key: str = Field(..., description="Path to image in S3 bucket")
    model_name: str = Field(default="ArcFace", description="Face recognition model to use")
    enforce_detection: bool = Field(default=False, description="Whether to enforce face detection")
    detector_backend: str = Field(default="retinaface", description="Face detector backend")

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