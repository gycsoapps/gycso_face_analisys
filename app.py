from fastapi import FastAPI, HTTPException
import logging
from typing import List

# Import from our modules
from config import get_settings
from models import (
    Base64ComparisonRequest, 
    S3ComparisonRequest, 
    FaceComparisonResponse,
    HealthResponse
)
from utils import decode_base64_to_image, get_image_from_s3, get_image_from_s3_cached
from face_service import FaceService

# Set up logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)


# API Endpoints
@app.post("/compare-base64", 
          response_model=FaceComparisonResponse, 
          summary="Compare two base64 encoded images",
          description="Takes two base64 encoded images and returns whether they contain the same person")
async def compare_base64_images(request: Base64ComparisonRequest):
    """Compare two base64 encoded images for face recognition"""
    try:
        # Decode base64 images to numpy arrays
        img1_array = decode_base64_to_image(request.image1)
        img2_array = decode_base64_to_image(request.image2)
        
        # Compare the images using FaceService
        result = await FaceService.compare_faces(
            img1_array=img1_array,
            img2_array=img2_array,
            model_name=request.model_name,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-with-s3", 
          response_model=FaceComparisonResponse,
          summary="Compare base64 image with S3 image",
          description="Compares a base64 encoded image with an image stored in AWS S3")
async def compare_with_s3_image(request: S3ComparisonRequest):
    """Compare a base64 encoded image with an image stored in S3"""
    try:
        # Get the images as numpy arrays
        img_array = decode_base64_to_image(request.image)
        s3_img_array = get_image_from_s3_cached(request.s3_key)
        
        # Compare the images using FaceService
        result = await FaceService.compare_faces(
            img1_array=img_array,
            img2_array=s3_img_array,
            model_name=request.model_name,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in S3 face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health", 
         response_model=HealthResponse, 
         summary="Health check endpoint")
async def health_check():
    """Simple health check endpoint to verify the API is running"""
    return {
        "status": "healthy", 
        "service": "face-recognition-api",
        "version": settings.API_VERSION
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 