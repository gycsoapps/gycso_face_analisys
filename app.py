from fastapi import FastAPI, HTTPException
import logging
import time
import os
from mangum import Mangum

# Import from our modules
from config import get_settings
from models import (
    Base64ComparisonRequest, 
    S3ComparisonRequest, 
    FaceComparisonResponse,
    HealthResponse,
    HTTPError
)
from utils import decode_base64_to_image, get_image_from_s3_cached
from face_service import FaceService

# Set up logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Preload models at startup to improve cold start performance
# Even though we preload in Docker, doing it here ensures models are loaded even if container is restarted
try:
    start_time = time.time()
    logger.info("Preloading face recognition models...")
    # Reduce TensorFlow logging during model loading
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Preload the model that will be used
    FaceService.preload_models(model_name=settings.DEFAULT_MODEL)
    logger.info(f"Models preloaded successfully in {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error preloading models: {str(e)}")

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
          description="Takes two base64 encoded images and returns whether they contain the same person",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def compare_base64_images(request: Base64ComparisonRequest):
    """Compare two base64 encoded images for face recognition"""
    start_time = time.time()
    logger.info("Starting base64 image comparison")
    
    try:
        # Decode base64 images to numpy arrays
        img1_array = decode_base64_to_image(request.image1)
        img2_array = decode_base64_to_image(request.image2)
        
        # Compare the images using FaceService
        result = FaceService.compare_faces(
            img1_array=img1_array,
            img2_array=img2_array,
            model_name=request.model_name,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend
        )
        
        logger.info(f"Base64 comparison completed in {time.time() - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-with-s3", 
          response_model=FaceComparisonResponse,
          summary="Compare base64 image with S3 image",
          description="Compares a base64 encoded image with an image stored in AWS S3",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def compare_with_s3_image(request: S3ComparisonRequest):
    """Compare a base64 encoded image with an image stored in S3"""
    start_time = time.time()
    logger.info(f"Starting S3 comparison with key: {request.s3_key}")
    
    try:
        # Get the images as numpy arrays
        img_array = decode_base64_to_image(request.image)
        print("image from base 64 decoded correctly")
        s3_img_array = get_image_from_s3_cached(request.s3_key)
        print("image from s3 cached correctly")
        
        # Compare the images using FaceService
        result = FaceService.compare_faces(
            img1_array=img_array,
            img2_array=s3_img_array,
            model_name=request.model_name,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend
        )
        
        logger.info(f"S3 comparison completed in {time.time() - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        logger.error(f"Error in S3 face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health", 
         response_model=HealthResponse, 
         summary="Health check endpoint")
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return {
        "status": "healthy", 
        "service": "face-recognition-api",
        "version": settings.API_VERSION
    }

# Create Lambda handler
handler = Mangum(app, lifespan="off")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 