from fastapi import FastAPI, HTTPException
import logging
import time
import os
from mangum import Mangum

# Import from our modules
from config import get_settings
from models import (
    Base64ComparisonRequest,
    EventBridgeRequest, 
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
    FaceService.preload_models(
        model_name=settings.DEFAULT_MODEL,
        detector_backend=settings.DEFAULT_DETECTOR
    )
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
@app.post("/api/photo/compare-base64", 
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
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/photo/compare-with-s3", 
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
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in S3 face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.post("/api/photo/health", 
         response_model=HealthResponse, 
         summary="Health check endpoint")
def health_check(request: EventBridgeRequest):
    """Simple health check endpoint to verify the API is running"""
    logger.info(f"Received EventBridge request: {request.body}")
    return {
        "status": "healthy", 
        "service": "face-recognition-api",
        "version": settings.API_VERSION
    }

# Create Lambda handler
handler = Mangum(app, lifespan="off")

# Wrapper para manejar tanto eventos de API Gateway como de EventBridge
def lambda_handler(event, context):
    """
    Handler personalizado para manejar eventos de API Gateway y EventBridge
    """
    logger.warning(f"Received event: {event}")
    
    # Verificar si es un evento de EventBridge 
    # Caso 1: Evento estándar de EventBridge (tiene source o detail-type)
    # Caso 2: Evento personalizado con payload específico {"body":"EventBridge call"}
    if 'source' in event or 'detail-type' in event or (isinstance(event, dict) and event.get('body') == 'EventBridge call'):
        logger.info("Processing EventBridge event")
        
        # Si el evento es el payload personalizado, simulamos una llamada al endpoint health
        if isinstance(event, dict) and 'body' in event:
            try:
                # Simulamos una llamada al endpoint health con el body recibido
                logger.info(f"Calling health endpoint with body: {event['body']}")
                
                # Crear objeto EventBridgeRequest
                from models import EventBridgeRequest
                request = EventBridgeRequest(body=event['body'])
                
                # Llamar al endpoint directamente
                result = health_check(request)
                
                # Formatear respuesta para Lambda
                return {
                    "statusCode": 200,
                    "body": {
                        "status": result["status"],
                        "service": result["service"],
                        "version": result["version"],
                        "message": "Successfully processed scheduled event with payload"
                    }
                }
            except Exception as e:
                logger.error(f"Error processing EventBridge payload: {str(e)}")
                return {
                    "statusCode": 500,
                    "body": {
                        "error": str(e)
                    }
                }
        
        # Respuesta estándar para otros eventos de EventBridge
        return {
            "statusCode": 200,
            "body": {
                "status": "healthy",
                "service": "face-recognition-api",
                "version": settings.API_VERSION,
                "message": "Successfully processed scheduled event"
            }
        }
    
    # Si no es un evento de EventBridge, pasar al handler de Mangum
    logger.info("Processing API Gateway event")
    return handler(event, context)

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) 