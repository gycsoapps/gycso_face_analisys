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
    HTTPError,
    DatabaseComparisonRequest,
    DatabaseComparisonResponse,
    EmbeddingData
)
from utils import decode_base64_to_image, get_image_from_s3_cached
from face_service import FaceService, embedding_db
from face_service_optimized import OptimizedFaceService, optimized_embedding_db

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

@app.post("/api/photo/compare-with-database", 
          response_model=DatabaseComparisonResponse,
          summary="Compare image with face database",
          description="Compares a base64 encoded image with all faces in the precomputed embedding database for access control",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def compare_with_database(request: DatabaseComparisonRequest):
    """Compare a face image against the entire face database for access control"""
    start_time = time.time()
    logger.info(f"Starting database comparison for user: {request.user_id or 'anonymous'}")
    
    try:
        # Get all embeddings from database
        database_embeddings = embedding_db.get_all_embeddings()
        
        if not database_embeddings:
            logger.warning("No embeddings found in database")
            return DatabaseComparisonResponse(
                access_granted=False,
                match_found=False,
                best_match=None,
                all_matches=[],
                processing_time=time.time() - start_time,
                total_faces_compared=0
            )
        
        # Decode input image
        img_array = decode_base64_to_image(request.image)
        
        # Compare with database
        result = FaceService.compare_with_database(
            img_array=img_array,
            database_embeddings=database_embeddings,
            model_name=request.model_name,
            threshold=request.threshold,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend
        )
        
        logger.info(f"Database comparison completed in {result['processing_time']:.2f} seconds. "
                   f"Access {'granted' if result['access_granted'] else 'denied'}")
        
        return DatabaseComparisonResponse(**result)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in database face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/photo/add-face-to-database",
          summary="Add face to database",
          description="Extract face embedding from image and add to database",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def add_face_to_database(request: dict):
    """Add a new face to the embedding database"""
    start_time = time.time()
    logger.info(f"Adding face to database for user: {request.get('user_id')}")
    
    try:
        user_id = request.get("user_id")
        image_b64 = request.get("image")
        metadata = request.get("metadata", {})
        model_name = request.get("model_name", "Facenet")
        enforce_detection = request.get("enforce_detection", True)
        detector_backend = request.get("detector_backend", "ssd")
        
        if not user_id or not image_b64:
            raise HTTPException(status_code=400, detail="user_id and image are required")
        
        # Decode image
        img_array = decode_base64_to_image(image_b64)
        
        # Extract embedding
        embedding_result = FaceService.extract_face_embedding(
            img_array=img_array,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend
        )
        
        # DeepFace.represent returns a list, get the first embedding
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            embedding = embedding_result[0]["embedding"]
        else:
            raise ValueError("No face embedding could be extracted from image")
        
        # Add to database
        embedding_db.add_embedding(user_id, embedding, metadata)
        
        processing_time = time.time() - start_time
        logger.info(f"Face added to database for user {user_id} in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "message": f"Face embedding added for user {user_id}",
            "processing_time": processing_time,
            "user_id": user_id,
            "total_faces_in_database": len(embedding_db.get_all_embeddings())
        }
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error adding face to database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photo/database-stats",
         summary="Get database statistics",
         description="Get current statistics about the face embedding database")
def get_database_stats():
    """Get statistics about the face embedding database"""
    try:
        stats = embedding_db.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/photo/remove-face/{user_id}",
           summary="Remove face from database",
           description="Remove a user's face embedding from the database")
def remove_face_from_database(user_id: str):
    """Remove a face embedding from the database"""
    try:
        success = embedding_db.remove_embedding(user_id)
        
        if success:
            return {
                "success": True,
                "message": f"Face embedding removed for user {user_id}",
                "total_faces_in_database": len(embedding_db.get_all_embeddings())
            }
        else:
            raise HTTPException(status_code=404, detail=f"No face embedding found for user {user_id}")
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error removing face from database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== OPTIMIZED ENDPOINTS FOR LARGE SCALE ====================

@app.post("/api/photo/compare-with-database-optimized", 
          response_model=DatabaseComparisonResponse,
          summary="Compare image with large face database (FAISS optimized)",
          description="Ultra-fast comparison for databases with 5K-10K+ faces using FAISS indexing",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def compare_with_database_optimized(request: DatabaseComparisonRequest):
    """Optimized face comparison for large databases using FAISS"""
    start_time = time.time()
    logger.info(f"Starting OPTIMIZED database comparison for user: {request.user_id or 'anonymous'}")
    
    try:
        # Check if database has embeddings
        stats = optimized_embedding_db.get_stats()
        if stats["total_embeddings"] == 0:
            logger.warning("No embeddings found in optimized database")
            return DatabaseComparisonResponse(
                access_granted=False,
                match_found=False,
                best_match=None,
                all_matches=[],
                processing_time=time.time() - start_time,
                total_faces_compared=0
            )
        
        # Decode input image
        img_array = decode_base64_to_image(request.image)
        
        # Use optimized comparison method
        result = optimized_embedding_db.compare_with_database(
            img_array=img_array,
            model_name=request.model_name,
            threshold=request.threshold,
            enforce_detection=request.enforce_detection,
            detector_backend=request.detector_backend,
            max_results=20  # Return more results for large databases
        )
        
        logger.info(f"OPTIMIZED comparison completed in {result['processing_time']:.2f} seconds using {result['search_method']}. "
                   f"Access {'granted' if result['access_granted'] else 'denied'}")
        
        return DatabaseComparisonResponse(**result)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in optimized database comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/photo/add-face-to-database-optimized",
          summary="Add face to optimized database",
          description="Add face embedding to large-scale optimized database with FAISS indexing",
          responses={
              500: {"model": HTTPError, "description": "Internal server error"}
          })
def add_face_to_database_optimized(request: dict):
    """Add a new face to the optimized embedding database"""
    start_time = time.time()
    logger.info(f"Adding face to OPTIMIZED database for user: {request.get('user_id')}")
    
    try:
        user_id = request.get("user_id")
        image_b64 = request.get("image")
        metadata = request.get("metadata", {})
        model_name = request.get("model_name", "Facenet")
        enforce_detection = request.get("enforce_detection", True)
        detector_backend = request.get("detector_backend", "ssd")
        
        if not user_id or not image_b64:
            raise HTTPException(status_code=400, detail="user_id and image are required")
        
        # Decode image
        img_array = decode_base64_to_image(image_b64)
        
        # Extract embedding using optimized service
        embedding_result = OptimizedFaceService.extract_face_embedding(
            img_array=img_array,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend
        )
        
        # Process embedding result
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            embedding = embedding_result[0]["embedding"]
        else:
            raise ValueError("No face embedding could be extracted from image")
        
        # Add to optimized database
        optimized_embedding_db.add_embedding(user_id, embedding, metadata)
        
        processing_time = time.time() - start_time
        logger.info(f"Face added to OPTIMIZED database for user {user_id} in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "message": f"Face embedding added to optimized database for user {user_id}",
            "processing_time": processing_time,
            "user_id": user_id,
            "total_faces_in_database": len(optimized_embedding_db.get_all_embeddings()),
            "faiss_index_rebuilt": True
        }
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error adding face to optimized database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photo/database-stats-optimized",
         summary="Get optimized database statistics",
         description="Get comprehensive statistics about the FAISS-optimized face database")
def get_database_stats_optimized():
    """Get statistics about the optimized face embedding database"""
    try:
        stats = optimized_embedding_db.get_stats()
        
        # Add performance metrics
        performance_info = {
            "expected_search_time_1k": "10-30ms",
            "expected_search_time_5k": "20-50ms", 
            "expected_search_time_10k": "30-80ms",
            "current_optimization_level": "HIGH" if stats["faiss_enabled"] else "BASIC",
            "recommended_for_size": f"Up to {50000 if stats['faiss_enabled'] else 2000} faces"
        }
        
        return {
            "success": True,
            "stats": stats,
            "performance": performance_info
        }
    except Exception as e:
        logger.error(f"Error getting optimized database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/photo/remove-face-optimized/{user_id}",
           summary="Remove face from optimized database",
           description="Remove a user's face embedding from the FAISS-optimized database")
def remove_face_from_database_optimized(user_id: str):
    """Remove a face embedding from the optimized database"""
    try:
        success = optimized_embedding_db.remove_embedding(user_id)
        
        if success:
            return {
                "success": True,
                "message": f"Face embedding removed from optimized database for user {user_id}",
                "total_faces_in_database": len(optimized_embedding_db.get_all_embeddings()),
                "faiss_index_rebuilt": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"No face embedding found for user {user_id}")
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error removing face from optimized database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/photo/batch-search-optimized",
          summary="Batch search multiple faces",
          description="Search multiple face images against the database in a single optimized request")
def batch_search_optimized(request: dict):
    """Batch search multiple images against the database"""
    start_time = time.time()
    
    try:
        images = request.get("images", [])  # List of base64 images
        user_ids = request.get("user_ids", [])  # Optional user IDs for logging
        threshold = request.get("threshold", 0.6)
        model_name = request.get("model_name", "Facenet")
        max_results_per_image = request.get("max_results_per_image", 5)
        
        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required")
        
        if len(images) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
        
        results = []
        total_processing_time = 0
        
        for i, image_b64 in enumerate(images):
            try:
                # Decode image
                img_array = decode_base64_to_image(image_b64)
                user_id = user_ids[i] if i < len(user_ids) else f"batch_image_{i}"
                
                # Search using optimized method
                result = optimized_embedding_db.compare_with_database(
                    img_array=img_array,
                    model_name=model_name,
                    threshold=threshold,
                    max_results=max_results_per_image
                )
                
                results.append({
                    "image_index": i,
                    "user_id": user_id,
                    "result": result
                })
                
                total_processing_time += result["processing_time"]
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "user_id": user_ids[i] if i < len(user_ids) else f"batch_image_{i}",
                    "error": str(e)
                })
        
        batch_time = time.time() - start_time
        
        # Summary statistics
        successful_searches = len([r for r in results if "result" in r])
        total_matches_found = sum([len(r["result"]["all_matches"]) for r in results if "result" in r])
        
        return {
            "success": True,
            "batch_summary": {
                "total_images": len(images),
                "successful_searches": successful_searches,
                "total_matches_found": total_matches_found,
                "total_processing_time": total_processing_time,
                "batch_time": batch_time,
                "average_time_per_image": total_processing_time / len(images) if images else 0
            },
            "results": results
        }
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in batch search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/photo/optimize-database",
          summary="Optimize database for performance",
          description="Rebuild FAISS index and optimize memory usage")
def optimize_database():
    """Manually trigger database optimization"""
    try:
        start_time = time.time()
        
        # Get initial stats
        initial_stats = optimized_embedding_db.get_stats()
        
        # Rebuild FAISS index
        optimized_embedding_db._build_faiss_index()
        
        # Optimize memory
        optimized_embedding_db.optimize_memory()
        
        # Get final stats
        final_stats = optimized_embedding_db.get_stats()
        
        optimization_time = time.time() - start_time
        
        return {
            "success": True,
            "message": "Database optimization completed",
            "optimization_time": optimization_time,
            "before": initial_stats,
            "after": final_stats
        }
        
    except Exception as e:
        logger.error(f"Error optimizing database: {str(e)}")
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