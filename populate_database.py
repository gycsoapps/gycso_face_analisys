#!/usr/bin/env python3
"""
Script para poblar la base de datos de embeddings faciales desde un directorio de imágenes
Uso: python populate_database.py --images_dir /path/to/images --model_name Facenet
"""

import os
import argparse
import base64
import logging
from pathlib import Path
from PIL import Image
import io
from face_service import FaceService, embedding_db

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def image_to_base64(image_path: str) -> str:
    """Convierte una imagen a base64"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode('utf-8')
            return base64_str
    except Exception as e:
        logger.error(f"Error converting image {image_path} to base64: {str(e)}")
        return None

def process_image_directory(images_dir: str, model_name: str = "Facenet", detector_backend: str = "ssd"):
    """
    Procesa todas las imágenes en un directorio y añade sus embeddings a la base de datos
    
    Args:
        images_dir: Directorio con las imágenes
        model_name: Modelo de reconocimiento facial a usar
        detector_backend: Backend de detección facial
    """
    
    # Extensiones de imagen soportadas
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.error(f"Directory {images_dir} does not exist")
        return
    
    # Buscar todas las imágenes
    image_files = []
    for ext in supported_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} images in {images_dir}")
    
    successful_adds = 0
    failed_adds = 0
    
    for image_path in image_files:
        try:
            # Usar el nombre del archivo (sin extensión) como user_id
            user_id = image_path.stem
            
            logger.info(f"Processing {image_path} for user {user_id}")
            
            # Convertir imagen a base64
            image_b64 = image_to_base64(str(image_path))
            if not image_b64:
                failed_adds += 1
                continue
            
            # Decodificar imagen
            from utils import decode_base64_to_image
            img_array = decode_base64_to_image(image_b64)
            
            # Extraer embedding
            embedding_result = FaceService.extract_face_embedding(
                img_array=img_array,
                model_name=model_name,
                enforce_detection=True,  # Requerir detección para asegurar calidad
                detector_backend=detector_backend
            )
            
            # Obtener el embedding
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                embedding = embedding_result[0]["embedding"]
            else:
                logger.warning(f"No face found in {image_path}")
                failed_adds += 1
                continue
            
            # Metadatos adicionales
            metadata = {
                "source_file": str(image_path),
                "image_size": os.path.getsize(image_path),
                "model_used": model_name,
                "detector_used": detector_backend
            }
            
            # Añadir a la base de datos
            embedding_db.add_embedding(user_id, embedding, metadata)
            successful_adds += 1
            logger.info(f"✓ Successfully added {user_id}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {image_path}: {str(e)}")
            failed_adds += 1
    
    logger.info(f"Processing complete. Success: {successful_adds}, Failed: {failed_adds}")
    logger.info(f"Total embeddings in database: {len(embedding_db.get_all_embeddings())}")

def batch_test_recognition(test_images_dir: str, threshold: float = 0.6, model_name: str = "Facenet"):
    """
    Prueba el reconocimiento contra imágenes de test
    
    Args:
        test_images_dir: Directorio con imágenes de prueba
        threshold: Umbral de similitud
        model_name: Modelo a usar
    """
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    test_dir = Path(test_images_dir)
    
    if not test_dir.exists():
        logger.error(f"Test directory {test_dir} does not exist")
        return
    
    # Buscar imágenes de test
    test_files = []
    for ext in supported_extensions:
        test_files.extend(test_dir.glob(f"*{ext}"))
        test_files.extend(test_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"Testing recognition with {len(test_files)} test images")
    
    for test_image in test_files:
        try:
            expected_user = test_image.stem
            logger.info(f"Testing {test_image} (expected: {expected_user})")
            
            # Convertir a base64 y comparar con base de datos
            image_b64 = image_to_base64(str(test_image))
            if not image_b64:
                continue
                
            from utils import decode_base64_to_image
            img_array = decode_base64_to_image(image_b64)
            
            # Comparar con base de datos
            database_embeddings = embedding_db.get_all_embeddings()
            result = FaceService.compare_with_database(
                img_array=img_array,
                database_embeddings=database_embeddings,
                model_name=model_name,
                threshold=threshold
            )
            
            if result["match_found"]:
                best_match = result["best_match"]
                logger.info(f"  ✓ Match found: {best_match['user_id']} "
                           f"(similarity: {best_match['similarity_score']:.3f}, "
                           f"distance: {best_match['distance']:.3f})")
                
                if best_match['user_id'] == expected_user:
                    logger.info("  ✓ Correct match!")
                else:
                    logger.warning(f"  ⚠ Wrong match! Expected {expected_user}")
            else:
                logger.warning(f"  ✗ No match found for {expected_user}")
                
        except Exception as e:
            logger.error(f"Error testing {test_image}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Populate face embedding database")
    parser.add_argument("--images_dir", required=True, help="Directory containing face images")
    parser.add_argument("--model_name", default="Facenet", help="Face recognition model to use")
    parser.add_argument("--detector_backend", default="ssd", help="Face detection backend")
    parser.add_argument("--test_dir", help="Directory with test images for validation")
    parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold")
    parser.add_argument("--clear_db", action="store_true", help="Clear existing database before adding")
    
    args = parser.parse_args()
    
    # Precargar modelos
    logger.info("Preloading face recognition models...")
    FaceService.preload_models(args.model_name, args.detector_backend)
    
    # Limpiar base de datos si se solicita
    if args.clear_db:
        logger.info("Clearing existing database...")
        embedding_db.embeddings = []
        embedding_db.save_embeddings()
    
    # Mostrar estadísticas iniciales
    initial_stats = embedding_db.get_stats()
    logger.info(f"Initial database stats: {initial_stats['total_embeddings']} embeddings")
    
    # Procesar imágenes
    process_image_directory(args.images_dir, args.model_name, args.detector_backend)
    
    # Pruebas si se especifica directorio de test
    if args.test_dir:
        logger.info("Running recognition tests...")
        batch_test_recognition(args.test_dir, args.threshold, args.model_name)

if __name__ == "__main__":
    main() 