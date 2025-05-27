#!/usr/bin/env python3
"""
Script optimizado para poblar bases de datos grandes (5K-10K+ imágenes)
Características:
- Procesamiento paralelo
- Progress tracking
- Batch processing
- FAISS indexing automático
- Manejo de memoria optimizado
"""

import os
import argparse
import base64
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import multiprocessing
from tqdm import tqdm
import json

from face_service_optimized import OptimizedFaceService, optimized_embedding_db
from config import get_settings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = get_settings()

def image_to_base64(image_path: str) -> str:
    """Convierte una imagen a base64"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image {image_path} to base64: {str(e)}")
        return None

def process_single_image(args: Tuple[Path, str, str, bool]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Procesa una sola imagen y extrae su embedding
    
    Args:
        args: Tupla con (image_path, model_name, detector_backend, enforce_detection)
        
    Returns:
        Tupla con (success, user_id, embedding_data_or_error)
    """
    image_path, model_name, detector_backend, enforce_detection = args
    
    try:
        user_id = image_path.stem
        
        # Convertir imagen a base64
        image_b64 = image_to_base64(str(image_path))
        if not image_b64:
            return False, user_id, {"error": "Could not convert to base64"}
        
        # Decodificar imagen
        from utils import decode_base64_to_image
        img_array = decode_base64_to_image(image_b64)
        
        # Extraer embedding
        embedding_result = OptimizedFaceService.extract_face_embedding(
            img_array=img_array,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend
        )
        
        # Obtener el embedding
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            embedding = embedding_result[0]["embedding"]
        else:
            return False, user_id, {"error": "No face found in image"}
        
        # Metadatos
        metadata = {
            "source_file": str(image_path),
            "image_size": os.path.getsize(image_path),
            "model_used": model_name,
            "detector_used": detector_backend,
            "processed_at": time.time()
        }
        
        embedding_data = {
            "user_id": user_id,
            "embedding": embedding,
            "metadata": metadata
        }
        
        return True, user_id, embedding_data
        
    except Exception as e:
        return False, user_id, {"error": str(e)}

def process_images_parallel(
    image_files: List[Path], 
    model_name: str = "Facenet", 
    detector_backend: str = "ssd",
    enforce_detection: bool = True,
    max_workers: int = None,
    batch_size: int = 100
) -> Tuple[List[Dict], List[Dict]]:
    """
    Procesa imágenes en paralelo con progress tracking
    
    Returns:
        Tupla con (successful_embeddings, failed_embeddings)
    """
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), settings.MAX_WORKERS)
    
    logger.info(f"Processing {len(image_files)} images with {max_workers} workers in batches of {batch_size}")
    
    successful_embeddings = []
    failed_embeddings = []
    
    # Procesar en batches para optimizar memoria
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}")
        
        # Preparar argumentos para el batch
        batch_args = [
            (image_path, model_name, detector_backend, enforce_detection) 
            for image_path in batch_files
        ]
        
        # Procesar batch en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar trabajos
            future_to_path = {
                executor.submit(process_single_image, args): args[0] 
                for args in batch_args
            }
            
            # Recoger resultados con progress bar
            with tqdm(total=len(batch_args), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                for future in as_completed(future_to_path):
                    image_path = future_to_path[future]
                    try:
                        success, user_id, data = future.result()
                        
                        if success:
                            successful_embeddings.append(data)
                        else:
                            failed_embeddings.append({
                                "user_id": user_id,
                                "image_path": str(image_path),
                                "error": data.get("error", "Unknown error")
                            })
                            
                    except Exception as e:
                        failed_embeddings.append({
                            "user_id": image_path.stem,
                            "image_path": str(image_path),
                            "error": f"Processing exception: {str(e)}"
                        })
                    
                    pbar.update(1)
        
        # Añadir embeddings al database por batch
        if successful_embeddings:
            logger.info(f"Adding {len(successful_embeddings)} embeddings to database...")
            batch_start_time = time.time()
            
            for embedding_data in successful_embeddings:
                optimized_embedding_db.add_embedding(
                    embedding_data["user_id"],
                    embedding_data["embedding"],
                    embedding_data["metadata"]
                )
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch added to database in {batch_time:.2f} seconds")
            
            # Limpiar memoria
            successful_embeddings.clear()
            
            # Optimizar memoria cada pocos batches
            if (batch_start // batch_size + 1) % 5 == 0:
                optimized_embedding_db.optimize_memory()
    
    return successful_embeddings, failed_embeddings

def process_large_database(
    images_dir: str, 
    model_name: str = "Facenet", 
    detector_backend: str = "ssd",
    max_workers: int = None,
    batch_size: int = 100,
    save_progress: bool = True
):
    """
    Procesa un directorio completo de imágenes para base de datos grande
    """
    
    # Extensiones soportadas
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.error(f"Directory {images_dir} does not exist")
        return
    
    # Buscar todas las imágenes
    logger.info("Scanning for images...")
    image_files = []
    for ext in supported_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} images in {images_dir}")
    
    if len(image_files) == 0:
        logger.warning("No images found!")
        return
    
    # Configurar progreso
    progress_file = images_dir / "processing_progress.json"
    start_index = 0
    
    if save_progress and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_index = progress_data.get("last_processed_index", 0)
                logger.info(f"Resuming from image {start_index}")
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
    
    # Filtrar imágenes ya procesadas
    remaining_files = image_files[start_index:]
    
    if not remaining_files:
        logger.info("All images already processed!")
        return
    
    logger.info(f"Processing {len(remaining_files)} remaining images")
    
    # Estadísticas iniciales
    initial_stats = optimized_embedding_db.get_stats()
    logger.info(f"Initial database has {initial_stats['total_embeddings']} embeddings")
    
    start_time = time.time()
    
    # Procesar imágenes
    successful_embeddings, failed_embeddings = process_images_parallel(
        remaining_files,
        model_name=model_name,
        detector_backend=detector_backend,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    total_time = time.time() - start_time
    
    # Estadísticas finales
    final_stats = optimized_embedding_db.get_stats()
    
    # Rebuilding FAISS index final
    logger.info("Building final optimized FAISS index...")
    index_start = time.time()
    optimized_embedding_db._build_faiss_index()
    index_time = time.time() - index_start
    
    # Resumen
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"FAISS index build time: {index_time:.2f} seconds")
    logger.info(f"Images processed: {len(remaining_files)}")
    logger.info(f"Failed: {len(failed_embeddings)}")
    logger.info(f"Database size: {initial_stats['total_embeddings']} → {final_stats['total_embeddings']}")
    logger.info(f"Average time per image: {total_time/len(remaining_files):.2f} seconds")
    
    if final_stats.get("memory_usage_mb"):
        logger.info(f"Memory usage: {final_stats['memory_usage_mb']:.1f} MB")
    
    # Guardar errores si los hay
    if failed_embeddings:
        error_file = images_dir / "processing_errors.json"
        with open(error_file, 'w') as f:
            json.dump(failed_embeddings, f, indent=2)
        logger.warning(f"Errors saved to {error_file}")
    
    # Guardar progreso completado
    if save_progress:
        progress_data = {
            "completed": True,
            "total_images": len(image_files),
            "last_processed_index": len(image_files),
            "completion_time": time.time(),
            "final_stats": final_stats
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

def benchmark_search_performance(num_test_searches: int = 100):
    """
    Realiza benchmark de rendimiento de búsqueda
    """
    stats = optimized_embedding_db.get_stats()
    if stats["total_embeddings"] == 0:
        logger.warning("No embeddings in database for benchmarking")
        return
    
    logger.info(f"Benchmarking search performance with {stats['total_embeddings']} embeddings...")
    
    # Obtener algunos embeddings para pruebas
    all_embeddings = optimized_embedding_db.get_all_embeddings()
    test_embeddings = all_embeddings[:min(num_test_searches, len(all_embeddings))]
    
    search_times = []
    
    for i, test_embedding in enumerate(test_embeddings):
        if i % 20 == 0:
            logger.info(f"Benchmark progress: {i}/{len(test_embeddings)}")
        
        # Simular búsqueda
        start_time = time.time()
        
        # Usar el embedding como query (simula búsqueda)
        matches = optimized_embedding_db.fast_search(
            np.array(test_embedding["embedding"], dtype=np.float32),
            k=10,
            threshold=0.8
        )
        
        search_time = time.time() - start_time
        search_times.append(search_time * 1000)  # Convert to ms
    
    # Estadísticas de rendimiento
    avg_time = sum(search_times) / len(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    p95_time = sorted(search_times)[int(0.95 * len(search_times))]
    
    logger.info("="*60)
    logger.info("SEARCH PERFORMANCE BENCHMARK")
    logger.info("="*60)
    logger.info(f"Database size: {stats['total_embeddings']} embeddings")
    logger.info(f"Test searches: {len(search_times)}")
    logger.info(f"Search method: {'FAISS' if stats['faiss_enabled'] else 'Linear'}")
    logger.info(f"Average search time: {avg_time:.1f} ms")
    logger.info(f"Min search time: {min_time:.1f} ms")
    logger.info(f"Max search time: {max_time:.1f} ms") 
    logger.info(f"95th percentile: {p95_time:.1f} ms")
    logger.info(f"Throughput: {1000/avg_time:.1f} searches/second")

def main():
    parser = argparse.ArgumentParser(description="Optimized large-scale face embedding database population")
    parser.add_argument("--images_dir", required=True, help="Directory containing face images")
    parser.add_argument("--model_name", default="Facenet", help="Face recognition model")
    parser.add_argument("--detector_backend", default="ssd", help="Face detection backend")
    parser.add_argument("--max_workers", type=int, help="Maximum parallel workers")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--clear_db", action="store_true", help="Clear existing database")
    parser.add_argument("--enable_hdf5", action="store_true", help="Use HDF5 storage for large datasets")
    parser.add_argument("--enable_redis", action="store_true", help="Enable Redis caching")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark after processing")
    parser.add_argument("--benchmark_only", action="store_true", help="Only run benchmark (skip processing)")
    
    args = parser.parse_args()
    
    # Configurar optimizaciones
    if args.enable_hdf5:
        os.environ["USE_HDF5_STORAGE"] = "true"
        logger.info("HDF5 storage enabled")
    
    if args.enable_redis:
        os.environ["USE_REDIS_CACHE"] = "true"
        logger.info("Redis caching enabled")
    
    # Precargar modelos
    if not args.benchmark_only:
        logger.info("Preloading optimized face recognition models...")
        OptimizedFaceService.preload_models(args.model_name, args.detector_backend)
    
    # Limpiar base de datos si se solicita
    if args.clear_db and not args.benchmark_only:
        logger.info("Clearing existing database...")
        optimized_embedding_db.embeddings = []
        optimized_embedding_db.save_embeddings()
        # Eliminar archivos de índice
        if Path(settings.FAISS_INDEX_PATH).exists():
            Path(settings.FAISS_INDEX_PATH).unlink()
    
    # Solo benchmark
    if args.benchmark_only:
        benchmark_search_performance()
        return
    
    # Estadísticas iniciales
    initial_stats = optimized_embedding_db.get_stats()
    logger.info(f"Initial optimized database stats: {initial_stats['total_embeddings']} embeddings")
    logger.info(f"FAISS enabled: {initial_stats['faiss_enabled']}")
    logger.info(f"HDF5 enabled: {initial_stats['hdf5_enabled']}")
    logger.info(f"Redis enabled: {initial_stats['redis_enabled']}")
    
    # Procesar imágenes
    process_large_database(
        args.images_dir,
        args.model_name,
        args.detector_backend,
        args.max_workers,
        args.batch_size
    )
    
    # Benchmark final si se solicita
    if args.benchmark:
        benchmark_search_performance()

if __name__ == "__main__":
    import numpy as np
    main() 