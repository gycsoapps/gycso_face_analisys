import os
import time
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up environment
os.environ["DEEPFACE_HOME"] = "/tmp"

import numpy as np
import faiss
import h5py
from deepface import DeepFace
from sklearn.preprocessing import normalize
from config import get_settings

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)
settings = get_settings()

class OptimizedFaceService:
    """Optimized face service for large-scale databases (5K-10K+ faces)"""
    
    @staticmethod
    def preload_models(model_name: str, detector_backend: str = "ssd") -> None:
        """Preload face recognition models to improve performance"""
        try:
            logger.info(f"Preloading optimized face model: {model_name} and detector: {detector_backend}")
            
            # Set DeepFace home
            if os.environ.get("DEEPFACE_HOME") != "/tmp":
                os.environ["DEEPFACE_HOME"] = "/tmp"
            
            # Preload models
            DeepFace.build_model(model_name)
            
            if detector_backend == "ssd":
                from deepface.detectors import Ssd
                detector = Ssd.SsdClient()
                _ = detector.build_model()
                
            logger.info(f"Optimized models preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading optimized models: {str(e)}")
            raise e
    
    @staticmethod
    def extract_face_embedding(
        img_array: np.ndarray,
        model_name: str,
        enforce_detection: bool,
        detector_backend: str
    ) -> np.ndarray:
        """Extract face embedding from image with optimizations"""
        try:
            if os.environ.get("DEEPFACE_HOME") != "/tmp":
                os.environ["DEEPFACE_HOME"] = "/tmp"
                
            embedding = DeepFace.represent(
                img_path=img_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            return embedding
        except Exception as e:
            logger.error(f"Error extracting optimized face embedding: {str(e)}")
            raise e

class FAISSEmbeddingDatabase:
    """High-performance embedding database using FAISS for similarity search"""
    
    def __init__(self, 
                 storage_path: str = "face_embeddings_optimized.json",
                 faiss_index_path: str = None,
                 use_redis: bool = False):
        
        self.storage_path = Path(storage_path)
        self.faiss_index_path = faiss_index_path or settings.FAISS_INDEX_PATH
        self.use_redis = use_redis and REDIS_AVAILABLE
        
        # Initialize data structures
        self.embeddings: List[Dict[str, Any]] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.user_id_map: Dict[int, str] = {}  # Maps FAISS index to user_id
        
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Redis client (optional)
        self.redis_client = None
        if self.use_redis:
            self._init_redis()
        
        # Load existing data
        self.load_embeddings()
        self._build_faiss_index()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.use_redis = False
            self.redis_client = None
    
    def load_embeddings(self) -> None:
        """Load embeddings from persistent storage"""
        try:
            if settings.USE_HDF5_STORAGE and Path(settings.HDF5_FILE_PATH).exists():
                self._load_from_hdf5()
            elif self.storage_path.exists():
                self._load_from_json()
            else:
                logger.info("No existing embeddings file found, starting with empty database")
                
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            self.embeddings = []
    
    def _load_from_json(self):
        """Load from JSON format"""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
            self.embeddings = data.get("embeddings", [])
    
    def _load_from_hdf5(self):
        """Load from HDF5 format (optimized for large datasets)"""
        with h5py.File(settings.HDF5_FILE_PATH, 'r') as f:
            embeddings_data = f['embeddings'][:]
            user_ids = f['user_ids'][:].astype(str)
            metadata_json = f['metadata'][:].astype(str)
            
            self.embeddings = []
            for i, (embedding, user_id, meta_str) in enumerate(zip(embeddings_data, user_ids, metadata_json)):
                metadata = json.loads(meta_str) if meta_str else {}
                self.embeddings.append({
                    "user_id": user_id,
                    "embedding": embedding.tolist(),
                    "metadata": metadata,
                    "created_at": metadata.get("created_at", time.time())
                })
    
    def save_embeddings(self) -> None:
        """Save embeddings to persistent storage"""
        try:
            if settings.USE_HDF5_STORAGE:
                self._save_to_hdf5()
            else:
                self._save_to_json()
                
            logger.info(f"Saved {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def _save_to_json(self):
        """Save to JSON format"""
        data = {
            "embeddings": self.embeddings,
            "last_updated": time.time(),
            "total_count": len(self.embeddings)
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_to_hdf5(self):
        """Save to HDF5 format (optimized for large datasets)"""
        if not self.embeddings:
            return
            
        with h5py.File(settings.HDF5_FILE_PATH, 'w') as f:
            # Prepare data
            embeddings_array = np.array([e["embedding"] for e in self.embeddings])
            user_ids = np.array([e["user_id"] for e in self.embeddings], dtype='S50')
            metadata_json = np.array([json.dumps(e.get("metadata", {})) for e in self.embeddings], dtype='S500')
            
            # Save datasets
            f.create_dataset('embeddings', data=embeddings_array)
            f.create_dataset('user_ids', data=user_ids)
            f.create_dataset('metadata', data=metadata_json)
            f.attrs['last_updated'] = time.time()
            f.attrs['total_count'] = len(self.embeddings)
    
    def _build_faiss_index(self) -> None:
        """Build or load FAISS index for fast similarity search"""
        if not self.embeddings:
            logger.info("No embeddings to index")
            return
        
        try:
            # Check if index file exists
            if Path(self.faiss_index_path).exists():
                self._load_faiss_index()
                return
            
            # Build new index
            logger.info(f"Building FAISS index for {len(self.embeddings)} embeddings...")
            start_time = time.time()
            
            # Create embedding matrix
            self.embedding_matrix = np.array([e["embedding"] for e in self.embeddings], dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            self.embedding_matrix = normalize(self.embedding_matrix, norm='l2')
            
            # Build user ID mapping
            self.user_id_map = {i: self.embeddings[i]["user_id"] for i in range(len(self.embeddings))}
            
            # Create FAISS index based on configuration
            dimension = self.embedding_matrix.shape[1]
            
            if settings.FAISS_INDEX_TYPE == "Flat":
                # Exact search (slower but 100% accurate)
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
                
            elif settings.FAISS_INDEX_TYPE == "IVF":
                # Approximate search with inverted file index
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, settings.FAISS_NLIST)
                
                # Train the index
                self.faiss_index.train(self.embedding_matrix)
                self.faiss_index.nprobe = settings.FAISS_NPROBE
                
            elif settings.FAISS_INDEX_TYPE == "HNSW":
                # Hierarchical Navigable Small World (very fast, good accuracy)
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 100
            
            # Add embeddings to index
            self.faiss_index.add(self.embedding_matrix)
            
            # Save index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            build_time = time.time() - start_time
            logger.info(f"FAISS index built and saved in {build_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            # Fallback to linear search
            self.faiss_index = None
    
    def _load_faiss_index(self):
        """Load existing FAISS index"""
        try:
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            
            # Rebuild embedding matrix and user mapping
            self.embedding_matrix = np.array([e["embedding"] for e in self.embeddings], dtype=np.float32)
            self.embedding_matrix = normalize(self.embedding_matrix, norm='l2')
            self.user_id_map = {i: self.embeddings[i]["user_id"] for i in range(len(self.embeddings))}
            
            logger.info(f"FAISS index loaded from {self.faiss_index_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self._build_faiss_index()
    
    def add_embedding(self, user_id: str, embedding: List[float], metadata: Dict = None) -> None:
        """Add or update an embedding in the database"""
        with self.lock:
            try:
                # Remove existing embedding for this user_id if it exists
                self.embeddings = [e for e in self.embeddings if e["user_id"] != user_id]
                
                # Add new embedding
                embedding_data = {
                    "user_id": user_id,
                    "embedding": embedding,
                    "metadata": metadata or {},
                    "created_at": time.time()
                }
                self.embeddings.append(embedding_data)
                
                # Save to storage
                self.save_embeddings()
                
                # Rebuild FAISS index if it exists
                if self.faiss_index is not None:
                    self._build_faiss_index()
                
                # Cache in Redis if available
                if self.redis_client:
                    cache_key = f"embedding:{user_id}"
                    self.redis_client.setex(cache_key, settings.REDIS_EXPIRE_TIME, json.dumps(embedding_data))
                
                logger.info(f"Added optimized embedding for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error adding embedding: {str(e)}")
                raise e
    
    def remove_embedding(self, user_id: str) -> bool:
        """Remove an embedding from the database"""
        with self.lock:
            try:
                original_count = len(self.embeddings)
                self.embeddings = [e for e in self.embeddings if e["user_id"] != user_id]
                
                if len(self.embeddings) < original_count:
                    self.save_embeddings()
                    
                    # Rebuild FAISS index
                    if self.faiss_index is not None:
                        self._build_faiss_index()
                    
                    # Remove from Redis cache
                    if self.redis_client:
                        cache_key = f"embedding:{user_id}"
                        self.redis_client.delete(cache_key)
                    
                    logger.info(f"Removed optimized embedding for user {user_id}")
                    return True
                    
                return False
                
            except Exception as e:
                logger.error(f"Error removing embedding: {str(e)}")
                return False
    
    def fast_search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Fast similarity search using FAISS index"""
        try:
            if self.faiss_index is None or len(self.embeddings) == 0:
                return []
            
            # Normalize query embedding
            query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')
            
            # Search using FAISS
            similarities, indices = self.faiss_index.search(query_embedding, k)
            
            matches = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Convert similarity to distance (FAISS returns cosine similarity)
                distance = 1.0 - similarity
                
                if distance <= threshold:
                    user_id = self.user_id_map.get(idx)
                    if user_id:
                        similarity_score = 1 / (1 + distance)
                        matches.append({
                            "user_id": user_id,
                            "similarity_score": float(similarity_score),
                            "distance": float(distance),
                            "faiss_similarity": float(similarity)
                        })
            
            # Sort by similarity score (descending)
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matches
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {str(e)}")
            return []
    
    def compare_with_database(self,
                             img_array: np.ndarray,
                             model_name: str,
                             threshold: float = 0.6,
                             enforce_detection: bool = False,
                             detector_backend: str = "ssd",
                             max_results: int = 10) -> Dict[str, Any]:
        """Optimized comparison with large database using FAISS"""
        start_time = time.time()
        
        try:
            # Extract embedding from input image
            input_embedding = OptimizedFaceService.extract_face_embedding(
                img_array=img_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            
            # Process embedding result
            if isinstance(input_embedding, list) and len(input_embedding) > 0:
                input_embedding = np.array(input_embedding[0]["embedding"], dtype=np.float32)
            else:
                raise ValueError("No face embedding could be extracted from input image")
            
            # Use FAISS for fast search if available
            if settings.USE_FAISS_INDEX and self.faiss_index is not None:
                matches = self.fast_search(input_embedding, k=max_results, threshold=threshold)
            else:
                # Fallback to linear search
                matches = self._linear_search(input_embedding, threshold)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            best_match = matches[0] if matches else None
            
            return {
                "access_granted": len(matches) > 0,
                "match_found": len(matches) > 0,
                "best_match": best_match,
                "all_matches": matches[:max_results],
                "processing_time": processing_time,
                "total_faces_compared": len(self.embeddings),
                "search_method": "FAISS" if (settings.USE_FAISS_INDEX and self.faiss_index) else "Linear"
            }
            
        except Exception as e:
            logger.error(f"Error in optimized database comparison: {str(e)}")
            raise e
    
    def _linear_search(self, query_embedding: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """Fallback linear search method"""
        matches = []
        min_distance = float('inf')
        
        for db_entry in self.embeddings:
            db_embedding = np.array(db_entry["embedding"], dtype=np.float32)
            distance = np.linalg.norm(query_embedding - db_embedding)
            
            if distance <= threshold:
                similarity_score = 1 / (1 + distance)
                matches.append({
                    "user_id": db_entry["user_id"],
                    "similarity_score": float(similarity_score),
                    "distance": float(distance)
                })
        
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches
    
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings in the database"""
        return self.embeddings.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            "total_embeddings": len(self.embeddings),
            "storage_path": str(self.storage_path),
            "storage_exists": self.storage_path.exists(),
            "users": [e["user_id"] for e in self.embeddings][:10],  # First 10 users
            "faiss_enabled": settings.USE_FAISS_INDEX,
            "faiss_index_type": settings.FAISS_INDEX_TYPE,
            "faiss_index_exists": Path(self.faiss_index_path).exists(),
            "redis_enabled": self.use_redis,
            "hdf5_enabled": settings.USE_HDF5_STORAGE,
            "memory_optimization": settings.ENABLE_MEMORY_OPTIMIZATION
        }
        
        if self.embedding_matrix is not None:
            stats["embedding_dimension"] = self.embedding_matrix.shape[1]
            stats["memory_usage_mb"] = self.embedding_matrix.nbytes / (1024 * 1024)
        
        return stats
    
    def optimize_memory(self):
        """Optimize memory usage for large datasets"""
        if not settings.ENABLE_MEMORY_OPTIMIZATION:
            return
        
        try:
            # Clear unnecessary data
            import gc
            gc.collect()
            
            # Optionally clear embedding matrix if FAISS index exists
            if self.faiss_index is not None and self.embedding_matrix is not None:
                memory_usage = self.embedding_matrix.nbytes / (1024 * 1024)
                if memory_usage > settings.MAX_MEMORY_USAGE_MB / 2:
                    logger.info(f"Clearing embedding matrix to save {memory_usage:.1f}MB")
                    self.embedding_matrix = None
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Error in memory optimization: {str(e)}")

# Global instance of optimized embedding database
optimized_embedding_db = FAISSEmbeddingDatabase() 