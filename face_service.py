import os
# El problema es que DeepFace añade /.deepface a cualquier ruta
# Por eso establecemos /tmp como base para que la ruta final sea /tmp/.deepface/weights
os.environ["DEEPFACE_HOME"] = "/tmp"
from deepface import DeepFace
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Crear los directorios necesarios solo una vez al importar el módulo
# os.makedirs("/tmp/.deepface/weights", exist_ok=True)

class FaceService:
    """Service for face recognition operations"""
    
    @staticmethod
    def preload_models(model_name: str, detector_backend: str = "ssd") -> None:
        """Preload face recognition models to improve performance"""
        try:
            logger.info(f"Preloading face model: {model_name} and detector: {detector_backend}")
            # Asegurarnos que DEEPFACE_HOME sigue siendo /tmp
            if os.environ.get("DEEPFACE_HOME") != "/tmp":
                logger.warning(f"DEEPFACE_HOME cambió a {os.environ.get('DEEPFACE_HOME')}, restableciendo a /tmp")
                os.environ["DEEPFACE_HOME"] = "/tmp"
            
            # Precargar el modelo de reconocimiento facial
            DeepFace.build_model(model_name)
            
            # Precargar explícitamente el detector SSD
            if detector_backend == "ssd":
                logger.info("Preloading SSD detector...")
                from deepface.detectors import Ssd
                detector = Ssd.SsdClient()
                _ = detector.build_model()
                logger.info("SSD detector preloaded successfully")
            
            logger.info(f"Models preloaded successfully: {model_name} with detector {detector_backend}")
        except Exception as e:
            logger.error(f"Error preloading models: {str(e)}")
            raise e
    
    @staticmethod
    def compare_faces(
        img1_array: np.ndarray, 
        img2_array: np.ndarray, 
        model_name: str,
        enforce_detection: bool = False,
        detector_backend: str = "ssd"
    ) -> Dict[str, Any]:
        """Compare two face images and return verification result"""
        try:
            # Asegurarnos que DEEPFACE_HOME sigue siendo /tmp
            if os.environ.get("DEEPFACE_HOME") != "/tmp":
                logger.warning(f"DEEPFACE_HOME cambió a {os.environ.get('DEEPFACE_HOME')}, restableciendo a /tmp")
                os.environ["DEEPFACE_HOME"] = "/tmp"
                
            result = DeepFace.verify(
                img1_path=img1_array,
                img2_path=img2_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            return result
        except Exception as e:
            logger.error(f"Error in face comparison: {str(e)}")
            raise e
    
    @staticmethod
    def extract_face_embedding(
        img_array: np.ndarray,
        model_name: str,
        enforce_detection: bool,
        detector_backend: str
    ) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            # Asegurarnos que DEEPFACE_HOME sigue siendo /tmp
            if os.environ.get("DEEPFACE_HOME") != "/tmp":
                logger.warning(f"DEEPFACE_HOME cambió a {os.environ.get('DEEPFACE_HOME')}, restableciendo a /tmp")
                os.environ["DEEPFACE_HOME"] = "/tmp"
                
            embedding = DeepFace.represent(
                img_path=img_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            return embedding
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            raise e
    
    @staticmethod
    def compare_with_database(
        img_array: np.ndarray,
        database_embeddings: List[Dict[str, Any]],
        model_name: str,
        threshold: float = 0.6,
        enforce_detection: bool = False,
        detector_backend: str = "ssd"
    ) -> Dict[str, Any]:
        """
        Compare a face image against a database of precomputed embeddings
        
        Args:
            img_array: Input image as numpy array
            database_embeddings: List of dicts with 'user_id' and 'embedding' keys
            model_name: Face recognition model to use
            threshold: Similarity threshold (lower distance = higher similarity)
            enforce_detection: Whether to enforce face detection
            detector_backend: Face detection backend
            
        Returns:
            Dict with comparison results
        """
        start_time = time.time()
        
        try:
            # Extract embedding from input image
            input_embedding = FaceService.extract_face_embedding(
                img_array=img_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            
            # DeepFace.represent returns a list, get the first embedding
            if isinstance(input_embedding, list) and len(input_embedding) > 0:
                input_embedding = np.array(input_embedding[0]["embedding"])
            else:
                raise ValueError("No face embedding could be extracted from input image")
            
            matches = []
            min_distance = float('inf')
            best_match = None
            
            # Compare with each embedding in database
            for db_entry in database_embeddings:
                db_embedding = np.array(db_entry["embedding"])
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(input_embedding - db_embedding)
                
                # Convert distance to similarity score (0-1, where 1 is perfect match)
                # Using inverse exponential decay for more intuitive scoring
                similarity_score = 1 / (1 + distance)
                
                # Check if this is a match based on threshold
                if distance <= threshold:
                    match = {
                        "user_id": db_entry["user_id"],
                        "similarity_score": float(similarity_score),
                        "distance": float(distance)
                    }
                    matches.append(match)
                    
                    # Track best match
                    if distance < min_distance:
                        min_distance = distance
                        best_match = match
            
            # Sort matches by similarity score (descending)
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            processing_time = time.time() - start_time
            
            return {
                "access_granted": len(matches) > 0,
                "match_found": len(matches) > 0,
                "best_match": best_match,
                "all_matches": matches,
                "processing_time": processing_time,
                "total_faces_compared": len(database_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error in database face comparison: {str(e)}")
            raise e

class EmbeddingDatabase:
    """In-memory database for face embeddings with persistence"""
    
    def __init__(self, storage_path: str = "face_embeddings.json"):
        self.storage_path = Path(storage_path)
        self.embeddings: List[Dict[str, Any]] = []
        self.load_embeddings()
    
    def load_embeddings(self) -> None:
        """Load embeddings from persistent storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.embeddings = data.get("embeddings", [])
                logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.storage_path}")
            else:
                logger.info("No existing embeddings file found, starting with empty database")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            self.embeddings = []
    
    def save_embeddings(self) -> None:
        """Save embeddings to persistent storage"""
        try:
            data = {
                "embeddings": self.embeddings,
                "last_updated": time.time()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.embeddings)} embeddings to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def add_embedding(self, user_id: str, embedding: List[float], metadata: Dict = None) -> None:
        """Add or update an embedding in the database"""
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
        self.save_embeddings()
        logger.info(f"Added embedding for user {user_id}")
    
    def remove_embedding(self, user_id: str) -> bool:
        """Remove an embedding from the database"""
        original_count = len(self.embeddings)
        self.embeddings = [e for e in self.embeddings if e["user_id"] != user_id]
        
        if len(self.embeddings) < original_count:
            self.save_embeddings()
            logger.info(f"Removed embedding for user {user_id}")
            return True
        return False
    
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings in the database"""
        return self.embeddings.copy()
    
    def get_embedding(self, user_id: str) -> Dict[str, Any]:
        """Get embedding for a specific user"""
        for embedding in self.embeddings:
            if embedding["user_id"] == user_id:
                return embedding
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_embeddings": len(self.embeddings),
            "storage_path": str(self.storage_path),
            "storage_exists": self.storage_path.exists(),
            "users": [e["user_id"] for e in self.embeddings]
        }

# Global instance of embedding database
embedding_db = EmbeddingDatabase()