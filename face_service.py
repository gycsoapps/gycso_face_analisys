import os
os.environ["DEEPFACE_HOME"] = "/tmp/.deepface"
from deepface import DeepFace
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FaceService:
    """Service for face recognition operations"""
    
    @staticmethod
    def preload_models(model_name: str) -> None:
        """Preload face recognition models to improve performance"""
        try:
            logger.info(f"Preloading face model: {model_name}")
            DeepFace.build_model(model_name)
            
                
            logger.info(f"Models preloaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Error preloading models: {str(e)}")
            raise e
    
    @staticmethod
    def compare_faces(
        img1_array: np.ndarray, 
        img2_array: np.ndarray, 
        model_name: str,
        enforce_detection: bool = False,
        detector_backend: str = "retinaface"
    ) -> Dict[str, Any]:
        """Compare two face images and return verification result"""
        try:
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