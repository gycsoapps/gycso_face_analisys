import os
# El problema es que DeepFace añade /.deepface a cualquier ruta
# Por eso establecemos /tmp como base para que la ruta final sea /tmp/.deepface/weights
os.environ["DEEPFACE_HOME"] = "/tmp"
from deepface import DeepFace
import numpy as np
import logging
from typing import Dict, Any

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