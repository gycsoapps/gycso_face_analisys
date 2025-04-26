import base64
import io
import numpy as np
from PIL import Image
import logging
from botocore.exceptions import ClientError
from fastapi import HTTPException
from functools import lru_cache
import boto3
from config import get_settings

logger = logging.getLogger(__name__)

def get_s3_client():
    """Get AWS S3 client with credentials from settings"""
    settings = get_settings()
    # return boto3.client(
    #     's3',
    #     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    #     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    #     region_name=settings.AWS_REGION
    # )
    return boto3.client('s3')

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array for DeepFace"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
            # Convert to RGB (removes alpha channel if present)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed (for OpenCV/DeepFace)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array[:, :, ::-1]
            
        return img_array
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@lru_cache(maxsize=100)
def get_image_from_s3_cached(s3_key: str) -> np.ndarray:
    """Get image from S3 with caching for frequently accessed images"""
    try:
        return get_image_from_s3(s3_key)
    except HTTPException as http_exc:
        # Ensure HTTPExceptions are propagated with their status code
        raise http_exc
    except Exception as e:
        # For any other exception, wrap it in a 500 error
        logger.error(f"Unexpected error in cached S3 access: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Unexpected error accessing cached S3 image: {str(e)}")

def get_image_from_s3(s3_key: str) -> np.ndarray:
    """Get image from S3 and return as numpy array"""
    settings = get_settings()
    
    s3_client = get_s3_client()
    # Get object from S3
    try:
        response = s3_client.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        image_data = response['Body'].read()
    except ClientError as e:
        # Check if error is 'not found'
        if e.response['Error']['Code'] == 'NoSuchKey' or e.response['Error']['Code'] == '404':
            logger.error(f"Image not found in S3: {s3_key}")
            raise HTTPException(status_code=404, 
                               detail=f"Image not found in S3 bucket: {s3_key}")
        else:
            logger.error(f"S3 error: {str(e)}")
            raise HTTPException(status_code=500, 
                               detail=f"Error accessing S3 image: {str(e)}")
    
    # Convert to numpy array
    try:
        image = Image.open(io.BytesIO(image_data))
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array[:, :, ::-1]
            
        return img_array
    except Exception as e:
        logger.error(f"Error processing image data: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Error processing image from S3: {str(e)}") 
    
