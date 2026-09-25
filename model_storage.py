"""
Model Storage Module - Supports both Vercel Blob (production) and local filesystem (development)
"""

import os
import io
import joblib
from typing import Optional, Dict, Any

# Try to import Vercel Blob client
try:
    from vercel.blob import BlobClient
    BLOB_AVAILABLE = True
except ImportError:
    BLOB_AVAILABLE = False

# Configuration
IS_VERCEL = bool(os.environ.get("VERCEL"))
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")

# Local filesystem fallback
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/tmp/keystroke_mfa" if IS_VERCEL else SCRIPT_DIR
)
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_MODEL_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Determine storage type
USE_BLOB = bool(BLOB_READ_WRITE_TOKEN) and BLOB_AVAILABLE and IS_VERCEL


def get_model_path(user_id: int) -> str:
    """
    Get the model path for a user.
    For Blob: returns the blob path (e.g., "models/user_1_model.pkl")
    For local: returns the local file path
    """
    if USE_BLOB:
        return f"models/user_{user_id}_model.pkl"
    else:
        return os.path.join(LOCAL_MODEL_DIR, f"user_{user_id}_model.pkl")


def save_model(user_id: int, model_data: Dict[str, Any]) -> bool:
    """
    Save a trained model for a user.
    
    Args:
        user_id: The user ID
        model_data: Dictionary containing the model and threshold
                   {"model": model, "threshold": threshold}
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert model to bytes
        buffer = io.BytesIO()
        joblib.dump(model_data, buffer)
        model_bytes = buffer.getvalue()
        
        if USE_BLOB:
            # Upload to Vercel Blob
            client = BlobClient()
            blob_path = get_model_path(user_id)
            
            client.put(
                blob_path,
                model_bytes,
                access="private",
                content_type="application/octet-stream",
                overwrite=True
            )
            print(f"Model saved to Vercel Blob: {blob_path}")
        else:
            # Save to local filesystem
            model_path = get_model_path(user_id)
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            print(f"Model saved to local filesystem: {model_path}")
        
        return True
    except Exception as e:
        print(f"Error saving model for user {user_id}: {e}")
        return False


def load_model(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Load a trained model for a user.
    
    Args:
        user_id: The user ID
    
    Returns:
        Dictionary containing the model and threshold, or None if not found
    """
    try:
        if USE_BLOB:
            # Download from Vercel Blob
            client = BlobClient()
            blob_path = get_model_path(user_id)
            
            blob = client.get(blob_path)
            model_bytes = blob.body
            
            # Load from bytes
            buffer = io.BytesIO(model_bytes)
            model_data = joblib.load(buffer)
            print(f"Model loaded from Vercel Blob: {blob_path}")
        else:
            # Load from local filesystem
            model_path = get_model_path(user_id)
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)
            print(f"Model loaded from local filesystem: {model_path}")
        
        return model_data
    except Exception as e:
        print(f"Error loading model for user {user_id}: {e}")
        return None


def delete_model(user_id: int) -> bool:
    """
    Delete a trained model for a user.
    
    Args:
        user_id: The user ID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if USE_BLOB:
            # Delete from Vercel Blob
            client = BlobClient()
            blob_path = get_model_path(user_id)
            client.delete(blob_path)
            print(f"Model deleted from Vercel Blob: {blob_path}")
        else:
            # Delete from local filesystem
            model_path = get_model_path(user_id)
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"Model deleted from local filesystem: {model_path}")
            else:
                print(f"Model file not found for deletion: {model_path}")
        
        return True
    except Exception as e:
        print(f"Error deleting model for user {user_id}: {e}")
        return False


def model_exists(user_id: int) -> bool:
    """
    Check if a model exists for a user.
    
    Args:
        user_id: The user ID
    
    Returns:
        True if model exists, False otherwise
    """
    try:
        if USE_BLOB:
            # Check Vercel Blob
            client = BlobClient()
            blob_path = get_model_path(user_id)
            blob = client.get(blob_path)
            return blob is not None
        else:
            # Check local filesystem
            model_path = get_model_path(user_id)
            return os.path.exists(model_path)
    except Exception:
        return False


def get_storage_info() -> Dict[str, str]:
    """
    Get information about the current storage configuration.
    
    Returns:
        Dictionary with storage information
    """
    return {
        "storage_type": "vercel_blob" if USE_BLOB else "local_filesystem",
        "is_vercel": IS_VERCEL,
        "blob_available": BLOB_AVAILABLE,
        "blob_token_configured": bool(BLOB_READ_WRITE_TOKEN),
        "local_model_dir": LOCAL_MODEL_DIR
    }
