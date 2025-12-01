#!/usr/bin/env python3
"""
Download ONNX model from Azure Blob Storage
Used during Docker build to cache the model in the image
"""
import os
import sys
import requests

def download_model():
    """Download model from MODEL_URL to MODEL_PATH"""
    model_url = os.getenv('MODEL_URL')
    model_path = os.getenv('MODEL_PATH')
    
    if not model_url or not model_path:
        print("ERROR: MODEL_URL and MODEL_PATH environment variables must be set")
        sys.exit(1)
    
    print(f"Downloading model from: {model_url}")
    print(f"Saving to: {model_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        # Download with streaming to handle large files
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size:
            print(f"File size: {total_size / 1024 / 1024:.2f} MB")
        
        # Write to file
        downloaded = 0
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size and downloaded % (1024 * 1024) == 0:  # Progress every MB
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%")
        
        print(f"✓ Model downloaded successfully to {model_path}")
        print(f"  Total size: {downloaded / 1024 / 1024:.2f} MB")
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
