"""
Script to create and upload sample test data to Azure Blob Storage
Run this script to prepare test data for the CI/CD pipeline

For the gender_googlenet model, this creates synthetic test images
"""
import os
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from PIL import Image
import io
import json

load_dotenv()

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TEST_DATA_CONTAINER = "testdata"
TEST_DATA_BLOB = "test_samples.npz"  # Changed to .npz for image data
TEST_METADATA_BLOB = "test_metadata.json"

def create_sample_test_data(n_samples=100):
    """
    Create sample test data for gender classification model testing
    GenderNet expects images in shape [batch, 3, 224, 224]
    
    Args:
        n_samples: Number of test samples (images)
    
    Returns:
        Tuple of (images_array, labels_array)
    """
    print(f"Creating {n_samples} test image samples for gender classification...")
    
    np.random.seed(42)
    
    # Generate random images in the format expected by gender_googlenet
    # Shape: [n_samples, channels=3, height=224, width=224]
    # Values normalized similar to ImageNet preprocessing
    images = np.random.randn(n_samples, 3, 224, 224).astype(np.float32)
    
    # Normalize to roughly match ImageNet statistics (mean~0, std~1)
    # This is synthetic data, so we'll use simple normalization
    images = images * 0.5 + 0.5  # Scale to [0, 1] range roughly
    
    # Generate random binary labels (0=female, 1=male)
    labels = np.random.randint(0, 2, size=n_samples)
    
    print(f"Test data created:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution: 0 (female): {np.sum(labels==0)}, 1 (male): {np.sum(labels==1)}")
    
    return images, labels


def upload_to_azure(images, labels, container_name, blob_name, metadata_blob):
    """
    Upload image data to Azure Blob Storage as .npz file
    
    Args:
        images: NumPy array of images
        labels: NumPy array of labels
        container_name: Azure container name
        blob_name: Blob name for data file
        metadata_blob: Blob name for metadata JSON
    """
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in environment variables")
    
    print(f"Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    
    # Create container if it doesn't exist
    try:
        container_client = blob_service_client.create_container(container_name)
        print(f"Container '{container_name}' created")
    except Exception as e:
        print(f"Container '{container_name}' already exists or error: {e}")
        container_client = blob_service_client.get_container_client(container_name)
    
    # Save images and labels to .npz format in memory
    buffer = io.BytesIO()
    np.savez_compressed(buffer, images=images, labels=labels)
    buffer.seek(0)
    npz_data = buffer.getvalue()
    
    # Upload data blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    
    print(f"Uploading data to {container_name}/{blob_name}...")
    blob_client.upload_blob(npz_data, overwrite=True)
    print(f"✓ Data upload successful! ({len(npz_data)} bytes)")
    
    # Create and upload metadata
    metadata = {
        "n_samples": len(images),
        "image_shape": list(images.shape),
        "labels_shape": list(labels.shape),
        "data_type": "float32",
        "model": "gender_googlenet",
        "description": "Synthetic test images for gender classification model",
        "label_mapping": {"0": "female", "1": "male"}
    }
    
    metadata_json = json.dumps(metadata, indent=2)
    metadata_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=metadata_blob
    )
    
    print(f"Uploading metadata to {container_name}/{metadata_blob}...")
    metadata_client.upload_blob(metadata_json, overwrite=True)
    
    print(f"✓ Upload complete!")
    print(f"  Data URL: https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}")
    print(f"  Metadata URL: https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{metadata_blob}")


def verify_upload(container_name, blob_name):
    """
    Verify that the blob was uploaded successfully
    
    Args:
        container_name: Azure container name
        blob_name: Blob name to verify
    """
    print(f"\nVerifying upload...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    
    # Check if blob exists
    if blob_client.exists():
        properties = blob_client.get_blob_properties()
        print(f"✓ Blob exists: {blob_name}")
        print(f"  Size: {properties.size} bytes ({properties.size / 1024 / 1024:.2f} MB)")
        print(f"  Last modified: {properties.last_modified}")
        
        # Download and verify content if it's .npz
        if blob_name.endswith('.npz'):
            download_data = blob_client.download_blob().readall()
            buffer = io.BytesIO(download_data)
            data = np.load(buffer)
            print(f"  Arrays in file: {list(data.keys())}")
            if 'images' in data:
                print(f"  Images shape: {data['images'].shape}")
            if 'labels' in data:
                print(f"  Labels shape: {data['labels'].shape}")
        
        return True
    else:
        print(f"✗ Blob does not exist: {blob_name}")
        return False


def main():
    """Main function to create and upload test data"""
    print("=" * 60)
    print("Azure Blob Storage - Gender Classification Test Data Setup")
    print("=" * 60)
    print()
    
    # Create test data (synthetic images)
    images, labels = create_sample_test_data(n_samples=100)
    
    # Save locally (optional)
    local_path = "test_data/test_samples.npz"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    np.savez_compressed(local_path, images=images, labels=labels)
    print(f"\n✓ Saved locally to: {local_path}")
    
    # Upload to Azure
    print()
    upload_to_azure(images, labels, TEST_DATA_CONTAINER, TEST_DATA_BLOB, TEST_METADATA_BLOB)
    
    # Verify upload
    verify_upload(TEST_DATA_CONTAINER, TEST_DATA_BLOB)
    verify_upload(TEST_DATA_CONTAINER, TEST_METADATA_BLOB)
    
    print()
    print("=" * 60)
    print("Setup complete! Test data ready for CI/CD pipeline.")
    print("Note: The tests can also generate synthetic data automatically")
    print("if Azure test data is not available.")
    print("=" * 60)


if __name__ == "__main__":
    main()
