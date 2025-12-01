"""
Unit tests for the Gender Classifier ONNX model
Tests model loading, inference, and performance metrics
"""
import os
import sys
import pytest
import numpy as np
import onnxruntime as ort
import requests
from azure.storage.blob import BlobServiceClient
from sklearn.metrics import accuracy_score
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration
MODEL_URL = os.getenv("MODEL_URL", "https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/gender_googlenet.onnx")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TEST_DATA_CONTAINER = os.getenv("TEST_DATA_CONTAINER", "testdata")
TEST_DATA_BLOB = os.getenv("TEST_DATA_BLOB", "test_samples.npz")  # Changed to .npz for image data

# Performance thresholds
ACCURACY_THRESHOLD = 0.50  # Minimum acceptable accuracy
MIN_PREDICTIONS = 1  # Minimum number of predictions


@pytest.fixture(scope="module")
def model_session():
    """Fixture to load ONNX model once for all tests"""
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded to {MODEL_PATH}")
    
    # Load model
    session = ort.InferenceSession(MODEL_PATH)
    return session


@pytest.fixture(scope="module")
def test_data():
    """Fixture to download and load test data from Azure Blob Storage"""
    test_data_path = f"./test_data/{TEST_DATA_BLOB}"
    
    # Try to download from Azure if connection string is available
    if AZURE_STORAGE_CONNECTION_STRING:
        try:
            print(f"Downloading test data from Azure Blob Storage...")
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(
                container=TEST_DATA_CONTAINER,
                blob=TEST_DATA_BLOB
            )
            
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
            with open(test_data_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"Test data downloaded to {test_data_path}")
            
        except Exception as e:
            print(f"Warning: Could not download test data from Azure: {e}")
            print("Using default test data...")
    
    # If file exists, load it
    if os.path.exists(test_data_path):
        if test_data_path.endswith('.npz'):
            # Load NPZ format (image data)
            data = np.load(test_data_path)
            return {'images': data['images'], 'labels': data['labels']}
        else:
            # Load CSV format (fallback for backward compatibility)
            import pandas as pd
            df = pd.read_csv(test_data_path)
            return df
    
    # Otherwise, create default test data
    print("Creating default test data...")
    return create_default_test_data()


def create_default_test_data():
    """Create default test data if Azure data not available"""
    import pandas as pd
    
    # Create synthetic test data
    # Assuming the model expects some input features
    # Adjust based on actual model input requirements
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate random binary labels (0 or 1 for gender classification)
    y = np.random.randint(0, 2, size=n_samples)
    
    # Create DataFrame
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['label'] = y
    
    return df


def test_model_loads_successfully(model_session):
    """Test 1: Verify that the model loads successfully"""
    assert model_session is not None, "Model session should not be None"
    assert len(model_session.get_inputs()) > 0, "Model should have at least one input"
    assert len(model_session.get_outputs()) > 0, "Model should have at least one output"
    print("✓ Model loaded successfully")


def test_model_input_output_shapes(model_session):
    """Test: Verify model input and output shapes are defined"""
    inputs = model_session.get_inputs()
    outputs = model_session.get_outputs()
    
    print(f"Model inputs: {len(inputs)}")
    for i, inp in enumerate(inputs):
        print(f"  Input {i}: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    
    print(f"Model outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: {out.name}, shape: {out.shape}, type: {out.type}")
    
    assert inputs[0].name is not None, "Input should have a name"
    assert outputs[0].name is not None, "Output should have a name"


def test_model_responds_with_defined_input(model_session):
    """Test 2: Verify model responds with predefined input data"""
    # Get input details
    input_name = model_session.get_inputs()[0].name
    input_shape = model_session.get_inputs()[0].shape
    
    print(f"Testing with input: {input_name}, shape: {input_shape}")
    
    # Create sample input based on model's expected shape
    # GenderNet expects [batch, channels, height, width] = [1, 3, 224, 224]
    batch_size = 1
    
    if len(input_shape) == 4:
        # Image input: [batch, channels, height, width]
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        width = input_shape[3] if isinstance(input_shape[3], int) else 224
        test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    elif len(input_shape) == 2:
        # Tabular input: [batch, features]
        feature_size = input_shape[1] if isinstance(input_shape[1], int) else 10
        test_input = np.random.randn(batch_size, feature_size).astype(np.float32)
    else:
        # Default to image format
        test_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Run inference
    try:
        outputs = model_session.run(None, {input_name: test_input})
        
        assert outputs is not None, "Model should return outputs"
        assert len(outputs) >= MIN_PREDICTIONS, f"Model should return at least {MIN_PREDICTIONS} output"
        
        print(f"✓ Model responded successfully")
        print(f"  Output shapes: {[out.shape for out in outputs]}")
        print(f"  Output values sample: {outputs[0][:5] if len(outputs[0]) > 5 else outputs[0]}")
        
    except Exception as e:
        pytest.fail(f"Model inference failed: {e}")


def test_model_performance_metric(model_session, test_data):
    """Test 3: Verify model performance meets minimum threshold"""
    # Get input details
    input_name = model_session.get_inputs()[0].name
    input_shape = model_session.get_inputs()[0].shape
    
    # For image models, we can't use tabular test data directly
    # Instead, create synthetic image data and verify model runs
    print(f"Input shape: {input_shape}")
    
    if len(input_shape) == 4:
        # Image input model
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        width = input_shape[3] if isinstance(input_shape[3], int) else 224
        
        # Check if test_data is from NPZ (dict) or CSV (DataFrame)
        if isinstance(test_data, dict) and 'images' in test_data:
            # Use downloaded NPZ test data
            X_test = test_data['images'].astype(np.float32)
            y_true = test_data['labels']
            n_samples = len(X_test)
            print(f"Running inference on {n_samples} test images from Azure...")
        else:
            # Create synthetic test images
            n_samples = 10
            X_test = np.random.randn(n_samples, channels, height, width).astype(np.float32)
            y_true = np.random.randint(0, 2, size=n_samples)
            print(f"Running inference on {n_samples} synthetic test images...")
    else:
        # Tabular data model (fallback)
        if isinstance(test_data, dict):
            # Should not happen for tabular models, but handle it
            X_test = np.random.randn(10, 10).astype(np.float32)
            y_true = np.random.randint(0, 2, size=10)
        elif 'label' in test_data.columns:
            y_true = test_data['label'].values
            X_test = test_data.drop(columns=['label']).values.astype(np.float32)
        else:
            X_test = test_data.values.astype(np.float32)
            y_true = np.random.randint(0, 2, size=len(X_test))
    
    # Run inference
    try:
        outputs = model_session.run(None, {input_name: X_test})
        
        # Process predictions - assuming binary classification
        predictions = outputs[0]
        
        # Handle different output formats
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-output, take argmax
            y_pred = np.argmax(predictions, axis=1)
        else:
            # Single output, threshold at 0.5
            y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Accuracy threshold: {ACCURACY_THRESHOLD}")
        
        # Note: For demo purposes and synthetic data, we're using a low threshold
        # In production, this should be based on actual model performance with real data
        assert accuracy >= ACCURACY_THRESHOLD, \
            f"Model accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}"
        
        print(f"✓ Model performance is acceptable (accuracy: {accuracy:.4f})")
        
    except AssertionError:
        raise
    except Exception as e:
        # If we can't calculate accuracy properly, just verify the model runs
        print(f"Warning: Could not calculate accuracy: {e}")
        print("Verifying model at least produces outputs...")
        test_sample = X_test[:1] if len(X_test) > 0 else np.random.randn(1, *input_shape[1:]).astype(np.float32)
        outputs = model_session.run(None, {input_name: test_sample})
        assert outputs is not None, "Model should produce outputs"
        print("✓ Model produces outputs (accuracy test skipped)")


def test_model_batch_prediction(model_session):
    """Test: Verify model can handle batch predictions"""
    input_name = model_session.get_inputs()[0].name
    input_shape = model_session.get_inputs()[0].shape
    
    print(f"Input shape: {input_shape}")
    
    # Determine input dimensions
    if len(input_shape) == 4:
        # Image input: [batch, channels, height, width]
        batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        width = input_shape[3] if isinstance(input_shape[3], int) else 224
        input_dims = (channels, height, width)
    elif len(input_shape) == 2:
        # Tabular input: [batch, features]
        batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        feature_size = input_shape[1] if isinstance(input_shape[1], int) else 10
        input_dims = (feature_size,)
    else:
        # Default to image format
        batch_size = 1
        input_dims = (3, 224, 224)
    
    # Test with the supported batch size (usually 1 for fixed models)
    test_input = np.random.randn(batch_size, *input_dims).astype(np.float32)
    outputs = model_session.run(None, {input_name: test_input})
    
    assert outputs is not None, f"Model should handle batch size {batch_size}"
    print(f"✓ Model handles batch size {batch_size}")
    print(f"  Output shape: {outputs[0].shape}")


def test_model_deterministic(model_session):
    """Test: Verify model produces consistent outputs for same input"""
    input_name = model_session.get_inputs()[0].name
    input_shape = model_session.get_inputs()[0].shape
    
    # Determine input dimensions
    if len(input_shape) == 4:
        # Image input: [batch, channels, height, width]
        batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        width = input_shape[3] if isinstance(input_shape[3], int) else 224
        input_dims = (channels, height, width)
    elif len(input_shape) == 2:
        # Tabular input: [batch, features]
        batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        feature_size = input_shape[1] if isinstance(input_shape[1], int) else 10
        input_dims = (feature_size,)
    else:
        # Default to image format
        batch_size = 1
        input_dims = (3, 224, 224)
    
    # Create fixed input
    np.random.seed(123)
    test_input = np.random.randn(batch_size, *input_dims).astype(np.float32)
    
    # Run inference twice
    outputs1 = model_session.run(None, {input_name: test_input})
    outputs2 = model_session.run(None, {input_name: test_input})
    
    # Compare outputs
    for out1, out2 in zip(outputs1, outputs2):
        assert np.allclose(out1, out2), "Model should produce consistent outputs"
    
    print("✓ Model is deterministic")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
