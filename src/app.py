"""
FastAPI application for Gender Classifier Model
Serves predictions and logs them to Azure Blob Storage
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import io
import base64
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import requests
from dotenv import load_dotenv
from PIL import Image
import cv2

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gender Classifier MLOps",
    description="ONNX model deployment with CI/CD",
    version="1.0.0"
)

# Setup templates directory
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Configuration
MODEL_URL = os.getenv("MODEL_URL", "https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/gender_googlenet.onnx")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
PREDICTIONS_CONTAINER = os.getenv("PREDICTIONS_CONTAINER", "predictions")

# Global variable for model session
model_session = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    input_data: List[List[float]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_data": [[1.0, 2.0, 3.0, 4.0]]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[Any]
    environment: str
    timestamp: str


class GenderPredictionResponse(BaseModel):
    """Response model for gender predictions from images"""
    gender: str
    confidence: float
    probabilities: Dict[str, float]
    environment: str
    timestamp: str


def download_model():
    """Download ONNX model from Azure Blob Storage"""
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        if os.path.exists(MODEL_PATH):
            logger.info(f"Model already exists at {MODEL_PATH}")
            return
        
        logger.info(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


def load_model():
    """Load ONNX model into runtime session"""
    global model_session
    try:
        if not os.path.exists(MODEL_PATH):
            download_model()
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model_session = ort.InferenceSession(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Log model metadata
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        logger.info(f"Model input name: {input_name}")
        logger.info(f"Model output name: {output_name}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for gender classification model
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed numpy array in shape (1, 3, 224, 224)
    """
    # Resize to 224x224
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Mean and std normalization (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def log_prediction_to_azure(prediction_data: Dict[str, Any]):
    """Log prediction to Azure Blob Storage"""
    if not AZURE_STORAGE_CONNECTION_STRING:
        logger.warning("Azure Storage connection string not configured, skipping prediction logging")
        return
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Determine filename based on environment
        filename = f"predicciones_{ENVIRONMENT}.txt"
        blob_client = blob_service_client.get_blob_client(
            container=PREDICTIONS_CONTAINER,
            blob=filename
        )
        
        # Format prediction log entry
        timestamp = prediction_data.get("timestamp", datetime.utcnow().isoformat())
        prediction = prediction_data.get("prediction", "")
        confidence = prediction_data.get("confidence", 0.0)
        input_type = prediction_data.get("input_type", "raw")
        
        log_entry = f"{timestamp} | ENV: {ENVIRONMENT} | TYPE: {input_type} | PREDICTION: {prediction} | CONFIDENCE: {confidence:.4f}\n"
        
        # Try to append to existing blob, or create new one
        try:
            # Download existing content
            existing_content = blob_client.download_blob().readall().decode('utf-8')
            new_content = existing_content + log_entry
        except ResourceNotFoundError:
            # Blob doesn't exist, create new
            new_content = log_entry
        
        # Upload updated content
        blob_client.upload_blob(new_content, overwrite=True)
        logger.info(f"Prediction logged to {filename}")
        
    except Exception as e:
        logger.error(f"Error logging prediction to Azure: {e}")
        # Don't raise - we don't want logging failures to break predictions


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info(f"Starting application in {ENVIRONMENT} environment")
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Web UI for Gender Classification"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "environment": ENVIRONMENT,
        "title": "Gender Classifier"
    })


@app.get("/api")
async def api_info():
    """API info endpoint"""
    return {
        "message": "Gender Classifier MLOps API",
        "environment": ENVIRONMENT,
        "status": "running",
        "endpoints": {
            "web_ui": "/",
            "api_health": "/health",
            "api_predict_image": "/predict/image",
            "api_predict_raw": "/predict",
            "api_model_info": "/model/info"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "model_loaded": model_session is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using the ONNX model with raw tensor data
    
    Args:
        request: PredictionRequest with input_data as nested list
                 Expected shape: [batch_size, channels, height, width]
                 For gender model: [1, 3, 224, 224]
    
    Returns:
        PredictionResponse with predictions, environment, and timestamp
    
    Note: This endpoint is for advanced use with preprocessed data.
          For image uploads, use /predict/image instead.
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get input name and expected shape from model
        input_name = model_session.get_inputs()[0].name
        expected_shape = model_session.get_inputs()[0].shape
        
        # Convert input data to numpy array
        input_array = np.array(request.input_data, dtype=np.float32)
        
        # Validate input shape
        logger.info(f"Received input shape: {input_array.shape}, Expected: {expected_shape}")
        
        # For 2D input, try to reshape to expected shape if possible
        if len(input_array.shape) == 2 and len(expected_shape) == 4:
            # Try to infer batch size
            batch_size = input_array.shape[0]
            # Expected format for gender model: [batch, 3, 224, 224]
            expected_size = 3 * 224 * 224  # 150528
            
            if input_array.shape[1] == expected_size:
                # Reshape flat array to [batch, channels, height, width]
                input_array = input_array.reshape(batch_size, 3, 224, 224)
                logger.info(f"Reshaped input to: {input_array.shape}")
            else:
                raise ValueError(
                    f"Input size {input_array.shape[1]} doesn't match expected size {expected_size}. "
                    f"For gender classification, expected shape is [batch, 3, 224, 224] or [batch, 150528]. "
                    f"Use /predict/image endpoint for easier image uploads."
                )
        
        # Run inference
        outputs = model_session.run(None, {input_name: input_array})
        
        # Convert outputs to list
        predictions = [output.tolist() for output in outputs]
        
        # Create response
        timestamp = datetime.utcnow().isoformat()
        response = PredictionResponse(
            predictions=predictions,
            environment=ENVIRONMENT,
            timestamp=timestamp
        )
        
        # Log prediction to Azure Blob Storage (legacy format)
        log_prediction_to_azure({
            "timestamp": timestamp,
            "prediction": str(predictions),
            "confidence": 0.0,
            "input_type": "raw"
        })
        
        return response
        
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/image", response_model=GenderPredictionResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """
    Predict gender from an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        GenderPredictionResponse with gender, confidence, and probabilities
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        input_array = preprocess_image(image)
        
        # Get input name from model
        input_name = model_session.get_inputs()[0].name
        
        # Run inference
        outputs = model_session.run(None, {input_name: input_array})
        
        # Get probabilities (assume output is [female_prob, male_prob])
        probs = outputs[0][0]  # Get first batch item
        
        # Apply softmax if needed
        probs_exp = np.exp(probs - np.max(probs))
        probs_softmax = probs_exp / probs_exp.sum()
        
        female_prob = float(probs_softmax[0])
        male_prob = float(probs_softmax[1])
        
        # Determine gender and confidence
        if female_prob > male_prob:
            gender = "female"
            confidence = female_prob
        else:
            gender = "male"
            confidence = male_prob
        
        # Create response
        timestamp = datetime.utcnow().isoformat()
        response = GenderPredictionResponse(
            gender=gender,
            confidence=confidence,
            probabilities={
                "female": female_prob,
                "male": male_prob
            },
            environment=ENVIRONMENT,
            timestamp=timestamp
        )
        
        # Log prediction to Azure Blob Storage
        log_prediction_to_azure({
            "timestamp": timestamp,
            "prediction": gender,
            "confidence": confidence,
            "input_type": "image"
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model metadata information"""
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs_info = [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type
            }
            for inp in model_session.get_inputs()
        ]
        
        outputs_info = [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type
            }
            for out in model_session.get_outputs()
        ]
        
        return {
            "inputs": inputs_info,
            "outputs": outputs_info,
            "environment": ENVIRONMENT
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
