"""
Flask REST API for fashion product classification
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import keras
import numpy as np
from PIL import Image
import io
import json
import base64
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import traceback
from functools import wraps
import time

app = Flask(__name__)
# Configure logging
def setup_logging():
    """Configure application logging"""
    log_dir = os.path.join('..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    
    # File handler for all logs
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'api.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'api_errors.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure app logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(error_handler)
    app.logger.setLevel(logging.INFO)
    
    app.logger.info("=" * 60)
    app.logger.info("Fashion Classification API Started")
    app.logger.info("=" * 60)

# Initialize logging
setup_logging()

# Request logging decorator
def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Log request
        app.logger.info(f"REQUEST: {request.method} {request.path} from {request.remote_addr}")
        
        try:
            response = f(*args, **kwargs)
            
            # Log response time
            duration = time.time() - start_time
            app.logger.info(f"RESPONSE: {request.path} completed in {duration:.2f}s")
            
            return response
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            app.logger.error(f"ERROR in {request.path}: {str(e)}")
            app.logger.error(traceback.format_exc())
            raise
    
    return decorated_function

# Statistics tracker
request_stats = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'total_images_processed': 0,
    'start_time': datetime.now()
}

def update_stats(success=True, num_images=1):
    """Update request statistics"""
    request_stats['total_requests'] += 1
    request_stats['total_images_processed'] += num_images
    if success:
        request_stats['successful_predictions'] += num_images
    else:
        request_stats['failed_predictions'] += num_images

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Global variables
model = None
class_names = None

def load_model_and_classes():
    """Load trained model and class names"""
    global model, class_names
    
    # Find latest model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError("No model found in models/ directory")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load class names
    classes_path = os.path.join(MODEL_DIR, 'class_names.json')
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"Model loaded with {len(class_names)} classes")

def preprocess_image(image_bytes):
    """Preprocess image for model"""
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
@app.route('/', methods=['GET'])
@log_request 
def home():
    """Welcome page with API information"""
    return jsonify({
        'service': 'Fashion Classification API',
        'version': '1.0',
        'status': 'running',
        'model': {
            'loaded': model is not None,
            'classes': len(class_names) if class_names else 0
        },
        'endpoints': {
            'health': '/health',
            'classes': '/classes',
            'predict': '/predict (POST)',
            'predict_batch': '/predict_batch (POST)'
        },
        'documentation': 'https://github.com/alexandraetnaer-max/image-classification-project',
        'cloud_deployment': 'Google Cloud Run - europe-west1'
    })
@app.route('/health', methods=['GET'])
@log_request 
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
@log_request 
def predict():
    """
    Predict category for a single image
    
    Request:
        - File upload (multipart/form-data)
        - JSON with base64 encoded image
    
    Response:
        {
            "category": "Tshirts",
            "confidence": 0.95,
            "all_probabilities": {...}
        }
    """
    try:
        # Get image from request
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            image_bytes = file.read()
        elif request.json and 'image' in request.json:
            # Base64 encoded
            image_data = request.json['image']
            image_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess
        img_array = preprocess_image(image_bytes)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get all probabilities
        all_probs = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
        
        # Sort by confidence
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        result = {
            'category': predicted_class,
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        update_stats(success=True, num_images=1)
        app.logger.info(f"Predicted: {predicted_class} with confidence {confidence:.4f}")
        
        return jsonify(result)
    
    except Exception as e:
        # Update statistics for failure
        update_stats(success=False, num_images=1)
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
@log_request 
def predict_batch():
    """
    Predict categories for multiple images
    
    Request:
        - Multiple file uploads
        - JSON array with base64 encoded images
    
    Response:
        {
            "results": [
                {"filename": "img1.jpg", "category": "Tshirts", "confidence": 0.95},
                ...
            ],
            "total_processed": 5
        }
    """
    try:
        results = []
        
        # Handle file uploads
        if request.files:
            files = request.files.getlist('files')
            
            for file in files:
                image_bytes = file.read()
                img_array = preprocess_image(image_bytes)
                
                predictions = model.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                results.append({
                    'filename': file.filename,
                    'category': class_names[predicted_class_idx],
                    'confidence': confidence
                })
        
        # Handle JSON array
        elif request.json and 'images' in request.json:
            images = request.json['images']
            
            for idx, image_data in enumerate(images):
                image_bytes = base64.b64decode(image_data)
                img_array = preprocess_image(image_bytes)
                
                predictions = model.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                results.append({
                    'filename': f'image_{idx}.jpg',
                    'category': class_names[predicted_class_idx],
                    'confidence': confidence
                })
        
        else:
            return jsonify({'error': 'No images provided'}), 400
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
@log_request 
def get_classes():
    """Get list of available classes"""
    return jsonify({
        'classes': class_names,
        'total': len(class_names)
    })
@app.route('/stats', methods=['GET'])
@log_request
def get_stats():
    """Get API statistics"""
    uptime = datetime.now() - request_stats['start_time']
    
    stats = {
        'uptime_seconds': int(uptime.total_seconds()),
        'uptime_human': str(uptime).split('.')[0],
        'total_requests': request_stats['total_requests'],
        'total_images_processed': request_stats['total_images_processed'],
        'successful_predictions': request_stats['successful_predictions'],
        'failed_predictions': request_stats['failed_predictions'],
        'success_rate': (
            request_stats['successful_predictions'] / request_stats['total_images_processed'] * 100
            if request_stats['total_images_processed'] > 0 else 0
        ),
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    app.logger.info(f"Stats requested: {stats}")
    return jsonify(stats)

# Load model on startup (important for gunicorn)
load_model_and_classes()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)