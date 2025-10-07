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

app = Flask(__name__)

# Configuration
MODEL_DIR = os.path.join('..', 'models')
RESULTS_DIR = os.path.join('..', 'results')

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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
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
        
        # Get all probabilities
        all_probs = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
        
        # Sort by confidence
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'category': class_names[predicted_class_idx],
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
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
def get_classes():
    """Get list of available classes"""
    return jsonify({
        'classes': class_names,
        'total': len(class_names)
    })

if __name__ == '__main__':
    print("Loading model...")
    load_model_and_classes()
    print("Starting Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=True)