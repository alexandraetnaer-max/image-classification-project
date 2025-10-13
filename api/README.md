Отлично! Давайте улучшим документацию API.

Улучшение документации API
ШАГ 1: Обновление api/README.md
Откройте api/README.md и замените всё содержимое на расширенную версию:
markdown# Fashion Classification API Documentation

REST API for classifying fashion product images into 10 categories using deep learning.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Base URL](#base-url)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Data Formats](#data-formats)
6. [Error Handling](#error-handling)
7. [Rate Limits](#rate-limits)
8. [Examples](#examples)

---

## Quick Start

**Cloud API (Production):**
```bash
curl https://fashion-classifier-api-728466800559.europe-west1.run.app/health
Local Development:
bash# Start API
cd api
python app.py

# API runs on http://localhost:5000
curl http://localhost:5000/health

Base URL
Production (Cloud):
https://fashion-classifier-api-728466800559.europe-west1.run.app
Development (Local):
http://localhost:5000

Authentication
Currently, the API does not require authentication (public access).
Future versions may implement:

API key authentication
JWT tokens
Rate limiting per user


Endpoints
1. Home / Service Info
Get API information and available endpoints.
Endpoint: GET /
Response:
json{
  "service": "Fashion Classification API",
  "version": "1.0",
  "status": "running",
  "model": {
    "loaded": true,
    "classes": 10
  },
  "endpoints": {
    "health": "/health",
    "classes": "/classes",
    "predict": "/predict (POST)",
    "predict_batch": "/predict_batch (POST)"
  },
  "documentation": "https://github.com/alexandraetnaer-max/image-classification-project",
  "cloud_deployment": "Google Cloud Run - europe-west1"
}
cURL Example:
bashcurl -X GET https://fashion-classifier-api-728466800559.europe-west1.run.app/
Status Codes:

200 OK - Success


2. Health Check
Check API availability and model status.
Endpoint: GET /health
Response:
json{
  "status": "healthy",
  "model_loaded": true,
  "classes": 10,
  "timestamp": "2025-10-13T14:30:00.123456"
}
cURL Example:
bashcurl -X GET https://fashion-classifier-api-728466800559.europe-west1.run.app/health
Status Codes:

200 OK - API is healthy
503 Service Unavailable - API is down or model not loaded

Use Cases:

Monitoring/alerting systems
Load balancer health checks
Pre-flight checks before sending images


3. Get Classes
Retrieve list of available fashion categories.
Endpoint: GET /classes
Response:
json{
  "classes": [
    "Casual Shoes",
    "Handbags",
    "Heels",
    "Kurtas",
    "Shirts",
    "Sports Shoes",
    "Sunglasses",
    "Tops",
    "Tshirts",
    "Watches"
  ],
  "total": 10
}
cURL Example:
bashcurl -X GET https://fashion-classifier-api-728466800559.europe-west1.run.app/classes
Status Codes:

200 OK - Success


4. Single Image Prediction
Classify a single fashion product image.
Endpoint: POST /predict
Content Types Supported:

multipart/form-data (file upload)
application/json (base64 encoded)

Option A: File Upload (Recommended)
Request:
bashcurl -X POST \
  -F "file=@/path/to/image.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
Form Data:

file (required): Image file

Formats: .jpg, .jpeg, .png, .bmp, .gif, .webp
Max size: 10 MB (recommended < 5 MB)
Recommended resolution: 224x224 to 1024x1024 pixels



Option B: Base64 Encoded
Request:
bashcurl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_ENCODED_STRING"}' \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
JSON Body:
json{
  "image": "/9j/4AAQSkZJRgABAQEAYABgAAD..." 
}
Response (Success):
json{
  "category": "Tshirts",
  "confidence": 0.9876,
  "all_probabilities": {
    "Tshirts": 0.9876,
    "Tops": 0.0087,
    "Shirts": 0.0023,
    "Kurtas": 0.0008,
    "Casual Shoes": 0.0003,
    "Sports Shoes": 0.0001,
    "Handbags": 0.0001,
    "Heels": 0.0001,
    "Watches": 0.0000,
    "Sunglasses": 0.0000
  },
  "timestamp": "2025-10-13T14:35:22.123456"
}
Response Fields:

category (string): Predicted category name
confidence (float): Confidence score (0-1) for predicted category
all_probabilities (object): Confidence scores for all 10 categories (sorted by confidence)
timestamp (string): ISO 8601 timestamp

Status Codes:

200 OK - Successful prediction
400 Bad Request - No image provided or invalid format
500 Internal Server Error - Processing error

Error Response:
json{
  "error": "No image provided"
}

5. Batch Prediction
Classify multiple images in a single request.
Endpoint: POST /predict_batch
Request:
bashcurl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict_batch
Form Data:

files (required, multiple): Image files
Max files per request: 50 (recommended: 10-20)
Same format restrictions as single prediction

Response (Success):
json{
  "predictions": [
    {
      "filename": "image1.jpg",
      "category": "Tshirts",
      "confidence": 0.9876
    },
    {
      "filename": "image2.jpg",
      "category": "Shoes",
      "confidence": 0.9234
    },
    {
      "filename": "image3.jpg",
      "category": "Handbags",
      "confidence": 0.8765
    }
  ],
  "total": 3,
  "timestamp": "2025-10-13T14:40:00.123456"
}
Status Codes:

200 OK - All images processed (check individual results for errors)
400 Bad Request - No files provided
413 Payload Too Large - Too many files or files too large


6. Statistics
Get API usage statistics (requires API to be running).
Endpoint: GET /stats
Response:
json{
  "uptime_seconds": 3600,
  "uptime_human": "1:00:00",
  "total_requests": 150,
  "total_images_processed": 145,
  "successful_predictions": 142,
  "failed_predictions": 3,
  "success_rate": 97.93,
  "model_loaded": true,
  "timestamp": "2025-10-13T15:00:00.123456"
}
cURL Example:
bashcurl -X GET https://fashion-classifier-api-728466800559.europe-west1.run.app/stats
Status Codes:

200 OK - Success

Note: Statistics reset when API restarts.

Data Formats
Supported Image Formats

JPEG/JPG (.jpg, .jpeg) - Recommended
PNG (.png)
BMP (.bmp)
GIF (.gif)
WebP (.webp)
TIFF (.tiff, .tif)

Image Requirements
Size Limits:

Minimum: 50x50 pixels
Maximum: 4096x4096 pixels
Recommended: 224x224 to 1024x1024 pixels
File size: < 10 MB (recommended < 5 MB)

Color Space:

RGB (3 channels)
Grayscale images are converted to RGB

Processing:

Images automatically resized to 224x224 for model input
Aspect ratio may change during resize
Normalization applied (0-1 range)

Categories
The model classifies images into these 10 categories:

Casual Shoes - Sneakers, casual footwear
Handbags - Purses, shoulder bags, totes
Heels - High heels, formal shoes
Kurtas - Traditional Indian tops
Shirts - Formal shirts, button-ups
Sports Shoes - Athletic shoes, running shoes
Sunglasses - Eyewear, shades
Tops - Casual tops, blouses
Tshirts - T-shirts, casual shirts
Watches - Wristwatches, timepieces


Error Handling
Error Response Format
All errors return JSON with an error field:
json{
  "error": "Error description"
}
Common Errors
400 Bad Request
Cause: Invalid request format or missing data
Examples:
json{"error": "No image provided"}
{"error": "Invalid image format"}
{"error": "Image file is empty"}
Solution: Check request format, file upload, image format

404 Not Found
Cause: Invalid endpoint
Example:
json{"error": "404 Not Found: The requested URL was not found"}
Solution: Check endpoint URL spelling

413 Payload Too Large
Cause: File too large or too many files
Example:
json{"error": "File size exceeds limit"}
Solution:

Reduce file size (compress images)
Send fewer files in batch
Use smaller resolution


415 Unsupported Media Type
Cause: Wrong Content-Type header
Example:
json{"error": "Unsupported Media Type"}
Solution:

Use multipart/form-data for file uploads
Use application/json for base64


500 Internal Server Error
Cause: Server-side processing error
Examples:
json{"error": "Model inference failed"}
{"error": "Image preprocessing error"}
Solution:

Check image is valid and not corrupted
Retry request
Contact support if persists


503 Service Unavailable
Cause: API or model not ready
Example:
json{"error": "Model not loaded"}
Solution: Wait for API to initialize (cold start ~10 seconds)

Rate Limits
Cloud API (Google Cloud Run)
Current Limits:

No authentication required
Google Cloud Run auto-scaling handles load
Soft limit: 100 requests/minute per IP
Hard limit: 2 million requests/month (free tier)

Exceeding Limits:

Requests may be throttled
HTTP 429 (Too Many Requests) returned
Wait 60 seconds and retry

Best Practices

Batch Processing: Use /predict_batch for multiple images
Caching: Cache predictions for identical images
Async Processing: Use batch processor for large volumes
Retries: Implement exponential backoff for retries


Examples
Complete Examples
Example 1: Python with requests
pythonimport requests

# Health check
response = requests.get('https://fashion-classifier-api-728466800559.europe-west1.run.app/health')
print(response.json())

# Single prediction
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'https://fashion-classifier-api-728466800559.europe-west1.run.app/predict',
        files=files
    )
    print(response.json())
Example 2: JavaScript/Node.js
javascriptconst FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('image.jpg'));

axios.post('https://fashion-classifier-api-728466800559.europe-west1.run.app/predict', form, {
  headers: form.getHeaders()
})
.then(response => console.log(response.data))
.catch(error => console.error(error));
Example 3: Base64 Encoding
pythonimport base64
import requests

# Read and encode image
with open('image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Send to API
response = requests.post(
    'https://fashion-classifier-api-728466800559.europe-west1.run.app/predict',
    json={'image': image_base64}
)
print(response.json())
Example 4: Batch Processing
bash# Upload multiple images
curl -X POST \
  -F "files=@tshirt1.jpg" \
  -F "files=@tshirt2.jpg" \
  -F "files=@shoe.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict_batch
Example 5: Error Handling
pythonimport requests

try:
    response = requests.post(
        'https://fashion-classifier-api-728466800559.europe-west1.run.app/predict',
        files={'file': open('image.jpg', 'rb')},
        timeout=30
    )
    response.raise_for_status()  # Raise exception for 4xx/5xx
    
    result = response.json()
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.json()}")
except Exception as e:
    print(f"Error: {e}")

Testing
Quick Test
bash# Download a test image
curl -o test.jpg https://via.placeholder.com/224

# Test prediction
curl -X POST \
  -F "file=@test.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
Load Testing
bash# Using Apache Bench
ab -n 100 -c 10 https://fashion-classifier-api-728466800559.europe-west1.run.app/health

# Using siege
siege -c 10 -t 30S https://fashion-classifier-api-728466800559.europe-west1.run.app/health

Performance
Response Times

Health check: < 100ms
Single prediction: 1-3 seconds
Batch prediction (10 images): 10-20 seconds
Cold start (first request): ~10 seconds

Optimization Tips

Image Size: Smaller images process faster
Batch Size: Optimal batch size is 5-10 images
Concurrent Requests: Max 5 concurrent for best performance
Caching: Cache results for frequently queried images


Monitoring
Health Monitoring
bash# Check every 5 minutes
*/5 * * * * curl -f https://fashion-classifier-api-728466800559.europe-west1.run.app/health || echo "API DOWN"
Logging
All requests are logged with:

Timestamp
Method and path
Remote IP
Response time
Status code

Logs available at: logs/api.log

Support

GitHub: https://github.com/alexandraetnaer-max/image-classification-project
Issues: https://github.com/alexandraetnaer-max/image-classification-project/issues
Documentation: See README.md and other docs


Changelog
Version 1.0 (Current)

Initial release
10 fashion categories
Single and batch prediction
Cloud deployment
Statistics endpoint
