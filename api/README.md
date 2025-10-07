Fashion Classification API
REST API for classifying fashion product images into 10 categories.
Available Categories

Casual Shoes
Handbags
Heels
Kurtas
Shirts
Sports Shoes
Sunglasses
Tops
Tshirts
Watches

Endpoints
1. Health Check
GET /health
Response:
json{
  "status": "healthy",
  "model_loaded": true,
  "classes": 10
}
2. Get Classes
GET /classes
Response:
json{
  "classes": ["Tshirts", "Shirts"],
  "total": 10
}
3. Single Image Prediction
POST /predict
Example:
bashcurl -X POST -F "file=@image.jpg" http://localhost:5000/predict
Response:
json{
  "category": "Tshirts",
  "confidence": 0.95
}
4. Batch Prediction
POST /predict_batch
Example:
bashcurl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" http://localhost:5000/predict_batch
Running the API
bashcd api
python app.py
Server runs on http://localhost:5000
Testing
bashpython api/test_api.py
