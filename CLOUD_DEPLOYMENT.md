Cloud Deployment Documentation
Deployed Service
Platform: Google Cloud Run
Region: europe-west1
Service URL: https://fashion-classifier-api-728466800559.europe-west1.run.app
Deployment Specifications

Memory: 2GB
Timeout: 300 seconds
Max Instances: 10
Authentication: Public
Container: Docker with Python 3.11

API Endpoints
Health Check: GET /health
Get Classes: GET /classes
Single Prediction: POST /predict
Batch Prediction: POST /predict_batch
Deployment Steps

Install Google Cloud SDK
Configure project: gcloud init
Enable APIs
Deploy with gcloud run deploy

Cost Estimate

Free Tier: 2 million requests per month
Current Usage: Within free tier
Estimated Cost: $0 per month

Monitoring
View logs with gcloud run services logs read
Scaling

Automatic scaling from 0 to 10 instances
Cold start time: approximately 10 seconds
Average response time: 2-3 seconds

Security

HTTPS only
Public API
Rate limiting by Cloud Run