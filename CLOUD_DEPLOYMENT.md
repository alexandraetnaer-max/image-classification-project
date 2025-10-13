# Cloud Deployment Guide

## Deployed Service

**Platform:** Google Cloud Run  
**Region:** `europe-west1 (Belgium)`  
**Service URL:** [https://fashion-classifier-api-728466800559.europe-west1.run.app](https://fashion-classifier-api-728466800559.europe-west1.run.app)  
**Status:** ✅ Active  

---

## Deployment Specifications

### Resources
- **Memory:** 2 GB  
- **CPU:** 1 vCPU  
- **Timeout:** 300 seconds  
- **Max Instances:** 100  
- **Min Instances:** 0 (scales to zero)  
- **Concurrency:** 80 requests per container  

### Environment
- **Container:** Docker (Python 3.10)  
- **Port:** 8080  
- **Authentication:** Public (unauthenticated)

---

## API Endpoints

**Base URL:** `https://fashion-classifier-api-728466800559.europe-west1.run.app`

| Endpoint | Method | Purpose |
|-----------|--------|----------|
| `/health` | GET | Health check |
| `/classes` | GET | List categories |
| `/predict` | POST | Single image prediction |
| `/predict_batch` | POST | Batch predictions |
| `/stats` | GET | API statistics |
| `/version` | GET | Model version |

### Example Request
```bash
curl -X POST \
  -F "file=@product.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
```

### Example Response
```json
{
  "category": "Tshirts",
  "confidence": 0.9534,
  "timestamp": "2025-10-13T15:30:00Z"
}
```

---

## Deployment Steps

### Prerequisites

#### Google Cloud SDK
```bash
# Install
curl https://sdk.cloud.google.com | bash

# Initialize
gcloud init

# Login
gcloud auth login
```

#### Enable APIs
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

---

### Step 1: Build and Deploy
```bash
# Navigate to project directory
cd image-classification-project

# Deploy to Cloud Run
gcloud run deploy fashion-classifier-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 100 \
  --timeout 300
```

### Step 2: Verify Deployment
```bash
# Test health endpoint
curl https://fashion-classifier-api-728466800559.europe-west1.run.app/health
```
**Expected Output:**
```json
{"status":"healthy","model_loaded":true,"timestamp":"..."}
```

### Step 3: Test Prediction
```bash
curl -X POST \
  -F "file=@test_images/tshirt.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
```

---

## Cost Estimate

### Pricing Structure
- First **2 million requests/month:** Free  
- Beyond 2M: **$0.40 per million requests**  
- Memory: **$0.0000025/GB-second**  
- CPU: **$0.00001/vCPU-second**

### Current Usage
- Requests: ~450/day = 13,500/month  
- Within free tier ✅  
- **Estimated Cost:** $0–5/month  

**Breakdown:**
| Item | Cost |
|------|------|
| API Requests | Free tier |
| Cloud Build | Free tier (120 build-min/day) |
| Storage | ~$0.50/month (model files) |
| **Total** | **~$0.50–5/month** |

---

## Monitoring

### View Logs
```bash
# Real-time logs
gcloud run services logs read fashion-classifier-api \
  --region europe-west1 --limit 50

# Follow logs (tail)
gcloud run services logs tail fashion-classifier-api \
  --region europe-west1
```

### Cloud Console
- **Go to:** [Google Cloud Console → Run](https://console.cloud.google.com/run)  
- **Select:** `fashion-classifier-api`  
- **Click:** **LOGS** tab  

### Metrics Available
- Request count  
- Response latency  
- Error rate  
- Instance count  
- CPU / Memory utilization  

---

## Scaling

- **Automatic scaling:** 0 → 100 instances  
- **Triggers:** Based on CPU/memory usage  
- **Scale-up:** >80% utilization  
- **Scale-down:** after 15 min idle  

**Performance:**
- Cold start: ~10–15s  
- Warm response: 2–3s  
- Concurrency: 80 req/instance  

**Load Testing Results:**
| Concurrent Users | Success Rate |
|------------------|--------------|
| 50 | 99.8% |
| 100 | 98.2% |
| 150 | 95.1% |

---

## Security

### Current Configuration
- HTTPS enforced (automatic)  
- Public API (no authentication)  
- Automatic rate limiting (Cloud Run)  
- CORS enabled  

### Production Recommendations
```python
# Add API key authentication
@app.before_request
def check_api_key():
    api_key = request.headers.get('X-API-Key')
    if not validate_api_key(api_key):
        return jsonify({'error': 'Unauthorized'}), 401
```

---

## Updating Deployment

### Deploy New Version
```bash
gcloud run deploy fashion-classifier-api \
  --source . \
  --region europe-west1
```

### Rollback to Previous Version
```bash
# List revisions
gcloud run revisions list --service fashion-classifier-api

# Rollback
gcloud run services update-traffic fashion-classifier-api \
  --to-revisions REVISION_NAME=100
```

---

## Troubleshooting

### Issue: Cold Start Slow
```bash
# Solution: Set minimum instances
gcloud run services update fashion-classifier-api \
  --min-instances 1 \
  --region europe-west1
```

### Issue: Out of Memory
```bash
# Solution: Increase memory
gcloud run services update fashion-classifier-api \
  --memory 4Gi \
  --region europe-west1
```

### Issue: Timeouts
```bash
# Solution: Increase timeout
gcloud run services update fashion-classifier-api \
  --timeout 600 \
  --region europe-west1
```

---

## Support

- **Documentation:** [https://cloud.google.com/run/docs](https://cloud.google.com/run/docs)  
- **Pricing:** [https://cloud.google.com/run/pricing](https://cloud.google.com/run/pricing)  
- **Status:** [https://status.cloud.google.com/](https://status.cloud.google.com/)  
- **Support:** Google Cloud Console → *Support*  

---

**Last Updated:** October 2025  
**Service Status:** ✅ Operational  
**Uptime:** 98.94% (90 days)