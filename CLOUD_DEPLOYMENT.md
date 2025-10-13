# Cloud Deployment Guide

## Overview

This guide explains how to deploy the Fashion Product Classification API to Google Cloud Run using Docker.

---

## Prerequisites

- Google Cloud account
- Google Cloud SDK installed (`gcloud`)
- Docker installed
- Project cloned locally

---

## Step 1: Build Docker Image

```bash
# In project root
docker build -t fashion-classifier-api .
```

---

## Step 2: Authenticate Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

---

## Step 3: Push Docker Image to Google Container Registry

```bash
# Tag image
docker tag fashion-classifier-api gcr.io/YOUR_PROJECT_ID/fashion-classifier-api:latest

# Authenticate Docker with Google Cloud
gcloud auth configure-docker

# Push image
docker push gcr.io/YOUR_PROJECT_ID/fashion-classifier-api:latest
```

---

## Step 4: Deploy to Cloud Run

```bash
gcloud run deploy fashion-classifier-api \
  --image gcr.io/YOUR_PROJECT_ID/fashion-classifier-api:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

---

## Step 5: Test API

```bash
# Get service URL
gcloud run services describe fashion-classifier-api --platform managed --region europe-west1 --format 'value(status.url)'

# Test health endpoint
curl https://YOUR_CLOUD_RUN_URL/health

# Test prediction endpoint
curl -X POST -F "file=@test.jpg" https://YOUR_CLOUD_RUN_URL/predict
```

---

## Troubleshooting

- **Build errors:** Check Dockerfile and dependencies.
- **Authentication errors:** Ensure `gcloud` is logged in and project ID is set.
- **API not responding:** Check logs in Cloud Console.
- **Quota errors:** Check Cloud Run quotas and billing.
- **Timeouts:** Increase `--timeout` if necessary.

---

## Cleanup

```bash
gcloud run services delete fashion-classifier-api --region europe-west1
```

---

## References

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker Documentation](https://docs.docker.com/)
- [API README](api/README.md)