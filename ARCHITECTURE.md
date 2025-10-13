# System Architecture

## Architecture Diagram

![System Architecture](../docs/architecture_diagram.png)

---

## Overview

The Fashion Product Classification System is a cloud-native machine learning application that automatically classifies fashion product images into 10 categories using deep learning and batch processing.

## Architecture Diagram

┌────────────────────────────────────────────────────────────────────────┐
│                         USER / CLIENT                                 │
└────────────────────────────┬───────────────────────────────────────────┘
│
│ HTTPS Requests
▼
┌────────────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD RUN                                   │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    FLASK REST API                           │      │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐        │      │
│  │  │   /health   │  │   /predict   │  │/predict_batch│        │      │
│  │  └─────────────┘  └──────────────┘  └──────────────┘        │      │
│
└───────────────────────────┬───────────────────────────────────────────┘
│                               │                                      │
│  ┌───────────────────────────▼──────────────────────────────────┐    │
│  │              ML MODEL (MobileNetV2)                         │    │
│  │              Trained on 25,465 images                       │    │
│  │              10 categories, 91.78% accuracy                 │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
│
│ Results
▼
┌────────────────────────────────────────────────────────────────────────┐
│                    LOCAL ENVIRONMENT                                  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │
│ BATCH PROCESSOR                              │   │
│  │  - Scans data/incoming/ folder                               │   │
│  │  - Sends images to Cloud API                                 │   │
│  │  - Organizes results by category                             │   │
│  │  - Generates logs and reports                                │   │
│  └────────────┬─────────────────────────────┬───────────────────┘      │
│               │                              │                       │
│               ▼                              ▼                       │
│  ┌─────────────────────┐      ┌──────────────────────────┐           │
│  │  DATA STORAGE       │      │  RESULTS & LOGS          │           │
│  │                     │      │                          │           │
│  │  data/incoming/     │      │  results/batch_results/  │           │
│  │  data/processed/    │      │  logs/                   │           │
│  └─────────────────────┘      └──────────────────────────┘           │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              TASK SCHEDULER (Windows)                        │      │
│  │              Runs batch_processor.py daily at 2:00 AM        │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                             │
│  (Kaggle Notebooks with GPU)                                          │
│                                                                       │
│  Dataset (Kaggle) → Data Preparation → Model Training →               │
│  Evaluation → Export Model → Deploy to Cloud                          │
└────────────────────────────────────────────────────────────────────────┘

## Components Description

### 1. Cloud API Layer (Google Cloud Run)

**Purpose:** Provide RESTful API for image classification

**Technology Stack:**
- Flask 3.1.0
- TensorFlow 2.20.0
- Gunicorn WSGI server
- Docker containerization

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /classes` - List categories
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch predictions

**Deployment:**
- Region: europe-west1
- Memory: 2GB
- Timeout: 300s
- Auto-scaling: 0-10 instances
- Public access (no authentication)

### 2. ML Model

**Architecture:** Transfer Learning with MobileNetV2

**Training Data:**
- Source: Kaggle Fashion Product Images
- Total images: 25,465
- Categories: 10
- Split: 70% train / 15% validation / 15% test

**Performance:**
- Training accuracy: 92.95%
- Validation accuracy: 91.78%
- Test accuracy: 90.31%

**Input:** 224x224 RGB images

**Output:** Category probabilities for 10 classes

### 3. Batch Processing System

**Purpose:** Automated nightly processing of new images

**Features:**
- Monitors data/incoming/ folder
- Sends images to Cloud API
- Organizes results by category
- Generates CSV reports and logs
- Handles multiple image formats (.jpg, .png, .webp)

**Automation:**
- Windows Task Scheduler
- Scheduled: Daily at 2:00 AM
- Script: run_batch_processing.bat

**Outputs:**
- Categorized images in data/processed_batches/
- CSV results in results/batch_results/
- JSON summaries
- Detailed logs in logs/

### 4. Data Storage

**Structure:**
data/
├── incoming/              # New images for processing
├── processed_batches/     # Organized by category
│   ├── Tshirts/
│   ├── Shoes/
│   └── ...
├── raw/                   # Original training data
└── processed/             # Train/val/test splits

### 5. Monitoring & Logging

**Health Monitoring:**
- /health endpoint checks model status
- Returns: model loaded, number of classes

**Logging:**
- Batch processing logs (timestamp, status, errors)
- API request/response logs (Cloud Run)
- Performance metrics (accuracy, confidence)

**Reporting:**
- CSV: filename, category, confidence, timestamp
- JSON: summary statistics, category distribution

## Data Flow

### Training Phase:
1. Download dataset from Kaggle
2. Prepare and split data (prepare_data.py)
3. Train model in Kaggle Notebooks with GPU
4. Evaluate and save model
5. Deploy to GitHub and Cloud Run

### Inference Phase (Real-time):
1. Client sends image to Cloud API
2. API preprocesses image (resize, normalize)
3. Model predicts category
4. API returns JSON response

### Inference Phase (Batch):
1. Images accumulate in data/incoming/
2. Task Scheduler triggers batch_processor.py
3. Script reads all images
4. Sends each to Cloud API
5. Organizes images by predicted category
6. Saves results to CSV and logs

## Scalability

**Current Setup:**
- Cloud Run auto-scales 0-10 instances
- Handles ~2M requests/month (free tier)
- Cold start: ~10 seconds
- Average response: 2-3 seconds

**Future Improvements:**
- Increase max instances for higher load
- Add caching layer (Redis)
- Implement request queuing
- Add load balancer

## Security

**Current:**
- HTTPS only
- Public API (no authentication)
- Rate limiting by Cloud Run
- No sensitive data stored

**Production Recommendations:**
- Add API key authentication
- Implement request throttling
- Add input validation
- Enable CORS properly
- Regular security updates

## Cost Analysis

**Google Cloud Run:**
- Free tier: 2M requests/month
- Current usage: ~1000 requests/month
- Estimated cost: $0/month

**Storage:**
- GitHub: Free (code only, no large files)
- Local: ~5GB for processed data

**Total Monthly Cost: $0** (within free tiers)

## Technologies Used

**Backend:**
- Python 3.11
- Flask 3.1.0
- TensorFlow 2.20.0
- Gunicorn 21.2.0

**ML/Data:**
- Keras (Transfer Learning)
- MobileNetV2 (pre-trained)
- NumPy, Pandas
- Pillow, OpenCV

**Infrastructure:**
- Docker
- Google Cloud Run
- GitHub
- Kaggle Notebooks (training)

**Automation:**
- Windows Task Scheduler
- Batch scripts

## Development Setup

See [README.md](README.md) for detailed setup instructions.

## Deployment

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for cloud deployment guide.