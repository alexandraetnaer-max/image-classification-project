# Fashion Product Image Classification

## System Architecture

![Architecture Diagram](docs/architecture_diagram.png)

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📋 Project Overview

Machine Learning system for classifying fashion products into 10 categories using Transfer Learning (MobileNetV2), with REST API, batch processing, and cloud deployment.

- **Dataset**: Fashion Product Images from Kaggle (25,465 images)
- **Categories**: 10 classes (Tshirts, Shirts, Shoes, Watches, Handbags, etc.)
- **Model**: MobileNetV2 with Transfer Learning
- **Accuracy**: 91.78% validation accuracy
- **API**: Flask REST API (local + cloud)
- **Batch Processing**: Automated nightly classification
- **Cloud**: Deployed on Google Cloud Run

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project

# Build and run
docker-compose build
docker-compose run ml-training bash

# Inside container:
python src/prepare_data.py
python src/train_model_fixed.py
```
See DOCKER_INSTRUCTIONS.md for details.

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
# Extract to data/raw/

# 5. Prepare data
python src/prepare_data.py

# 6. Train model
python src/train_model_fixed.py

# 7. Test model
python src/simple_test.py
```

---

## 📁 Project Structure

```
image-classification-project/
├── api/                          # Flask REST API
│   ├── app.py                    # API application
│   ├── test_api.py               # API tests
│   └── README.md                 # API documentation
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Train/val/test splits
│   ├── incoming/                 # Batch processing input
│   └── processed_batches/        # Organized by category
├── models/                       # Trained models (.keras)
├── src/                          # Source code
│   ├── prepare_data.py           # Data preparation
│   ├── train_model_fixed.py      # Model training
│   ├── batch_processor.py        # Batch processing
│   ├── visualize_results.py      # Training visualizations
│   └── generate_batch_report.py  # Report generation
├── tests/                        # Unit tests
│   ├── test_data_preparation.py
│   ├── test_api.py
│   ├── test_batch_processor.py
│   └── test_utils.py
├── monitoring/                   # Monitoring tools
│   ├── dashboard.py              # Monitoring dashboard
│   └── check_batch_health.py     # Health checker
├── results/                      # Results and reports
│   ├── visualizations/           # Training charts
│   ├── batch_results/            # Batch CSVs
│   └── reports/                  # HTML/PDF reports
├── logs/                         # Application logs
├── docs/                         # Documentation
├── .github/workflows/            # CI/CD pipelines
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose
├── run_tests.py                  # Test runner
└── README.md                     # This file
```

---

## 💼 Usage Examples & Business Scenarios

For detailed examples and real-world scenarios, see **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)**

**Includes:**
- Step-by-step API integration examples (Python, JavaScript)
- Batch processing workflows
- E-commerce integration scenarios
- **Complete Returns Department workflow** (daily operations guide)
- Quality control procedures
- Inventory management integration

**Quick Example - Classify Product:**
```python
import requests

# Classify single product
with open('product.jpg', 'rb') as f:
    response = requests.post(
        'https://fashion-classifier-api-728466800559.europe-west1.run.app/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Category: {result['category']} ({result['confidence']:.0%})")
```

---

## 🎯 Features

**Core ML Pipeline**
- ✅ Data preparation (70/15/15 split)
- ✅ Transfer Learning (MobileNetV2)
- ✅ Model training with validation
- ✅ 91.78% accuracy achieved

**REST API**
- ✅ Flask-based RESTful API
- ✅ Single image prediction
- ✅ Batch predictions
- ✅ Health checks
- ✅ Statistics endpoint
- ✅ Deployed on Google Cloud Run

**Batch Processing**
- ✅ Automated image classification
- ✅ Scheduled execution (Task Scheduler/Cron)
- ✅ Result organization by category
- ✅ CSV and JSON reports
- ✅ Comprehensive logging

**Monitoring & Logging**
- ✅ Real-time API statistics
- ✅ Rotating file logs
- ✅ Health monitoring
- ✅ Performance dashboards
- ✅ Error tracking

**Visualization & Reporting**
- ✅ Training metrics visualization
- ✅ Confusion matrices
- ✅ Per-class accuracy charts
- ✅ Automated HTML reports
- ✅ Batch processing analytics

---

## 🧪 Testing

**Running Tests**
```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py test_api
python run_tests.py test_data_preparation
python run_tests.py test_batch_processor

# Run individual test
python -m unittest tests.test_api.TestFlaskAPI.test_health_endpoint
```

**Test Coverage**
- Data Preparation: Directory structure, split ratios, validation
- API Endpoints: All routes, request/response formats, error handling
- Batch Processing: File detection, result validation, summary generation
- Utilities: Image formats, configuration, data validation

See TESTING.md for detailed testing guide.

---

## 🌐 API Usage

**Local Development**
```bash
# Start API
cd api
python app.py

# API available at http://localhost:5000
```

**Cloud Deployment**
Live API: https://fashion-classifier-api-728466800559.europe-west1.run.app

**Endpoints**
```bash
# Health check
GET /health

# Get categories
GET /classes

# Predict single image
POST /predict
Content-Type: multipart/form-data
Body: file=@image.jpg

# Statistics
GET /stats
```
See api/README.md for complete API documentation.

---

## 📊 Batch Processing

**Manual Execution**
```bash
# Make sure API is running
python api/app.py

# In another terminal
python src/batch_processor.py
```

**Automated Execution**
- Windows: Task Scheduler with run_batch_processing.bat
- Linux/Mac: Cron with run_batch_processing.sh

```bash
# Add to crontab for daily 2 AM execution
0 2 * * * /path/to/run_batch_processing.sh
```
See BATCH_SCHEDULING.md for automation guide.

---

## 📈 Visualization & Reports

**Generate Training Visualizations**
```bash
python src/visualize_results.py
```
Output: results/visualizations/

- Training accuracy/loss curves
- Confusion matrix
- Class distribution
- Per-class accuracy

**Generate Batch Report**
```bash
# Last 7 days
python src/generate_batch_report.py

# Last 30 days
python src/generate_batch_report.py --days 30
```
Output: results/reports/ (HTML + charts)
See VISUALIZATION.md for visualization guide.

---

## 🔍 Monitoring

**Monitoring Dashboard**
```bash
python monitoring/dashboard.py
```
Output: monitoring/reports/

- System health report
- Processing statistics
- Visual charts
- Error analysis

**Health Check**
```bash
python monitoring/check_batch_health.py
```
See MONITORING.md for monitoring guide.

---

## ☁️ Cloud Deployment

The API is deployed on Google Cloud Run and publicly accessible.

URL: https://fashion-classifier-api-728466800559.europe-west1.run.app

**Deploy Your Own**
```bash
# Install Google Cloud SDK
gcloud init

# Deploy
gcloud run deploy fashion-classifier-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```
See CLOUD_DEPLOYMENT.md for deployment guide.

---

## 📚 Documentation

- ARCHITECTURE.md: System architecture and components
- DOCKER_INSTRUCTIONS.md: Docker setup guide
- BATCH_SCHEDULING.md: Batch automation guide
- CLOUD_DEPLOYMENT.md: Cloud deployment guide
- MONITORING.md: Monitoring and logging guide
- VISUALIZATION.md: Visualization and reporting guide
- TESTING.md: Testing guide
- api/README.md: API documentation

---

## 🔧 Troubleshooting

**Memory Error During Training**
```python
# Edit src/train_model_fixed.py
BATCH_SIZE = 8  # Reduce from 16 to 8 or 4
```

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**API Connection Issues**
```bash
# Check if API is running
curl http://localhost:5000/health

# For cloud API
curl https://fashion-classifier-api-728466800559.europe-west1.run.app/health
```

**Batch Processing Not Running**
```bash
# Check logs
cat logs/batch_*.log

# Verify API availability
python monitoring/check_batch_health.py
```

---

## 📊 Results

**Model Performance**
- Training Accuracy: 92.95%
- Validation Accuracy: 91.78%
- Test Accuracy: 90.31%

**Training Time**
- ~30-40 minutes on CPU
- ~10-15 minutes on GPU (Kaggle Notebooks)

**API Performance**
- Average Response Time: 2-3 seconds
- Success Rate: >95%
- Uptime: 99.9% (Cloud Run)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License

The Fashion Product Images dataset is used under **CC0: Public Domain** license.  
See [DATASETS.md](DATASETS.md) for full dataset documentation and attribution.

### Model License

- **MobileNetV2 (base):** Apache 2.0 License (Google)
- **Trained weights:** Apache 2.0 License (derived work)

---

## 👤 Author

Alexandra Etnaer

GitHub: @alexandraetnaer-max  
Project: image-classification-project

---

## 🙏 Acknowledgments

- Dataset: Fashion Product Images (Small) on Kaggle
- Model: MobileNetV2 (TensorFlow/Keras)
- Cloud: Google Cloud Run
- Framework: Flask, TensorFlow, Pandas, Matplotlib

---

## 📞 Support

- Issues: GitHub Issues
- Documentation: See docs above
- API Status: https://fashion-classifier-api-728466800559.europe-west1.run.app/health