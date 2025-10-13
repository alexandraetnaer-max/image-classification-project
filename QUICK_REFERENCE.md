```markdown
# Quick Reference Guide

Fast reference for common tasks and commands.

---

## API Endpoints

### Production URL
https://fashion-classifier-api-728466800559.europe-west1.run.app

### Quick Commands

**Health Check:**
```bash
curl https://fashion-classifier-api-728466800559.europe-west1.run.app/health
Get Categories:
bashcurl https://fashion-classifier-api-728466800559.europe-west1.run.app/classes
Classify Image:
bashcurl -X POST -F "file=@image.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict

Local Development
Start API Server
bashcd api
python app.py
# API: http://localhost:5000
Run Batch Processing
bashpython src/batch_processor.py
Run Tests
bashpython run_tests.py
Generate Reports
bash# Training visualizations
python src/visualize_results.py

# Batch report
python src/generate_batch_report.py

# Monitoring dashboard
python monitoring/dashboard.py

Docker Commands
Build & Run
bashdocker-compose build
docker-compose up
Run Training
bashdocker-compose run ml-training bash
python src/prepare_data.py
python src/train_model_fixed.py

File Locations
Models: models/*.keras
Data: data/incoming/ (batch input)
Results: results/batch_results/ (CSV outputs)
Logs: logs/*.log
Reports: results/reports/ (HTML)

Common Workflows
Daily Batch Processing

Place images → data/incoming/
Wait for scheduled run (2 AM) OR run manually
Check results → results/batch_results/
Images organized → data/processed_batches/

New Model Training

Prepare data → python src/prepare_data.py
Train model → python src/train_model_fixed.py
Test → python src/simple_test.py
Visualize → python src/visualize_results.py

Deploy to Cloud
bashgcloud run deploy fashion-classifier-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi

Monitoring
Check API Stats:
bashcurl https://fashion-classifier-api-728466800559.europe-west1.run.app/stats
View Logs:
bash# API logs
tail -f logs/api.log

# Batch logs
tail -f logs/batch_*.log
Health Dashboard:
bashpython monitoring/check_batch_health.py

Troubleshooting
IssueSolutionAPI timeoutIncrease timeout: requests.post(..., timeout=60)Model not loadedCheck models/ folder, restart APIBatch failsCheck logs: logs/batch_*.logLow confidenceRetake photo with better lightingMemory errorReduce batch size in code

Support

Documentation: See all .md files in root
Examples: USAGE_EXAMPLES.md
Testing: TESTING.md
API Docs: api/README.md
Issues: GitHub Issues


Categories Reference

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


Performance Benchmarks

API Response: 2-3 seconds
Batch (100 images): ~5 minutes
Model Accuracy: 91.78%
Success Rate: >95%


Key Metrics

60 Tests - 100% pass rate
25,465 Images - Training dataset
10 Categories - Fashion products
91.78% Accuracy - Validation
6 Endpoints - REST API