```markdown
# Docker Instructions

Complete guide for running the Fashion Classification System using Docker.

---

## Prerequisites

**1. Install Docker Desktop:**
- Windows/Mac: https://www.docker.com/products/docker-desktop/
- Linux: https://docs.docker.com/engine/install/

**2. Verify Installation:**
```bash
docker --version
# Should show: Docker version 24.0.0 or higher

docker-compose --version
# Should show: Docker Compose version 2.20.0 or higher

Quick Start
Option 1: Docker Compose (Recommended)
Start All Services:
bash# Clone repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project

# Start services
docker-compose up

# Access API at: http://localhost:5000
Stop Services:
bashdocker-compose down
Option 2: Docker Only
Build Image:
bashdocker build -t fashion-classifier .
Run Container:
bashdocker run -p 5000:5000 fashion-classifier

# Access API at: http://localhost:5000

Docker Compose Configuration
File: docker-compose.yml
yamlversion: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
Services:

api - Flask API service


Dockerfile
Location: Root directory
dockerfileFROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "api/app.py"]

Running Training with Docker
Start Training Container:
bashdocker-compose run ml-training bash

# Inside container:
python src/prepare_data.py
python src/train_model_fixed.py

Volume Mapping
Persistent Data:
yamlvolumes:
  - ./models:/app/models      # Model files
  - ./data:/app/data          # Training data
  - ./logs:/app/logs          # Log files
  - ./results:/app/results    # Results
Benefits:

Data persists after container stops
Easy access from host machine
Share data between containers


Environment Variables
Set in docker-compose.yml:
yamlenvironment:
  - PORT=5000
  - MODEL_PATH=/app/models/fashion_classifier.keras
  - LOG_LEVEL=INFO
Or in .env file:
bashPORT=5000
MODEL_PATH=/app/models/fashion_classifier.keras
LOG_LEVEL=INFO

Useful Commands
View Logs:
bash# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
Enter Container:
bash# Interactive shell
docker-compose exec api bash

# Run command
docker-compose exec api python src/simple_test.py
Restart Services:
bashdocker-compose restart
Rebuild After Code Changes:
bashdocker-compose up --build
Clean Up:
bash# Stop and remove containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Remove volumes (WARNING: deletes data)
docker-compose down -v

Testing with Docker
Run Tests:
bashdocker-compose exec api python run_tests.py
Run Specific Test:
bashdocker-compose exec api python -m pytest tests/test_api.py -v

Production Deployment
Build Production Image:
bashdocker build -t fashion-classifier:prod -f Dockerfile.prod .
Run with Gunicorn:
bashdocker run -p 8080:8080 \
  -e PORT=8080 \
  fashion-classifier:prod

Troubleshooting
Issue: Port Already in Use
bash# Solution: Change port
docker-compose up
# Edit docker-compose.yml: ports: - "5001:5000"
Issue: Permission Denied
bash# Solution: Run as root or fix permissions
sudo docker-compose up
# Or: chmod -R 777 data/ logs/
Issue: Model Not Loading
bash# Solution: Check volume mapping
docker-compose exec api ls -la /app/models/
# Should show fashion_classifier.keras
Issue: Out of Memory
bash# Solution: Increase Docker memory
# Docker Desktop → Settings → Resources → Memory: 4GB+

Docker Best Practices
1. Use .dockerignore:
__pycache__/
*.pyc
.git/
venv/
data/raw/
logs/*.log
2. Multi-stage Builds:
dockerfile# Build stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
CMD ["python", "api/app.py"]
3. Health Checks:
dockerfileHEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

Resources

Docker Docs: https://docs.docker.com/
Docker Compose: https://docs.docker.com/compose/
Best Practices: https://docs.docker.com/develop/dev-best-practices/


Last Updated: October 2025