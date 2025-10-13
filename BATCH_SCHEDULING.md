# Batch Processing Automation Guide

## Overview

The batch processing system automatically classifies fashion product images overnight. This document provides detailed instructions for setting up automated execution on different platforms.

---

## Table of Contents

1. [Windows Task Scheduler](#windows-task-scheduler)
2. [Linux/Mac Cron](#linuxmac-cron)
3. [GitHub Actions (Cloud)](#github-actions)
4. [Docker-based Automation](#docker-based-automation)
5. [Results Storage](#results-storage)
6. [Monitoring](#monitoring)

---

## Windows Task Scheduler

### Prerequisites

- Python environment set up
- API accessible (local or cloud)

### Step-by-Step Setup

#### 1. Test Manual Execution

First, verify the batch processor works:
```cmd
cd C:\path\to\image-classification-project
venv\Scripts\activate
python src/batch_processor.py
2. Open Task Scheduler

Press Win + R
Type taskschd.msc
Press Enter

3. Create Basic Task

Click "Create Basic Task" in the right panel
Name: Fashion Image Batch Processing
Description: Automatically process new fashion product images nightly
Click Next

4. Set Trigger

Select "Daily"
Click Next
Set Start time: 02:00:00 (2:00 AM)
Set Recur every: 1 days
Click Next

5. Set Action

Select "Start a program"
Click Next
Program/script: Browse to run_batch_processing.bat

Full path: C:\Users\YourName\Documents\image-classification-project\run_batch_processing.bat


Start in: C:\Users\YourName\Documents\image-classification-project\
Click Next

6. Advanced Settings

Check "Open the Properties dialog..."
Click Finish
In Properties dialog:

General tab:

Select "Run whether user is logged on or not"
Check "Run with highest privileges"


Conditions tab:

Uncheck "Start the task only if the computer is on AC power" (for laptops)


Settings tab:

Check "If the task fails, restart every: 10 minutes"
Set "Attempt to restart up to: 3 times"




Click OK

7. Test Scheduled Task
Right-click the task → Run
Check logs in logs/ folder to verify execution.

Linux/Mac Cron
Prerequisites
bashcd /path/to/image-classification-project
source venv/bin/activate
python src/batch_processor.py  # Test manually
Cron Setup
1. Create Wrapper Script
Create run_batch_processing.sh:
bash#!/bin/bash

# Navigate to project directory
cd /path/to/image-classification-project

# Activate virtual environment
source venv/bin/activate

# Run batch processor
python src/batch_processor.py

# Deactivate
deactivate

# Optional: Send notification on completion
# echo "Batch processing completed" | mail -s "Fashion Classifier" your@email.com
Make executable:
bashchmod +x run_batch_processing.sh
2. Edit Crontab
bashcrontab -e
Add this line for daily 2:00 AM execution:
cron0 2 * * * /path/to/image-classification-project/run_batch_processing.sh >> /path/to/image-classification-project/logs/cron.log 2>&1
Cron Syntax Explained:
0 2 * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, Sunday = 0 or 7)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
Other Schedule Examples:
cron0 */6 * * *    # Every 6 hours
0 2 * * 1-5    # 2 AM on weekdays only
0 2 1 * *      # 2 AM on 1st of each month
*/30 * * * *   # Every 30 minutes
3. Verify Cron Job
bashcrontab -l  # List all cron jobs

GitHub Actions
For cloud-based automation without maintaining a server.
Create Workflow File
Create .github/workflows/batch-processing.yml:
yamlname: Nightly Batch Processing

on:
  schedule:
    # Runs at 02:00 UTC every day
    - cron: '0 2 * * *'
  workflow_dispatch:  # Allows manual trigger

jobs:
  batch-process:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install requests pandas pillow
    
    - name: Download images from storage
      run: |
        # If images are in cloud storage, download them
        # aws s3 sync s3://your-bucket/incoming data/incoming/
        # Or use GitHub artifacts from another workflow
        echo "Images ready for processing"
    
    - name: Run batch processor
      env:
        API_URL: ${{ secrets.API_URL }}
      run: |
        python src/batch_processor.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: batch-results
        path: |
          results/batch_results/
          logs/
        retention-days: 30
    
    - name: Commit and push results (optional)
      run: |
        git config user.name "GitHub Actions Bot"
        git config user.email "actions@github.com"
        git add results/ logs/
        git commit -m "Batch processing results $(date +'%Y-%m-%d')" || echo "No changes"
        git push || echo "Nothing to push"
Setup Secrets
In GitHub repository:

Go to Settings → Secrets and variables → Actions
Add secret: API_URL = https://your-cloud-api.com

Manual Trigger

Go to Actions tab in GitHub
Select "Nightly Batch Processing"
Click "Run workflow"


Docker-based Automation
Docker Compose with Scheduler
Create docker-compose.scheduler.yml:
yamlversion: '3.8'

services:
  batch-scheduler:
    build: .
    container_name: batch-scheduler
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - API_URL=https://fashion-classifier-api.run.app
    command: >
      sh -c "
      while true; do
        echo 'Running batch processing...'
        python src/batch_processor.py
        echo 'Sleeping until next run (24 hours)...'
        sleep 86400
      done
      "
    restart: unless-stopped
Start scheduler:
bashdocker-compose -f docker-compose.scheduler.yml up -d
Kubernetes CronJob
For production environments:
yamlapiVersion: batch/v1
kind: CronJob
metadata:
  name: fashion-batch-processor
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: batch-processor
            image: your-registry/fashion-classifier:latest
            env:
            - name: API_URL
              value: "https://fashion-classifier-api.run.app"
            command:
            - python
            - src/batch_processor.py
          restartPolicy: OnFailure

Results Storage
Directory Structure
results/
└── batch_results/
    ├── batch_results_20251007_020000.csv
    ├── batch_results_20251008_020000.csv
    ├── summary_20251007_020000.json
    └── summary_20251008_020000.json

data/
├── incoming/              # New images placed here
│   ├── image1.jpg
│   └── image2.jpg
└── processed_batches/     # Organized after processing
    ├── Tshirts/
    │   ├── image1.jpg
    │   └── ...
    ├── Shoes/
    └── ...

logs/
├── batch_20251007_020000.log
└── batch_20251008_020000.log
CSV Format
batch_results_YYYYMMDD_HHMMSS.csv:
csvfilename,category,confidence,status,timestamp
image1.jpg,Tshirts,0.9876,success,2025-10-07T02:01:23
image2.jpg,Shoes,0.8543,success,2025-10-07T02:01:25
image3.jpg,Handbags,0.9234,success,2025-10-07T02:01:27
JSON Summary Format
summary_YYYYMMDD_HHMMSS.json:
json{
  "timestamp": "20251007_020000",
  "total_processed": 150,
  "successful": 148,
  "failed": 2,
  "categories": {
    "Tshirts": 45,
    "Shoes": 32,
    "Handbags": 28,
    "Watches": 20,
    "Tops": 15,
    "Sunglasses": 8
  },
  "average_confidence": 0.89,
  "processing_time_seconds": 180
}
Automatic Cleanup
Add to batch_processor.py or create separate script:
pythonimport os
from datetime import datetime, timedelta

def cleanup_old_results(days=30):
    """Remove results older than specified days"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    results_dir = 'results/batch_results'
    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        if file_time < cutoff_date:
            os.remove(filepath)
            print(f"Removed old file: {filename}")

Monitoring
Check Last Run
bash# Windows
dir logs\*.log /O-D /B

# Linux/Mac
ls -lt logs/*.log | head -1
View Latest Results
bash# Windows
type results\batch_results\batch_results_*.csv | findstr /V "filename"

# Linux/Mac
tail -n 20 results/batch_results/batch_results_*.csv
Email Notifications
Add to run_batch_processing.bat (Windows):
batch@echo off
python src/batch_processor.py

if %errorlevel% neq 0 (
    echo Batch processing failed! | mail -s "Alert: Batch Processing Failed" admin@example.com
)
Or Python-based notification:
pythonimport smtplib
from email.mime.text import MIMEText

def send_notification(subject, message):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'batch@yourcompany.com'
    msg['To'] = 'admin@yourcompany.com'
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your-email', 'your-password')
        server.send_message(msg)
Dashboard (Optional)
Create monitoring/dashboard.py:
pythonimport pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def generate_dashboard():
    # Read all results
    csv_files = glob('results/batch_results/batch_results_*.csv')
    
    if not csv_files:
        print("No results found")
        return
    
    # Combine all results
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    
    # Generate statistics
    print(f"Total images processed: {len(df)}")
    print(f"Success rate: {(df['status']=='success').sum()/len(df)*100:.1f}%")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    
    # Plot
    df['category'].value_counts().plot(kind='bar')
    plt.title('Images by Category')
    plt.tight_layout()
    plt.savefig('results/category_distribution.png')
    print("Dashboard saved to results/category_distribution.png")

if __name__ == "__main__":
    generate_dashboard()

Troubleshooting
Common Issues
1. Batch processor can't connect to API

Check if API is running: curl https://your-api-url/health
Verify API_URL in configuration
Check firewall/network settings

2. Permission errors

Run with elevated privileges (Windows: Run as Administrator)
Check file/folder permissions: chmod -R 755 data/

3. No images processed

Verify images exist in data/incoming/
Check image formats are supported
Review logs for errors

4. Out of memory

Reduce batch size in batch_processor.py
Process images in smaller chunks
Increase system resources


Best Practices

Test First: Always test manually before scheduling
Monitor Logs: Check logs regularly for errors
Backup Results: Archive old results before cleanup
Error Handling: Implement retry logic for transient failures
Documentation: Keep this document updated with changes


Support

GitHub Issues: https://github.com/alexandraetnaer-max/image-classification-project/issues
API Documentation: See api/README.md
Architecture: See ARCHITECTURE.md
