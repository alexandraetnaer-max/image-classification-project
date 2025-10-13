# Monitoring and Logging Guide

## Overview

The Fashion Classification system includes comprehensive monitoring and logging capabilities to track API performance, batch processing health, and system status.

---

## Table of Contents

1. [Logging System](#logging-system)
2. [API Monitoring](#api-monitoring)
3. [Batch Processing Monitoring](#batch-processing-monitoring)
4. [Health Checks](#health-checks)
5. [Dashboard](#dashboard)
6. [Alerts and Notifications](#alerts-and-notifications)

---

## Logging System

### Log Files Location
logs/
├── api.log                 # All API requests and responses
├── api_errors.log          # API errors only
├── batch_YYYYMMDD_HHMMSS.log    # Batch processing logs
└── batch_runner_YYYYMMDD_HHMMSS.log  # Scheduler logs

### Log Rotation

- **Max file size:** 10 MB per log file
- **Backup count:** 10 files kept
- **Total max size:** ~100 MB per log type

### Log Levels

- **INFO:** Normal operations, requests, responses
- **WARNING:** Recoverable issues
- **ERROR:** Failed operations, exceptions
- **CRITICAL:** System failures

### Viewing Logs

**Latest API logs:**
```bash
# Windows
type logs\api.log | more

# Linux/Mac
tail -f logs/api.log
Latest errors only:
bash# Windows
findstr ERROR logs\api_errors.log

# Linux/Mac
grep ERROR logs/api_errors.log
Follow logs in real-time:
bash# Linux/Mac
tail -f logs/api.log

# Windows PowerShell
Get-Content logs\api.log -Wait -Tail 50

API Monitoring
Available Endpoints
1. Health Check
bashGET /health
Response:
json{
  "status": "healthy",
  "model_loaded": true,
  "classes": 10,
  "timestamp": "2025-10-07T12:00:00"
}
Use case: Basic availability check

2. Statistics
bashGET /stats
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
  "timestamp": "2025-10-07T12:00:00"
}
Use case: Performance monitoring, success rate tracking

API Monitoring Best Practices
1. Regular Health Checks
bash# Check every 5 minutes with cron
*/5 * * * * curl -s https://your-api.com/health | grep "healthy" || echo "API DOWN"
2. Monitor Response Times
bashcurl -w "\nTime: %{time_total}s\n" https://your-api.com/health
3. Track Success Rate
pythonimport requests

response = requests.get('https://your-api.com/stats')
data = response.json()

if data['success_rate'] < 95:
    print(f"⚠️ Success rate dropped to {data['success_rate']:.1f}%")

Batch Processing Monitoring
Batch Process Logs
Each batch run creates detailed logs with:

Start/end timestamps
Images processed
Success/failure status
Category distribution
Processing time

Example log entry:
[2025-10-07 02:00:15] ============================================================
[2025-10-07 02:00:15] BATCH PROCESSING STARTED
[2025-10-07 02:00:15] ============================================================
[2025-10-07 02:00:17] ✓ API is healthy and ready
[2025-10-07 02:00:17] Found 150 images to process
[2025-10-07 02:00:17] 
Processing [1/150]: product_001.jpg
[2025-10-07 02:00:20]   ✓ Category: Tshirts (confidence: 98.5%)
[2025-10-07 02:00:20]   ✓ Moved to: data\processed_batches\Tshirts\product_001.jpg
Batch Results
CSV Format:
csvfilename,category,confidence,status,timestamp
product_001.jpg,Tshirts,0.985,success,2025-10-07T02:00:20
product_002.jpg,Shoes,0.923,success,2025-10-07T02:00:23
JSON Summary:
json{
  "timestamp": "20251007_020000",
  "total_processed": 150,
  "successful": 148,
  "failed": 2,
  "categories": {
    "Tshirts": 45,
    "Shoes": 32,
    "Handbags": 28
  }
}
Monitoring Batch Health
Check if batch ran recently:
bashpython monitoring/check_batch_health.py
Output:
============================================================
BATCH PROCESSING HEALTH CHECK
============================================================

Recent Execution: ✓ HEALTHY
  Last run was 2.3 hours ago

Recent Errors: ✓ No errors detected

Last 7 Days Statistics:
  Total runs: 7
  Images processed: 1,050
  Successful: 1,038
  Failed: 12
  Success rate: 98.9%

============================================================

Health Checks
Automated Health Monitoring Script
Create monitoring/health_check.sh:
bash#!/bin/bash

API_URL="https://your-api-url.com"
ALERT_EMAIL="admin@example.com"

# Check API health
response=$(curl -s "$API_URL/health")
status=$(echo $response | jq -r '.status')

if [ "$status" != "healthy" ]; then
    echo "API UNHEALTHY: $response" | mail -s "ALERT: API Down" $ALERT_EMAIL
    exit 1
fi

# Check batch processing
python monitoring/check_batch_health.py
if [ $? -ne 0 ]; then
    echo "Batch processing issues detected" | mail -s "WARNING: Batch Issues" $ALERT_EMAIL
fi

echo "All systems healthy"
Schedule with cron:
cron*/15 * * * * /path/to/health_check.sh

Dashboard
Generate Monitoring Dashboard
bashpython monitoring/dashboard.py
Generates:

Text report with system health
Visual charts (PNG)

Daily processing volume
Success/failure breakdown
Category distribution
Summary statistics



Output location: monitoring/reports/
Sample output:
FASHION CLASSIFIER - MONITORING REPORT
======================================================================
Generated: 2025-10-07 15:30:00

API HEALTH:
  ✓ Status: HEALTHY
     Message: API is logging

BATCH PROCESSING HEALTH:
  ✓ Status: HEALTHY
     Message: Last run 2.1 hours ago

BATCH STATISTICS (Last 7 Days):
  Total Runs: 7
  Images Processed: 1,050
  Success Rate: 98.9%
  Failed: 12

  Top Categories:
    - Tshirts: 315
    - Shoes: 240
    - Handbags: 189

DISK USAGE:
  results/batch_results: 45.23 MB
  logs: 12.45 MB
  data: 2.34 GB

RECENT ERRORS:
  ✓ No recent errors detected

======================================================================
Automated Daily Dashboard
Add to cron for daily reports:
cron0 9 * * * cd /path/to/project && python monitoring/dashboard.py && mail -s "Daily Report" admin@example.com < monitoring/reports/report_*.txt

Alerts and Notifications
Email Notifications
Python script for alerts:
pythonimport smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(subject, message, severity='INFO'):
    """Send email alert"""
    sender = "alerts@yourcompany.com"
    recipient = "admin@yourcompany.com"
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = f"[{severity}] {subject}"
    
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, 'your-app-password')
            server.send_message(msg)
        print(f"Alert sent: {subject}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Example usage
if success_rate < 90:
    send_alert(
        "Low Success Rate",
        f"Batch processing success rate dropped to {success_rate:.1f}%",
        severity='WARNING'
    )
Slack Notifications
pythonimport requests

def send_slack_alert(message):
    """Send alert to Slack"""
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    payload = {
        "text": message,
        "username": "Fashion Classifier Bot",
        "icon_emoji": ":robot_face:"
    }
    
    requests.post(webhook_url, json=payload)

# Example
send_slack_alert("⚠️ Batch processing detected 5 failures in last run")
Alert Triggers
Common alert conditions:

API Down - Health check fails
High Error Rate - Success rate < 95%
No Recent Batch Run - Last run > 26 hours ago
Disk Space Low - < 10% free space
High Response Time - Average > 5 seconds
Model Load Failure - model_loaded = false


Metrics to Monitor
Key Performance Indicators (KPIs)

API Uptime: Target 99.9%
Success Rate: Target > 95%
Average Response Time: Target < 2s
Daily Processing Volume: Track trends
Error Rate: Target < 5%

Dashboard Integration
For production environments, integrate with:

Grafana - Visualization
Prometheus - Metrics collection
Datadog - Full observability
New Relic - APM monitoring


Troubleshooting
Common Issues
1. API not responding
bash# Check if process is running
ps aux | grep gunicorn

# Check logs
tail -50 logs/api_errors.log

# Restart API
docker-compose restart
2. Batch processing stuck
bash# Check for running process
ps aux | grep batch_processor

# Kill if hung
pkill -f batch_processor

# Check logs
tail -100 logs/batch_*.log
3. High error rate
bash# Analyze errors
python monitoring/dashboard.py

# Check model health
curl https://your-api.com/health

# Review recent predictions
cat results/batch_results/batch_results_*.csv | grep "error"

Best Practices

Regular Monitoring: Check health daily
Log Retention: Keep logs for 30 days minimum
Automated Alerts: Set up for critical issues
Performance Tracking: Monitor trends over time
Documentation: Keep monitoring procedures updated


Support

GitHub Issues: Report monitoring issues
Logs Location: logs/ directory
Dashboard Reports: monitoring/reports/
