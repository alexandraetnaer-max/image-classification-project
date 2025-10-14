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
```
logs/
├── api.log                 # All API requests and responses
├── api_errors.log          # API errors only
├── batch_YYYYMMDD_HHMMSS.log    # Batch processing logs
└── batch_runner_YYYYMMDD_HHMMSS.log  # Scheduler logs
```

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
```

**Latest errors only:**
```bash
# Windows
findstr ERROR logs\api_errors.log

# Linux/Mac
grep ERROR logs/api_errors.log
```

**Follow logs in real-time:**
```bash
# Linux/Mac
tail -f logs/api.log

# Windows PowerShell
Get-Content logs\api.log -Wait -Tail 50
```

---

## API Monitoring

### Available Endpoints

**1. Health Check**
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": 10,
  "timestamp": "2025-10-07T12:00:00"
}
```

**2. Statistics**
```bash
GET /stats
```
**Response:**
```json
{
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
```

---

## Batch Processing Monitoring

Each batch run creates detailed logs with:

- Start/end timestamps  
- Images processed  
- Success/failure status  
- Category distribution  
- Processing time  

---

## Health Checks

### Automated Health Monitoring Script

Create `monitoring/health_check.sh`:
```bash
#!/bin/bash

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
```

---

## Dashboard

```bash
python monitoring/dashboard.py
```

---

## Alerts and Notifications

### Email Notifications

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(subject, message, severity='INFO'):
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
```

---

## Support

- **GitHub Issues:** Report monitoring issues  
- **Logs Location:** logs/ directory  
- **Dashboard Reports:** monitoring/reports/
