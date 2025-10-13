markdown# Monitoring Examples and Alert Scenarios

Practical examples for monitoring the Fashion Classification System with real-world alert scenarios.

---

## Table of Contents

1. [Setting Up Alerts](#setting-up-alerts)
2. [Email Alerts](#email-alerts)
3. [Slack Alerts](#slack-alerts)
4. [Execution History](#execution-history)
5. [Common Alert Scenarios](#common-alert-scenarios)
6. [Automated Monitoring](#automated-monitoring)
7. [Troubleshooting Alerts](#troubleshooting-alerts)

---

## Setting Up Alerts

### Environment Variables

Create `.env` file in project root:
```bash
# Email Configuration
ALERT_EMAIL=admin@yourcompany.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Slack Configuration
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
Gmail App Password Setup

Go to Google Account settings
Security → 2-Step Verification
App passwords → Generate new
Select "Mail" and "Other device"
Copy the 16-character password
Use in SMTP_PASSWORD

Slack Webhook Setup

Go to https://api.slack.com/apps
Create New App → From scratch
Add "Incoming Webhooks"
Activate and create webhook
Copy webhook URL
Use in SLACK_WEBHOOK


Email Alerts
Manual Test
pythonfrom monitoring.alerting import AlertManager

alerts = AlertManager()

# Send test email
alerts.send_email_alert(
    subject="Test Alert",
    message="This is a test email from the monitoring system.",
    severity="INFO"
)
Example Email Output
Subject: [WARNING] Fashion Classifier - High Error Rate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

High Error Rate

Severity: WARNING
Time: 2025-10-13 14:30:00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Error rate is 15.5% (threshold: 10%). This may 
indicate issues with the model or API.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated alert from Fashion 
Classification System

Slack Alerts
Manual Test
pythonfrom monitoring.alerting import AlertManager

alerts = AlertManager()

# Send test Slack message
alerts.send_slack_alert(
    message="Batch processing completed: 150 images, 98% success rate",
    severity="INFO"
)
Example Slack Output
🟢 INFO: Fashion Classifier Alert

Batch processing completed: 150 images, 98% success rate

Fashion Classification System
Today at 2:05 AM

Execution History
View Recent Runs
pythonfrom monitoring.execution_history import ExecutionHistoryDB

db = ExecutionHistoryDB()

# Get last 7 days of batch runs
recent = db.get_recent_batch_runs(days=7)
print(recent)
Output:
   id           timestamp  total_images  successful  failed  success_rate
0   5  2025-10-13 02:00:00           150         147       3          98.0
1   4  2025-10-12 02:00:00           200         198       2          99.0
2   3  2025-10-11 02:00:00           180         175       5          97.2
View Failed Runs
python# Get failed runs
failed = db.get_failed_runs(days=30)

for idx, row in failed.iterrows():
    print(f"❌ {row['timestamp']}: {row['error_message']}")
Generate Health Report
pythonfrom monitoring.execution_history import generate_execution_report

# Generate and display comprehensive report
generate_execution_report()
Output:
======================================================================
EXECUTION HISTORY REPORT
======================================================================
Generated: 2025-10-13 15:00:00

BATCH PROCESSING (Last 7 Days):
----------------------------------------------------------------------
  Total runs: 7
  Successful: 7
  Failed: 0
  Avg success rate: 98.2%
  Total images: 1,050

  Recent runs:
    ✓ 2025-10-13 02:00 - 150 images, 98.0% success
    ✓ 2025-10-12 02:00 - 200 images, 99.0% success
    ✓ 2025-10-11 02:00 - 180 images, 97.2% success

FAILED RUNS (Last 30 Days):
----------------------------------------------------------------------
  ✓ No failed runs!

API HEALTH (Last 24 Hours):
----------------------------------------------------------------------
  Total calls: 450
  Successful: 441
  Success rate: 98.0%
  Avg response time: 2.15s
  Max response time: 5.32s

======================================================================

Common Alert Scenarios
Scenario 1: API Down
When: API health check fails
Alert:
pythonfrom monitoring.alerting import AlertManager, AlertScenarios

alerts = AlertManager()
scenarios = AlertScenarios(alerts)

# Trigger alert
scenarios.api_down()
Email Subject: [CRITICAL] Fashion Classifier - API Down
Message:
Fashion Classification API is not responding. 
Please check the service immediately.
Action Required:

Check if API process is running
Review API logs: logs/api.log
Restart API: python api/app.py
Verify with: curl http://localhost:5000/health


Scenario 2: High Error Rate
When: Error rate exceeds 10%
Trigger:
pythonscenarios.high_error_rate(error_rate=15.5, threshold=10)
Alert Channels: Email + Slack
Message:
Error rate is 15.5% (threshold: 10%). 
This may indicate issues with the model or API.
Action Required:

Check recent images for quality issues
Review API error logs
Verify model is loaded correctly
Check if specific categories failing


Scenario 3: Batch Processing Failed
When: Batch run encounters fatal error
Trigger:
pythonscenarios.batch_processing_failed(
    error_message="ConnectionError: Unable to reach API"
)
Alert Channels: Email + Slack
Action Required:

Check batch processing logs
Verify API is accessible
Retry batch processing manually
Check disk space and permissions


Scenario 4: Low Confidence Predictions
When: Many predictions have confidence < 70%
Trigger:
pythonscenarios.low_confidence_predictions(count=45, threshold=50)
Alert Severity: WARNING
Message:
45 predictions with confidence < 70% detected. 
This may indicate poor image quality or model degradation.
Action Required:

Review low-confidence images
Check image quality (lighting, angle, clarity)
Consider retraining model if systematic issue
Flag images for manual review


Scenario 5: Disk Space Low
When: Available disk space < 5GB
Trigger:
pythonscenarios.disk_space_low(available_gb=3.2)
Alert Severity: WARNING
Action Required:

Clean up old logs: rm logs/*.log.1 logs/*.log.2
Archive old results
Remove old processed images
Increase disk space if needed


Scenario 6: Model Not Loaded
When: API starts but model fails to load
Trigger:
pythonscenarios.model_not_loaded()
Alert Severity: CRITICAL
Action Required:

Check model files exist in models/
Verify model file integrity
Check available RAM
Review API startup logs


Scenario 7: Batch Processing Success
When: Batch completes successfully (info only)
Trigger:
pythonscenarios.batch_processing_success({
    'total': 150,
    'success_rate': 98.0,
    'time': 180.5
})
Alert Channels: Slack only
Message:
✓ Nightly batch processing completed successfully.

Stats:
- Total processed: 150
- Success rate: 98.0%
- Processing time: 180.5s

Automated Monitoring
Daily Health Check Script
Create monitoring/daily_check.py:
python"""
Daily automated health check
Run via cron/Task Scheduler
"""
from monitoring.alerting import AlertManager, AlertScenarios
from monitoring.execution_history import ExecutionHistoryDB
import requests

def daily_health_check():
    """Perform daily health check and send alerts if needed"""
    
    alerts = AlertManager()
    scenarios = AlertScenarios(alerts)
    db = ExecutionHistoryDB()
    
    # Check 1: API Health
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code != 200:
            scenarios.api_down()
    except:
        scenarios.api_down()
    
    # Check 2: Recent batch runs
    recent_runs = db.get_recent_batch_runs(days=1)
    if recent_runs.empty:
        alerts.send_alert(
            subject="No Batch Processing",
            message="No batch processing runs detected in last 24 hours.",
            severity="WARNING"
        )
    
    # Check 3: Error rate
    if not recent_runs.empty:
        avg_success_rate = recent_runs['success_rate'].mean()
        if avg_success_rate < 90:
            alerts.send_alert(
                subject="Low Success Rate",
                message=f"Average success rate: {avg_success_rate:.1f}%",
                severity="WARNING"
            )
    
    # Check 4: Disk space
    import shutil
    stat = shutil.disk_usage('.')
    available_gb = stat.free / (1024**3)
    if available_gb < 5:
        scenarios.disk_space_low(available_gb)
    
    print("✓ Daily health check completed")

if __name__ == "__main__":
    daily_health_check()
Schedule (Cron):
bash# Run daily at 9 AM
0 9 * * * cd /path/to/project && python monitoring/daily_check.py

Continuous Monitoring Script
Create monitoring/continuous_monitor.py:
python"""
Continuous monitoring - runs in background
"""
import time
import requests
from monitoring.alerting import AlertManager, AlertScenarios

def continuous_monitor(check_interval=300):
    """
    Monitor API health continuously
    
    Args:
        check_interval: Seconds between checks (default: 5 minutes)
    """
    alerts = AlertManager()
    scenarios = AlertScenarios(alerts)
    consecutive_failures = 0
    
    print(f"Starting continuous monitoring (check every {check_interval}s)")
    
    while True:
        try:
            response = requests.get('http://localhost:5000/health', timeout=10)
            
            if response.status_code == 200:
                consecutive_failures = 0
                print("✓ API healthy")
            else:
                consecutive_failures += 1
                print(f"⚠ API returned {response.status_code}")
                
                if consecutive_failures >= 3:
                    scenarios.api_down()
                    consecutive_failures = 0
                    
        except Exception as e:
            consecutive_failures += 1
            print(f"✗ API check failed: {e}")
            
            if consecutive_failures >= 3:
                scenarios.api_down()
                consecutive_failures = 0
        
        time.sleep(check_interval)

if __name__ == "__main__":
    continuous_monitor(check_interval=300)  # 5 minutes
Run in background:
bash# Linux/Mac
nohup python monitoring/continuous_monitor.py &

# Windows (use Task Scheduler to run at startup)

Troubleshooting Alerts
Issue: Emails not sending
Possible causes:

Gmail blocking "less secure apps"
Wrong app password
Firewall blocking SMTP port

Solutions:
python# Test SMTP connection
import smtplib

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your-email@gmail.com', 'your-app-password')
    print("✓ SMTP connection successful")
    server.quit()
except Exception as e:
    print(f"✗ SMTP connection failed: {e}")

Issue: Slack webhooks not working
Solutions:
python# Test webhook
import requests
import json

webhook = 'YOUR_WEBHOOK_URL'
payload = {"text": "Test message"}

response = requests.post(webhook, data=json.dumps(payload))
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

Issue: Too many alerts
Solutions:

Rate limiting:

pythonclass AlertManager:
    def __init__(self):
        self.last_alert_time = {}
        self.min_alert_interval = 3600  # 1 hour
    
    def should_send_alert(self, alert_type):
        """Check if enough time passed since last alert"""
        last_time = self.last_alert_time.get(alert_type, 0)
        current_time = time.time()
        
        if current_time - last_time > self.min_alert_interval:
            self.last_alert_time[alert_type] = current_time
            return True
        return False

Alert digest:
Send one daily summary instead of individual alerts.


Best Practices

Test alerts first: Always test with INFO severity before deploying
Set appropriate thresholds: Adjust based on your workload
Use rate limiting: Avoid alert fatigue
Monitor the monitors: Ensure alerting system itself is working
Document on-call procedures: Clear steps for each alert type
Regular review: Check alert history weekly to tune thresholds


Support

Documentation: MONITORING.md
Execution History: python monitoring/execution_history.py
Dashboard: python monitoring/dashboard.py

