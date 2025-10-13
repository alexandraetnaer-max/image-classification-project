"""
Alerting System for Fashion Classification
Sends notifications via email and Slack when issues detected
"""
import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import os

# Configuration
ALERT_EMAIL = os.getenv('ALERT_EMAIL', 'admin@example.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK', '')

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_log = Path('logs/alerts.log')
        self.alert_log.parent.mkdir(exist_ok=True)
    
    def log_alert(self, severity, message, details=None):
        """
        Log alert to file
        
        Args:
            severity: 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            message: Alert message
            details: Additional details (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        alert = {
            'timestamp': timestamp,
            'severity': severity,
            'message': message,
            'details': details
        }
        
        self.alert_history.append(alert)
        
        # Log to file
        with open(self.alert_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {severity}: {message}\n")
            if details:
                f.write(f"  Details: {details}\n")
        
        return alert
    
    def send_email_alert(self, subject, message, severity='WARNING'):
        """
        Send email alert
        
        Args:
            subject: Email subject
            message: Alert message
            severity: Alert severity level
        """
        if not SMTP_USER or not SMTP_PASSWORD:
            print("Email credentials not configured. Skipping email alert.")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = SMTP_USER
            msg['To'] = ALERT_EMAIL
            msg['Subject'] = f"[{severity}] Fashion Classifier - {subject}"
            
            # HTML body
            html_body = f"""
            <html>
              <head>
                <style>
                  body {{ font-family: Arial, sans-serif; }}
                  .alert {{ padding: 20px; border-radius: 5px; }}
                  .warning {{ background-color: #fff3cd; border: 1px solid #ffc107; }}
                  .error {{ background-color: #f8d7da; border: 1px solid #dc3545; }}
                  .critical {{ background-color: #dc3545; color: white; }}
                  .info {{ background-color: #d1ecf1; border: 1px solid #17a2b8; }}
                </style>
              </head>
              <body>
                <div class="alert {severity.lower()}">
                  <h2>{subject}</h2>
                  <p><strong>Severity:</strong> {severity}</p>
                  <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                  <hr>
                  <p>{message}</p>
                </div>
                <br>
                <p><em>This is an automated alert from Fashion Classification System</em></p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            
            print(f"✓ Email alert sent: {subject}")
            self.log_alert(severity, f"Email sent: {subject}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send email: {e}")
            self.log_alert('ERROR', f"Email failed: {e}")
            return False
    
    def send_slack_alert(self, message, severity='WARNING'):
        """
        Send Slack alert
        
        Args:
            message: Alert message
            severity: Alert severity level
        """
        if not SLACK_WEBHOOK:
            print("Slack webhook not configured. Skipping Slack alert.")
            return False
        
        try:
            # Color coding
            colors = {
                'INFO': '#36a64f',      # Green
                'WARNING': '#ff9800',   # Orange
                'ERROR': '#f44336',     # Red
                'CRITICAL': '#9c27b0'   # Purple
            }
            
            color = colors.get(severity, '#808080')
            
            # Create Slack message
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{severity}: Fashion Classifier Alert",
                        "text": message,
                        "footer": "Fashion Classification System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(
                SLACK_WEBHOOK,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                print(f"✓ Slack alert sent")
                self.log_alert(severity, f"Slack sent: {message}")
                return True
            else:
                print(f"✗ Slack alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to send Slack alert: {e}")
            self.log_alert('ERROR', f"Slack failed: {e}")
            return False
    
    def send_alert(self, subject, message, severity='WARNING', channels=None):
        """
        Send alert through multiple channels
        
        Args:
            subject: Alert subject
            message: Alert message
            severity: 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            channels: List of channels ['email', 'slack'] or None for all
        """
        if channels is None:
            channels = ['email', 'slack']
        
        results = {}
        
        if 'email' in channels:
            results['email'] = self.send_email_alert(subject, message, severity)
        
        if 'slack' in channels:
            results['slack'] = self.send_slack_alert(message, severity)
        
        return results

# Pre-defined alert scenarios
class AlertScenarios:
    """Common alert scenarios"""
    
    def __init__(self, alert_manager):
        self.alerts = alert_manager
    
    def api_down(self):
        """Alert: API is down"""
        self.alerts.send_alert(
            subject="API Down",
            message="Fashion Classification API is not responding. Please check the service immediately.",
            severity="CRITICAL"
        )
    
    def high_error_rate(self, error_rate, threshold=10):
        """Alert: High error rate detected"""
        self.alerts.send_alert(
            subject="High Error Rate",
            message=f"Error rate is {error_rate:.1f}% (threshold: {threshold}%). "
                   f"This may indicate issues with the model or API.",
            severity="ERROR"
        )
    
    def batch_processing_failed(self, error_message):
        """Alert: Batch processing failed"""
        self.alerts.send_alert(
            subject="Batch Processing Failed",
            message=f"Nightly batch processing encountered an error:\n{error_message}\n\n"
                   f"Please check logs for details.",
            severity="ERROR"
        )
    
    def low_confidence_predictions(self, count, threshold=50):
        """Alert: Many low confidence predictions"""
        self.alerts.send_alert(
            subject="Low Confidence Predictions",
            message=f"{count} predictions with confidence < 70% detected. "
                   f"This may indicate poor image quality or model degradation.",
            severity="WARNING"
        )
    
    def disk_space_low(self, available_gb):
        """Alert: Low disk space"""
        self.alerts.send_alert(
            subject="Low Disk Space",
            message=f"Only {available_gb:.1f} GB available. "
                   f"Please clean up old logs and results.",
            severity="WARNING"
        )
    
    def model_not_loaded(self):
        """Alert: Model failed to load"""
        self.alerts.send_alert(
            subject="Model Load Failed",
            message="ML model failed to load. API is running but predictions will fail. "
                   "Check model files in models/ directory.",
            severity="CRITICAL"
        )
    
    def batch_processing_success(self, stats):
        """Info: Batch processing completed successfully"""
        self.alerts.send_alert(
            subject="Batch Processing Complete",
            message=f"Nightly batch processing completed successfully.\n\n"
                   f"Stats:\n"
                   f"- Total processed: {stats.get('total', 0)}\n"
                   f"- Success rate: {stats.get('success_rate', 0):.1f}%\n"
                   f"- Processing time: {stats.get('time', 0):.1f}s",
            severity="INFO",
            channels=['slack']  # Only Slack for success
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize
    alerts = AlertManager()
    scenarios = AlertScenarios(alerts)
    
    # Test alerts
    print("Testing alert system...")
    print()
    
    # Test 1: Info alert
    print("1. Testing INFO alert...")
    alerts.send_alert(
        subject="System Test",
        message="This is a test alert to verify the alerting system is working.",
        severity="INFO"
    )
    
    # Test 2: Warning alert
    print("\n2. Testing WARNING alert...")
    scenarios.low_confidence_predictions(count=25)
    
    # Test 3: Error alert
    print("\n3. Testing ERROR alert...")
    scenarios.high_error_rate(error_rate=15.5)
    
    # Test 4: Critical alert
    print("\n4. Testing CRITICAL alert...")
    scenarios.api_down()
    
    print("\n✓ Alert tests completed")
    print(f"Check alert log: {alerts.alert_log}")