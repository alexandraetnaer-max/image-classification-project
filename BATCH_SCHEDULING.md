Batch Processing Automation
Automatic Scheduling on Windows
Option 1: Task Scheduler (Recommended)

Open Task Scheduler (search "Task Scheduler" in Start menu)
Click "Create Basic Task"
Configure:

Name: Fashion Image Batch Processing
Description: Automatically process new fashion product images
Trigger: Daily at 2:00 AM
Action: Start a program
Program: Full path to run_batch_processing.bat
Start in: Project directory path


Advanced settings:

Run whether user is logged on or not
Run with highest privileges



Option 2: Manual Execution
Run batch processing manually:
python src/batch_processor.py
Workflow

Place images in data/incoming/ folder
Automatic processing runs at 2:00 AM daily
Results saved to results/batch_results/
Images organized in data/processed_batches/
Logs saved to logs/

Monitoring

Latest log: logs/batch_YYYYMMDD_HHMMSS.log
Results: results/batch_results/
Summary: JSON files in results folder

Troubleshooting
API not available: Make sure API is running with python api/app.py
No images found: Place images in data/incoming/ folder
Permission errors: Run Task Scheduler with administrator privileges

