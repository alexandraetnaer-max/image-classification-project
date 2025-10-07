@echo off
echo Starting Batch Processing...
cd /d %~dp0

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run batch processor
python src\batch_processor.py

REM Keep window open if there were errors
if errorlevel 1 pause

echo Batch processing complete