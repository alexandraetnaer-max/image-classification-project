#!/bin/bash

###############################################################################
# Fashion Product Batch Processing Runner
# Automates nightly image classification
###############################################################################

set -e  # Exit on error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
VENV_PATH="$PROJECT_DIR/venv"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/batch_runner_$TIMESTAMP.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting Batch Processing"
log "=========================================="

# Navigate to project directory
cd "$PROJECT_DIR"
log "Working directory: $PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log "ERROR: Virtual environment not found at $VENV_PATH"
    log "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check Python
PYTHON_VERSION=$(python --version 2>&1)
log "Using Python: $PYTHON_VERSION"

# Check if batch processor exists
if [ ! -f "src/batch_processor.py" ]; then
    log "ERROR: batch_processor.py not found"
    exit 1
fi

# Run batch processor
log "Running batch processor..."
python src/batch_processor.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    log "=========================================="
    log "Batch Processing Completed Successfully"
    log "=========================================="
    
    # Optional: Send success notification
    # echo "Batch processing completed successfully" | mail -s "Fashion Classifier Success" admin@example.com
else
    log "=========================================="
    log "Batch Processing Failed with exit code: $EXIT_CODE"
    log "=========================================="
    
    # Optional: Send error notification
    # echo "Batch processing failed. Check logs: $LOG_FILE" | mail -s "Fashion Classifier ERROR" admin@example.com
    
    exit $EXIT_CODE
fi

# Optional: Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "batch_runner_*.log" -type f -mtime +30 -delete
log "Cleaned up old log files"

exit 0