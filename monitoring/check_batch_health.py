"""
Batch Processing Health Monitor
Checks the status of recent batch runs
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
RESULTS_DIR = Path('results/batch_results')
LOGS_DIR = Path('logs')
MAX_AGE_HOURS = 26  # Alert if no run in last 26 hours

def check_recent_runs():
    """Check if batch processing ran recently"""
    if not RESULTS_DIR.exists():
        return False, "Results directory not found"
    
    # Find most recent result file
    result_files = sorted(RESULTS_DIR.glob('batch_results_*.csv'))
    
    if not result_files:
        return False, "No batch results found"
    
    latest_file = result_files[-1]
    file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
    age_hours = (datetime.now() - file_time).total_seconds() / 3600
    
    if age_hours > MAX_AGE_HOURS:
        return False, f"Last run was {age_hours:.1f} hours ago"
    
    return True, f"Last run was {age_hours:.1f} hours ago"

def check_recent_errors():
    """Check recent logs for errors"""
    if not LOGS_DIR.exists():
        return []
    
    errors = []
    log_files = sorted(LOGS_DIR.glob('batch_*.log'))[-5:]  # Last 5 runs
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ERROR' in content or 'Failed' in content:
                errors.append(log_file.name)
    
    return errors

def get_statistics():
    """Get batch processing statistics"""
    if not RESULTS_DIR.exists():
        return {}
    
    summary_files = sorted(RESULTS_DIR.glob('summary_*.json'))
    
    if not summary_files:
        return {}
    
    # Read last 7 summaries
    stats = {
        'total_processed': 0,
        'total_successful': 0,
        'total_failed': 0,
        'runs': 0
    }
    
    for summary_file in summary_files[-7:]:
        with open(summary_file, 'r') as f:
            data = json.load(f)
            stats['total_processed'] += data.get('total_processed', 0)
            stats['total_successful'] += data.get('successful', 0)
            stats['total_failed'] += data.get('failed', 0)
            stats['runs'] += 1
    
    if stats['total_processed'] > 0:
        stats['success_rate'] = (stats['total_successful'] / stats['total_processed']) * 100
    else:
        stats['success_rate'] = 0
    
    return stats

def main():
    print("=" * 60)
    print("BATCH PROCESSING HEALTH CHECK")
    print("=" * 60)
    print()
    
    # Check recent runs
    is_healthy, message = check_recent_runs()
    status = "✓ HEALTHY" if is_healthy else "✗ WARNING"
    print(f"Recent Execution: {status}")
    print(f"  {message}")
    print()
    
    # Check for errors
    errors = check_recent_errors()
    if errors:
        print(f"Recent Errors: ✗ {len(errors)} log(s) with errors")
        for error_log in errors:
            print(f"  - {error_log}")
    else:
        print("Recent Errors: ✓ No errors detected")
    print()
    
    # Statistics
    stats = get_statistics()
    if stats:
        print("Last 7 Days Statistics:")
        print(f"  Total runs: {stats['runs']}")
        print(f"  Images processed: {stats['total_processed']}")
        print(f"  Successful: {stats['total_successful']}")
        print(f"  Failed: {stats['total_failed']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
    else:
        print("Statistics: No data available")
    
    print()
    print("=" * 60)
    
    return is_healthy

if __name__ == "__main__":
    healthy = main()
    exit(0 if healthy else 1)