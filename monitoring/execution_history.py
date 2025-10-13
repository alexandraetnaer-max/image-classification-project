"""
Execution History Tracker
Tracks all batch processing runs and API usage
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

class ExecutionHistoryDB:
    """SQLite database for execution history"""
    
    def __init__(self, db_path='logs/execution_history.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Batch runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_images INTEGER,
                    successful INTEGER,
                    failed INTEGER,
                    success_rate REAL,
                    duration_seconds REAL,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            # API calls table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT,
                    status_code INTEGER,
                    response_time REAL,
                    success BOOLEAN
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT,
                    subject TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            conn.commit()
    
    def log_batch_run(self, stats):
        """
        Log batch processing run
        
        Args:
            stats: Dictionary with run statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO batch_runs (
                    timestamp, total_images, successful, failed,
                    success_rate, duration_seconds, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats.get('timestamp', datetime.now().isoformat()),
                stats.get('total_images', 0),
                stats.get('successful', 0),
                stats.get('failed', 0),
                stats.get('success_rate', 0.0),
                stats.get('duration_seconds', 0.0),
                stats.get('status', 'unknown'),
                stats.get('error_message', None)
            ))
            conn.commit()
    
    def log_api_call(self, endpoint, status_code, response_time, success=True):
        """Log API call"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_calls (
                    timestamp, endpoint, status_code, response_time, success
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                endpoint,
                status_code,
                response_time,
                success
            ))
            conn.commit()
    
    def log_alert(self, severity, subject, message):
        """Log alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, severity, subject, message)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                severity,
                subject,
                message
            ))
            conn.commit()
    
    def get_recent_batch_runs(self, days=7):
        """Get recent batch runs"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM batch_runs
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', conn, params=(f'-{days} days',))
        return df
    
    def get_failed_runs(self, days=30):
        """Get failed batch runs"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM batch_runs
                WHERE status = 'failed'
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', conn, params=(f'-{days} days',))
        return df
    
    def get_api_statistics(self, hours=24):
        """Get API call statistics"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    AVG(response_time) as avg_response_time,
                    MAX(response_time) as max_response_time
                FROM api_calls
                WHERE timestamp >= datetime('now', ?)
            ''', conn, params=(f'-{hours} hours',))
        return df.iloc[0].to_dict() if not df.empty else {}
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'batch_processing': {},
            'api_health': {},
            'recent_alerts': []
        }
        
        # Batch processing stats (last 7 days)
        batch_runs = self.get_recent_batch_runs(days=7)
        if not batch_runs.empty:
            report['batch_processing'] = {
                'total_runs': len(batch_runs),
                'successful_runs': len(batch_runs[batch_runs['status'] == 'success']),
                'failed_runs': len(batch_runs[batch_runs['status'] == 'failed']),
                'avg_success_rate': batch_runs['success_rate'].mean(),
                'total_images_processed': batch_runs['total_images'].sum()
            }
        
        # API health (last 24 hours)
        api_stats = self.get_api_statistics(hours=24)
        report['api_health'] = api_stats
        
        # Recent alerts (last 7 days)
        with sqlite3.connect(self.db_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 10
            ''', conn)
            report['recent_alerts'] = alerts_df.to_dict('records') if not alerts_df.empty else []
        
        return report

def generate_execution_report():
    """Generate and display execution history report"""
    db = ExecutionHistoryDB()
    
    print("=" * 70)
    print("EXECUTION HISTORY REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Batch processing history
    print("BATCH PROCESSING (Last 7 Days):")
    print("-" * 70)
    batch_runs = db.get_recent_batch_runs(days=7)
    
    if not batch_runs.empty:
        print(f"  Total runs: {len(batch_runs)}")
        print(f"  Successful: {len(batch_runs[batch_runs['status'] == 'success'])}")
        print(f"  Failed: {len(batch_runs[batch_runs['status'] == 'failed'])}")
        print(f"  Avg success rate: {batch_runs['success_rate'].mean():.1f}%")
        print(f"  Total images: {batch_runs['total_images'].sum()}")
        print()
        
        print("  Recent runs:")
        for idx, row in batch_runs.head(5).iterrows():
            status_icon = "✓" if row['status'] == 'success' else "✗"
            print(f"    {status_icon} {row['timestamp'][:19]} - "
                  f"{row['total_images']} images, "
                  f"{row['success_rate']:.1f}% success")
    else:
        print("  No batch runs recorded")
    
    print()
    
    # Failed runs
    print("FAILED RUNS (Last 30 Days):")
    print("-" * 70)
    failed_runs = db.get_failed_runs(days=30)
    
    if not failed_runs.empty:
        print(f"  Total failed: {len(failed_runs)}")
        print()
        print("  Details:")
        for idx, row in failed_runs.head(5).iterrows():
            print(f"    ✗ {row['timestamp'][:19]}")
            if row['error_message']:
                print(f"       Error: {row['error_message']}")
    else:
        print("  ✓ No failed runs!")
    
    print()
    
    # API statistics
    print("API HEALTH (Last 24 Hours):")
    print("-" * 70)
    api_stats = db.get_api_statistics(hours=24)
    
    if api_stats and api_stats.get('total_calls', 0) > 0:
        print(f"  Total calls: {api_stats['total_calls']}")
        print(f"  Successful: {api_stats['successful_calls']}")
        print(f"  Success rate: {(api_stats['successful_calls'] / api_stats['total_calls'] * 100):.1f}%")
        print(f"  Avg response time: {api_stats['avg_response_time']:.2f}s")
        print(f"  Max response time: {api_stats['max_response_time']:.2f}s")
    else:
        print("  No API calls recorded")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    generate_execution_report()