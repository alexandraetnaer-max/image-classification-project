"""
Comprehensive Monitoring Dashboard
Displays health status, statistics, and recent activity
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
RESULTS_DIR = Path('results/batch_results')
LOGS_DIR = Path('logs')
OUTPUT_DIR = Path('monitoring/reports')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class MonitoringDashboard:
    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.logs_dir = LOGS_DIR
        self.output_dir = OUTPUT_DIR
        
    def check_api_health(self):
        """Check API log health"""
        api_log = self.logs_dir / 'api.log'
        
        if not api_log.exists():
            return {
                'status': 'unknown',
                'message': 'No API logs found'
            }
        
        # Check last log entry time
        with open(api_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                # Parse timestamp if possible
                return {
                    'status': 'healthy',
                    'message': 'API is logging',
                    'last_activity': last_line[:19]  # Timestamp
                }
        
        return {'status': 'unknown', 'message': 'Empty log file'}
    
    def check_batch_health(self):
        """Check batch processing health"""
        if not self.results_dir.exists():
            return {
                'status': 'error',
                'message': 'Results directory not found'
            }
        
        result_files = sorted(self.results_dir.glob('batch_results_*.csv'))
        
        if not result_files:
            return {
                'status': 'warning',
                'message': 'No batch results found'
            }
        
        latest_file = result_files[-1]
        file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        
        if age_hours > 26:
            return {
                'status': 'warning',
                'message': f'Last run {age_hours:.1f} hours ago',
                'last_run': file_time.isoformat()
            }
        
        return {
            'status': 'healthy',
            'message': f'Last run {age_hours:.1f} hours ago',
            'last_run': file_time.isoformat()
        }
    
    def get_batch_statistics(self, days=7):
        """Get batch processing statistics"""
        summary_files = sorted(self.results_dir.glob('summary_*.json'))
        
        if not summary_files:
            return None
        
        # Get recent summaries
        recent_summaries = summary_files[-days:]
        
        stats = {
            'period_days': days,
            'total_runs': 0,
            'total_processed': 0,
            'total_successful': 0,
            'total_failed': 0,
            'categories': {},
            'daily_stats': []
        }
        
        for summary_file in recent_summaries:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
                stats['total_runs'] += 1
                stats['total_processed'] += data.get('total_processed', 0)
                stats['total_successful'] += data.get('successful', 0)
                stats['total_failed'] += data.get('failed', 0)
                
                # Aggregate categories
                for cat, count in data.get('categories', {}).items():
                    if cat:
                        stats['categories'][cat] = stats['categories'].get(cat, 0) + count
                
                # Daily stat
                stats['daily_stats'].append({
                    'date': data.get('timestamp', ''),
                    'processed': data.get('total_processed', 0),
                    'successful': data.get('successful', 0),
                    'failed': data.get('failed', 0)
                })
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = (stats['total_successful'] / stats['total_processed']) * 100
        else:
            stats['success_rate'] = 0
        
        return stats
    
    def check_disk_space(self):
        """Check disk space usage"""
        import shutil
        
        usage = {}
        
        for directory in [self.results_dir, self.logs_dir, Path('data')]:
            if directory.exists():
                total_size = sum(
                    f.stat().st_size for f in directory.rglob('*') if f.is_file()
                )
                usage[str(directory)] = {
                    'size_mb': total_size / (1024 * 1024),
                    'size_human': self._human_readable_size(total_size)
                }
        
        return usage
    
    def _human_readable_size(self, size_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def check_recent_errors(self):
        """Find recent errors in logs"""
        errors = []
        
        # Check API error log
        api_error_log = self.logs_dir / 'api_errors.log'
        if api_error_log.exists():
            with open(api_error_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                errors.extend([
                    {'source': 'API', 'message': line.strip()}
                    for line in lines[-10:]  # Last 10 errors
                ])
        
        # Check batch logs
        batch_logs = sorted(self.logs_dir.glob('batch_*.log'))[-5:]
        for log_file in batch_logs:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '✗' in content or 'ERROR' in content:
                    errors.append({
                        'source': 'Batch',
                        'file': log_file.name,
                        'message': 'Contains errors'
                    })
        
        return errors
    
    def generate_visualizations(self):
        """Generate visualization charts"""
        stats = self.get_batch_statistics(days=7)
        
        if not stats or not stats['daily_stats']:
            print("No data for visualizations")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fashion Classifier - Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Daily processing volume
        ax1 = axes[0, 0]
        dates = [item['date'][:8] for item in stats['daily_stats']]
        processed = [item['processed'] for item in stats['daily_stats']]
        
        ax1.bar(dates, processed, color='skyblue')
        ax1.set_title('Daily Processing Volume')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Images Processed')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Success/Failure rate
        ax2 = axes[0, 1]
        successful = [item['successful'] for item in stats['daily_stats']]
        failed = [item['failed'] for item in stats['daily_stats']]
        
        ax2.bar(dates, successful, label='Successful', color='green', alpha=0.7)
        ax2.bar(dates, failed, bottom=successful, label='Failed', color='red', alpha=0.7)
        ax2.set_title('Success/Failure Breakdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Category distribution
        ax3 = axes[1, 0]
        categories = list(stats['categories'].keys())
        counts = list(stats['categories'].values())
        
        ax3.barh(categories, counts, color='coral')
        ax3.set_title('Category Distribution (7 Days)')
        ax3.set_xlabel('Number of Images')
        
        # 4. Overall statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        SUMMARY STATISTICS (Last 7 Days)
        {'='*40}
        
        Total Runs: {stats['total_runs']}
        Total Processed: {stats['total_processed']}
        Successful: {stats['total_successful']}
        Failed: {stats['total_failed']}
        Success Rate: {stats['success_rate']:.1f}%
        
        Most Common Category:
        {max(stats['categories'], key=stats['categories'].get) if stats['categories'] else 'N/A'}
        ({max(stats['categories'].values()) if stats['categories'] else 0} images)
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_file}")
        
        return output_file
    
    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n" + "="*70)
        print("FASHION CLASSIFIER - MONITORING REPORT")
        print("="*70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # API Health
        print("API HEALTH:")
        api_health = self.check_api_health()
        status_symbol = "✓" if api_health['status'] == 'healthy' else "✗"
        print(f"  {status_symbol} Status: {api_health['status'].upper()}")
        print(f"     Message: {api_health['message']}")
        print()
        
        # Batch Health
        print("BATCH PROCESSING HEALTH:")
        batch_health = self.check_batch_health()
        status_symbol = "✓" if batch_health['status'] == 'healthy' else "⚠" if batch_health['status'] == 'warning' else "✗"
        print(f"  {status_symbol} Status: {batch_health['status'].upper()}")
        print(f"     Message: {batch_health['message']}")
        print()
        
        # Statistics
        print("BATCH STATISTICS (Last 7 Days):")
        stats = self.get_batch_statistics()
        if stats:
            print(f"  Total Runs: {stats['total_runs']}")
            print(f"  Images Processed: {stats['total_processed']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Failed: {stats['total_failed']}")
            print()
            
            if stats['categories']:
                print("  Top Categories:")
                sorted_cats = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]
                for cat, count in sorted_cats:
                    print(f"    - {cat}: {count}")
        else:
            print("  No statistics available")
        print()
        
        # Disk Usage
        print("DISK USAGE:")
        disk_usage = self.check_disk_space()
        for path, usage in disk_usage.items():
            print(f"  {path}: {usage['size_human']}")
        print()
        
        # Recent Errors
        print("RECENT ERRORS:")
        errors = self.check_recent_errors()
        if errors:
            for error in errors[-5:]:  # Last 5 errors
                print(f"  ✗ [{error['source']}] {error.get('file', '')} - {error['message']}")
        else:
            print("  ✓ No recent errors detected")
        
        print()
        print("="*70)
        
        # Save report to file
        report_file = self.output_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            # Redirect print to file (simplified version)
            f.write(f"Fashion Classifier - Monitoring Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"API Health: {api_health['status']}\n")
            f.write(f"Batch Health: {batch_health['status']}\n")
            if stats:
                f.write(f"\nStatistics:\n")
                f.write(f"  Total processed: {stats['total_processed']}\n")
                f.write(f"  Success rate: {stats['success_rate']:.1f}%\n")
        
        print(f"Report saved: {report_file}")
        
        return report_file

def main():
    dashboard = MonitoringDashboard()
    
    # Generate text report
    dashboard.generate_report()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    dashboard.generate_visualizations()
    
    print("\n✓ Monitoring dashboard complete!")

if __name__ == "__main__":
    main()