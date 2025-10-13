"""
Batch Processing Report Generator
Generates comprehensive PDF/HTML reports from batch processing results
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from jinja2 import Template

# Configuration
RESULTS_DIR = Path('results/batch_results')
REPORTS_DIR = Path('results/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")

class BatchReportGenerator:
    def __init__(self, days=7):
        self.results_dir = RESULTS_DIR
        self.reports_dir = REPORTS_DIR
        self.days = days
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_batch_data(self):
        """Load batch processing data"""
        csv_files = sorted(self.results_dir.glob('batch_results_*.csv'))
        summary_files = sorted(self.results_dir.glob('summary_*.json'))
        
        if not csv_files:
            print("No batch results found")
            return None, None
        
        # Load recent CSVs
        cutoff_date = datetime.now() - timedelta(days=self.days)
        recent_csvs = []
        
        for csv_file in csv_files:
            file_date = datetime.fromtimestamp(csv_file.stat().st_mtime)
            if file_date >= cutoff_date:
                recent_csvs.append(csv_file)
        
        if not recent_csvs:
            print(f"No batch results in last {self.days} days")
            return None, None
        
        # Combine all CSVs
        dfs = []
        for csv_file in recent_csvs:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if not dfs:
            return None, None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Load summaries
        summaries = []
        for summary_file in summary_files[-self.days:]:
            try:
                with open(summary_file, 'r') as f:
                    summaries.append(json.load(f))
            except Exception as e:
                print(f"Error loading {summary_file}: {e}")
        
        return combined_df, summaries
    
    def generate_statistics(self, df, summaries):
        """Generate statistics from data"""
        stats = {
            'period_days': self.days,
            'total_images': len(df),
            'successful': len(df[df['status'] == 'success']),
            'failed': len(df[df['status'] != 'success']),
            'success_rate': 0,
            'avg_confidence': 0,
            'total_runs': len(summaries),
            'categories': {},
            'daily_stats': []
        }
        
        if stats['total_images'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['total_images']) * 100
        
        # Average confidence for successful predictions
        successful_df = df[df['status'] == 'success']
        if len(successful_df) > 0:
            stats['avg_confidence'] = successful_df['confidence'].mean()
        
        # Category distribution
        if 'category' in df.columns:
            stats['categories'] = df['category'].value_counts().to_dict()
        
        # Daily statistics from summaries
        for summary in summaries:
            stats['daily_stats'].append({
                'date': summary.get('timestamp', ''),
                'processed': summary.get('total_processed', 0),
                'successful': summary.get('successful', 0),
                'failed': summary.get('failed', 0)
            })
        
        return stats
    
    def create_visualizations(self, df, stats):
        """Create visualization charts"""
        charts = {}
        
        # 1. Category Distribution Pie Chart
        if stats['categories']:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            categories = list(stats['categories'].keys())
            counts = list(stats['categories'].values())
            
            colors = plt.cm.Set3(range(len(categories)))
            wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            ax.set_title(f'Category Distribution\n({stats["total_images"]} images)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            chart_file = self.reports_dir / f'category_pie_{self.timestamp}.png'
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            charts['category_pie'] = chart_file
            plt.close()
        
        # 2. Daily Processing Volume
        if stats['daily_stats']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = [s['date'][:8] for s in stats['daily_stats']]
            processed = [s['processed'] for s in stats['daily_stats']]
            
            ax.bar(dates, processed, color='steelblue', edgecolor='navy', linewidth=1.5)
            ax.set_title('Daily Processing Volume', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Images Processed', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(processed):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            chart_file = self.reports_dir / f'daily_volume_{self.timestamp}.png'
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            charts['daily_volume'] = chart_file
            plt.close()
        
        # 3. Success/Failure Breakdown
        if stats['daily_stats']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = [s['date'][:8] for s in stats['daily_stats']]
            successful = [s['successful'] for s in stats['daily_stats']]
            failed = [s['failed'] for s in stats['daily_stats']]
            
            x = range(len(dates))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], successful, width, label='Successful', 
                   color='green', alpha=0.7)
            ax.bar([i + width/2 for i in x], failed, width, label='Failed', 
                   color='red', alpha=0.7)
            
            ax.set_title('Success vs Failure', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            chart_file = self.reports_dir / f'success_failure_{self.timestamp}.png'
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            charts['success_failure'] = chart_file
            plt.close()
        
        # 4. Confidence Distribution
        successful_df = df[df['status'] == 'success']
        if len(successful_df) > 0 and 'confidence' in successful_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(successful_df['confidence'], bins=20, color='skyblue', 
                   edgecolor='black', alpha=0.7)
            ax.axvline(stats['avg_confidence'], color='red', linestyle='--', 
                      linewidth=2, label=f'Average: {stats["avg_confidence"]:.2%}')
            
            ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Confidence Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            
            plt.tight_layout()
            chart_file = self.reports_dir / f'confidence_dist_{self.timestamp}.png'
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            charts['confidence_dist'] = chart_file
            plt.close()
        
        return charts
    
    def generate_html_report(self, stats, charts):
        """Generate HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Processing Report - {{ report_date }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .stat-card.warning {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background: white;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Fashion Product Batch Processing Report</h1>
        <p><strong>Report Period:</strong> Last {{ stats.period_days }} days</p>
        <p><strong>Generated:</strong> {{ report_date }}</p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Images</div>
                <div class="stat-value">{{ stats.total_images }}</div>
            </div>
            <div class="stat-card success">
                <div class="stat-label">Successful</div>
                <div class="stat-value">{{ stats.successful }}</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">Failed</div>
                <div class="stat-value">{{ stats.failed }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{{ "%.1f"|format(stats.success_rate) }}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{{ "%.1f"|format(stats.avg_confidence * 100) }}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Runs</div>
                <div class="stat-value">{{ stats.total_runs }}</div>
            </div>
        </div>
        
        <h2>📈 Visualizations</h2>
        
        {% if charts.category_pie %}
        <div class="chart">
            <h3>Category Distribution</h3>
            <img src="{{ charts.category_pie.name }}" alt="Category Distribution">
        </div>
        {% endif %}
        
        {% if charts.daily_volume %}
        <div class="chart">
            <h3>Daily Processing Volume</h3>
            <img src="{{ charts.daily_volume.name }}" alt="Daily Volume">
        </div>
        {% endif %}
        
        {% if charts.success_failure %}
        <div class="chart">
            <h3>Success vs Failure</h3>
            <img src="{{ charts.success_failure.name }}" alt="Success vs Failure">
        </div>
        {% endif %}
        
        {% if charts.confidence_dist %}
        <div class="chart">
            <h3>Confidence Distribution</h3>
            <img src="{{ charts.confidence_dist.name }}" alt="Confidence Distribution">
        </div>
        {% endif %}
        
        <h2>📋 Category Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {% for category, count in stats.categories.items() %}
                <tr>
                    <td>{{ category }}</td>
                    <td>{{ count }}</td>
                    <td>{{ "%.1f"|format((count / stats.total_images) * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Fashion Product Classification System | Generated by Batch Report Generator</p>
            <p>GitHub: <a href="https://github.com/alexandraetnaer-max/image-classification-project">image-classification-project</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        
        html_content = template.render(
            stats=stats,
            charts=charts,
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        output_file = self.reports_dir / f'batch_report_{self.timestamp}.html'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML Report: {output_file}")
        return output_file
    
    def generate_text_report(self, stats):
        """Generate simple text report"""
        output_file = self.reports_dir / f'batch_report_{self.timestamp}.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BATCH PROCESSING REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: Last {stats['period_days']} days\n")
            f.write("\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Images Processed: {stats['total_images']}\n")
            f.write(f"  Successful: {stats['successful']}\n")
            f.write(f"  Failed: {stats['failed']}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.1f}%\n")
            f.write(f"  Average Confidence: {stats['avg_confidence']:.2%}\n")
            f.write(f"  Total Runs: {stats['total_runs']}\n")
            f.write("\n")
            
            f.write("CATEGORY BREAKDOWN:\n")
            f.write("-" * 70 + "\n")
            for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_images']) * 100
                f.write(f"  {category:20s}: {count:5d} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            f.write("DAILY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            for day_stat in stats['daily_stats']:
                f.write(f"  {day_stat['date']}: Processed={day_stat['processed']}, "
                       f"Success={day_stat['successful']}, Failed={day_stat['failed']}\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Text Report: {output_file}")
        return output_file
    
    def generate_report(self):
        """Generate complete report"""
        print("=" * 70)
        print("BATCH PROCESSING REPORT GENERATOR")
        print("=" * 70)
        print()
        
        # Load data
        print(f"Loading batch data (last {self.days} days)...")
        df, summaries = self.load_batch_data()
        
        if df is None:
            print("No data available for report generation")
            return
        
        print(f"✓ Loaded {len(df)} records from {len(summaries)} batch runs")
        
        # Generate statistics
        print("\nGenerating statistics...")
        stats = self.generate_statistics(df, summaries)
        print(f"✓ Statistics computed")
        
        # Create visualizations
        print("\nCreating visualizations...")
        charts = self.create_visualizations(df, stats)
        print(f"✓ Generated {len(charts)} charts")
        
        # Generate reports
        print("\nGenerating reports...")
        html_file = self.generate_html_report(stats, charts)
        text_file = self.generate_text_report(stats)
        
        print()
        print("=" * 70)
        print("✓ REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nOpen report: {html_file}")
        print(f"Text version: {text_file}")
        
        return html_file, text_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate batch processing report')
    parser.add_argument('--days', type=int, default=7, help='Number of days to include (default: 7)')
    
    args = parser.parse_args()
    
    generator = BatchReportGenerator(days=args.days)
    generator.generate_report()

if __name__ == "__main__":
    main()