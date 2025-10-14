# Visualization and Reporting Guide

## Overview

The Fashion Classification system includes comprehensive visualization and reporting capabilities for both model training results and batch processing operations.

---

## Table of Contents

1. [Training Results Visualization](#training-results-visualization)
2. [Batch Processing Reports](#batch-processing-reports)
3. [Available Visualizations](#available-visualizations)
4. [Automated Report Generation](#automated-report-generation)

---

## Training Results Visualization

### Generate Training Visualizations
```bash
python src/visualize_results.py
```
Output Location: `results/visualizations/`

**Generated Charts:**
- Training Metrics (`training_metrics.png`)
  - Training vs Validation Accuracy
  - Training vs Validation Loss
  - Final accuracy and loss values

- Confusion Matrix (`confusion_matrix.png`, `confusion_matrix_counts.png`)
  - Normalized version (percentages)
  - Count version (absolute numbers)
  - Shows prediction accuracy per class

- Class Distribution (`class_distribution.png`)
  - Number of training images per category
  - Total dataset size
  - Helps identify class imbalance

- Per-Class Accuracy (`per_class_accuracy.png`)
  - Accuracy for each category
  - Color-coded: Green (>90%), Orange (80-90%), Red (<80%)
  - Average accuracy line

**Example Output:**  
All training visualizations show:
- Model Performance: 91.78% validation accuracy
- Class Balance: Tshirts (4,946) to Sunglasses (751)
- Training Stability: Smooth convergence curves
- Category Accuracy: 85-95% range across classes

---

## Batch Processing Reports

### Generate Batch Report
```bash
# Default: Last 7 days
python src/generate_batch_report.py

# Custom period
python src/generate_batch_report.py --days 30
```
Output Location: `results/reports/`

**Generated Files:**
- HTML Report (`batch_report_YYYYMMDD_HHMMSS.html`)
  - Interactive, styled report
  - Embedded charts
  - Responsive design
  - Ready to share

- Text Report (`batch_report_YYYYMMDD_HHMMSS.txt`)
  - Plain text summary
  - Console-friendly
  - Easy to parse

- Chart Images:
  - `category_pie_*.png` - Category distribution
  - `daily_volume_*.png` - Processing volume over time
  - `success_failure_*.png` - Success rate breakdown
  - `confidence_dist_*.png` - Confidence score distribution

**HTML Report Features**
- Summary Statistics Cards:
  - Total images processed
  - Success/failure counts
  - Success rate percentage
  - Average confidence score
  - Number of batch runs

- Interactive Visualizations:
  - Category distribution pie chart
  - Daily processing volume bar chart
  - Success vs failure comparison
  - Confidence score histogram

- Category Breakdown Table:
  - Images per category
  - Percentage of total
  - Sorted by count

**Opening Reports**

In Browser:
```bash
# Windows
start results/reports/batch_report_*.html

# Mac
open results/reports/batch_report_*.html

# Linux
xdg-open results/reports/batch_report_*.html
```
In VS Code:  
Right-click HTML file → "Open with Live Server"

---

## Available Visualizations

1. **Training Metrics**
    - Purpose: Track model learning progress
    - Shows:
        - Accuracy improvement over epochs
        - Loss reduction over epochs
        - Training vs validation comparison
        - Convergence patterns
    - Use Case: Identify overfitting, underfitting, or training issues

2. **Confusion Matrix**
    - Purpose: Understand classification errors
    - Shows:
        - Correct predictions (diagonal)
        - Common misclassifications (off-diagonal)
        - Per-class performance
    - Interpretation:
        - High diagonal values: Good classification
        - High off-diagonal: Confusion between classes
        - Example: "Shirts" confused with "Tops"

3. **Class Distribution**
    - Purpose: Assess dataset balance
    - Shows:
        - Number of images per category
        - Relative sizes of classes
        - Total dataset size
    - Use Case:
        - Identify class imbalance
        - Plan data augmentation
        - Understand model bias

4. **Per-Class Accuracy**
    - Purpose: Evaluate category-specific performance
    - Shows:
        - Accuracy for each category
        - Color-coded performance levels
        - Average accuracy benchmark
    - Use Case:
        - Identify weak categories
        - Focus improvement efforts
        - Validate model fairness

5. **Category Distribution (Batch)**
    - Purpose: Analyze batch processing results
    - Shows:
        - Distribution of processed images by category
        - Real-world usage patterns
        - Popular categories
    - Use Case:
        - Understand user behavior
        - Plan resource allocation
        - Validate production performance

6. **Processing Volume**
    - Purpose: Track batch processing activity
    - Shows:
        - Daily image processing count
        - Processing trends over time
        - Peak usage periods
    - Use Case:
        - Capacity planning
        - Identify anomalies
        - Monitor system health

7. **Success Rate**
    - Purpose: Monitor prediction reliability
    - Shows:
        - Successful vs failed predictions
        - Daily success rate trends
        - Error patterns
    - Use Case:
        - Quality assurance
        - Identify issues
        - Track improvements

8. **Confidence Distribution**
    - Purpose: Assess prediction certainty
    - Shows:
        - Distribution of confidence scores
        - Average confidence level
        - Low-confidence predictions
    - Use Case:
        - Quality filtering
        - Threshold optimization
        - Manual review targeting

---

## Automated Report Generation

**Daily Reports**

Add to cron/Task Scheduler:
```bash
# Daily at 9 AM
0 9 * * * cd /path/to/project && python src/generate_batch_report.py
```

**Weekly Reports**
```bash
# Every Monday at 9 AM
0 9 * * 1 cd /path/to/project && python src/generate_batch_report.py --days 7
```

**Monthly Reports**
```bash
# First day of month at 9 AM
0 9 1 * * cd /path/to/project && python src/generate_batch_report.py --days 30
```

**Email Reports**

Combine with email:
```bash
# Generate and email
python src/generate_batch_report.py && \
  echo "See attached report" | mail -s "Monthly Batch Report" \
  -a results/reports/batch_report_*.html admin@example.com
```

**Customization**

Modify Chart Styles  
Edit `src/visualize_results.py` or `src/generate_batch_report.py`:
```python
# Change colors
colors = plt.cm.Set3(range(len(categories)))  # Try: viridis, plasma, coolwarm

# Change figure size
plt.figure(figsize=(16, 10))  # Default: (12, 8)

# Change DPI (resolution)
plt.savefig(output_file, dpi=300)  # Default: 150
```

Add Custom Metrics  
In `src/generate_batch_report.py`, add to `generate_statistics()`:
```python
# Example: Calculate average processing time
stats['avg_processing_time'] = df['processing_time'].mean()
```
Then display in HTML template.

---

## Best Practices

- Regular Generation: Generate reports weekly for trends
- Version Control: Keep historical reports for comparison
- Share Reports: HTML reports are perfect for stakeholders
- Monitor Trends: Watch for declining accuracy or confidence
- Archive Old Reports: Clean up reports older than 90 days

---

## Troubleshooting

**No Training History Found**
- Problem: visualize_results.py finds no history
- Solution:
    - Check models/ for training_history_*.json
    - Re-run training: python src/train_model_fixed.py

**No Batch Results**
- Problem: Report generator finds no data
- Solution:
    - Check results/batch_results/ for CSV files
    - Run batch processing: python src/batch_processor.py
    - Verify date range with --days parameter

**Charts Not Displaying in HTML**
- Problem: HTML shows broken image links
- Solution:
    - Ensure chart PNGs are in same directory as HTML
    - Use relative paths in HTML
    - Check file permissions

**Low Image Quality**
- Problem: Charts look pixelated
- Solution:
    ```python
    # Increase DPI in visualization scripts
    plt.savefig(output_file, dpi=300)  # Higher = better quality
    ```

---

## Integration Examples

**Dashboard Integration**

Use generated charts in dashboards:
```python
# Copy latest charts to dashboard directory
import shutil

src = "results/visualizations/"
dst = "/var/www/dashboard/images/"

shutil.copy(f"{src}training_metrics.png", dst)
shutil.copy(f"{src}confusion_matrix.png", dst)
```

**API Endpoint**

Serve charts via API:
```python
from flask import send_file

@app.route('/charts/<chart_name>')
def get_chart(chart_name):
    return send_file(f'results/visualizations/{chart_name}')
```

---

## Sample Reports

Check `results/reports/` and `results/visualizations/` for examples of:

- Training performance visualization
- Batch processing summaries
- Confidence analysis
- Category distributions

---

## Support

- GitHub: Report visualization issues
- Documentation: See README.md for general setup
- Examples: Check results/ directory