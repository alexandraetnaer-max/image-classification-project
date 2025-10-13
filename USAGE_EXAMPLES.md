markdown# Usage Examples and Scenarios

Complete guide with practical examples for using the Fashion Classification System.

---

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [API Integration Examples](#api-integration-examples)
3. [Batch Processing Scenarios](#batch-processing-scenarios)
4. [Business Use Cases](#business-use-cases)
5. [E-commerce Integration](#e-commerce-integration)
6. [Returns Department Workflow](#returns-department-workflow)

---

## Quick Start Examples

### Example 1: Classify Single Product Image

**Scenario:** You have a product photo and want to identify its category.

**Step 1: Prepare your image**
```bash
# Make sure you have an image file
ls product_photo.jpg
Step 2: Send to API
bashcurl -X POST \
  -F "file=@product_photo.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict
Response:
json{
  "category": "Tshirts",
  "confidence": 0.9534,
  "timestamp": "2025-10-13T15:30:00"
}
Interpretation:

Product is classified as "Tshirts"
Model is 95.34% confident
High confidence = reliable prediction


Example 2: Classify Multiple Products
Scenario: You have a folder with 10 product images to classify.
Step 1: Navigate to folder
bashcd /path/to/product/images
ls
# product_1.jpg  product_2.jpg  product_3.jpg  ...
Step 2: Batch classify
bashcurl -X POST \
  -F "files=@product_1.jpg" \
  -F "files=@product_2.jpg" \
  -F "files=@product_3.jpg" \
  https://fashion-classifier-api-728466800559.europe-west1.run.app/predict_batch
Response:
json{
  "predictions": [
    {"filename": "product_1.jpg", "category": "Tshirts", "confidence": 0.95},
    {"filename": "product_2.jpg", "category": "Shoes", "confidence": 0.89},
    {"filename": "product_3.jpg", "category": "Handbags", "confidence": 0.92}
  ],
  "total": 3
}

Example 3: Automated Nightly Classification
Scenario: Process all new product images every night at 2 AM.
Step 1: Place images in incoming folder
bash# Throughout the day, add images here
cp new_products/*.jpg data/incoming/
Step 2: Set up automation
Windows (Task Scheduler):

Open Task Scheduler
Create task: "Nightly Product Classification"
Trigger: Daily at 2:00 AM
Action: run_batch_processing.bat

Linux/Mac (Cron):
bash# Edit crontab
crontab -e

# Add this line
0 2 * * * /path/to/run_batch_processing.sh
Step 3: Results
Next morning, check:
bash# Organized by category
ls data/processed_batches/
# Tshirts/  Shoes/  Handbags/  ...

# Results CSV
cat results/batch_results/batch_results_*.csv
# filename,category,confidence,status,timestamp
# product_1.jpg,Tshirts,0.95,success,2025-10-13T02:01:23

API Integration Examples
Python Integration
Example 1: Single Product Classification
pythonimport requests
from pathlib import Path

def classify_product(image_path):
    """
    Classify a single product image
    
    Args:
        image_path: Path to product image
        
    Returns:
        dict: Classification result
    """
    api_url = "https://fashion-classifier-api-728466800559.europe-west1.run.app/predict"
    
    # Open and send image
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Product: {image_path}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
result = classify_product('product_photo.jpg')
Example 2: Batch Classification with Progress
pythonimport requests
from pathlib import Path
from tqdm import tqdm

def classify_products_batch(image_folder):
    """
    Classify all products in a folder with progress bar
    
    Args:
        image_folder: Folder containing product images
        
    Returns:
        list: Classification results
    """
    api_url = "https://fashion-classifier-api-728466800559.europe-west1.run.app/predict"
    
    # Get all images
    image_paths = list(Path(image_folder).glob('*.jpg'))
    image_paths += list(Path(image_folder).glob('*.png'))
    
    results = []
    
    # Process with progress bar
    for image_path in tqdm(image_paths, desc="Classifying"):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                result['filename'] = image_path.name
                results.append(result)
    
    return results

# Usage
results = classify_products_batch('product_images/')

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('classification_results.csv', index=False)
print(f"Classified {len(results)} products")
Example 3: Real-time Classification with Retry Logic
pythonimport requests
import time
from typing import Optional

def classify_with_retry(
    image_path: str,
    max_retries: int = 3,
    timeout: int = 30
) -> Optional[dict]:
    """
    Classify image with automatic retry on failure
    
    Args:
        image_path: Path to image
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Classification result or None
    """
    api_url = "https://fashion-classifier-api-728466800559.europe-west1.run.app/predict"
    
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    api_url,
                    files=files,
                    timeout=timeout
                )
            
            if response.status_code == 200:
                return response.json()
            
            # If server error, retry
            if response.status_code >= 500:
                print(f"Server error, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            # Client error, don't retry
            print(f"Client error: {response.status_code}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"Timeout, retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    print(f"Failed after {max_retries} attempts")
    return None

# Usage
result = classify_with_retry('product.jpg')

JavaScript/Node.js Integration
javascriptconst axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

/**
 * Classify product image
 * @param {string} imagePath - Path to image file
 * @returns {Promise<object>} Classification result
 */
async function classifyProduct(imagePath) {
    const apiUrl = 'https://fashion-classifier-api-728466800559.europe-west1.run.app/predict';
    
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    
    try {
        const response = await axios.post(apiUrl, form, {
            headers: form.getHeaders(),
            timeout: 30000
        });
        
        console.log(`Category: ${response.data.category}`);
        console.log(`Confidence: ${(response.data.confidence * 100).toFixed(2)}%`);
        
        return response.data;
    } catch (error) {
        console.error('Error:', error.message);
        return null;
    }
}

// Usage
classifyProduct('product.jpg')
    .then(result => {
        if (result) {
            console.log('Classification successful');
        }
    });

Batch Processing Scenarios
Scenario 1: Daily Product Upload Processing
Context: E-commerce site receives 500-1000 new product photos daily.
Workflow:

Morning (9 AM): Photographers upload new products

bash# Images uploaded to server
/uploads/2025-10-13/
├── product_001.jpg
├── product_002.jpg
└── ...

Throughout Day: Images accumulate in incoming folder

bash# Automated sync
rsync -av /uploads/$(date +%Y-%m-%d)/ data/incoming/

Night (2 AM): Batch processing runs automatically

bash# Task Scheduler / Cron executes
python src/batch_processor.py

Morning (8 AM): Results ready for review

bash# Check results
cat results/batch_results/batch_results_20251013_020000.csv

# Organized by category
ls data/processed_batches/
# Tshirts/     (45 images)
# Shoes/       (32 images)
# Handbags/    (28 images)

Review & Action:

pythonimport pandas as pd

# Load results
df = pd.read_csv('results/batch_results/batch_results_latest.csv')

# Check low confidence predictions
low_confidence = df[df['confidence'] < 0.7]
print(f"Review {len(low_confidence)} products with low confidence")

# Generate report
summary = df['category'].value_counts()
print("Daily Product Breakdown:")
print(summary)

Scenario 2: Inventory Categorization
Context: Warehouse has 10,000 uncategorized product photos.
Workflow:
Week 1: Setup
bash# Split into batches (1000 images each)
mkdir -p batches/batch_{01..10}
split -n 10 -d product_list.txt batches/

# Create processing script
cat > process_inventory.sh << 'EOF'
#!/bin/bash
for batch in batches/batch_*; do
    echo "Processing $batch..."
    cp $batch/* data/incoming/
    python src/batch_processor.py
    sleep 60  # Wait between batches
done
EOF

chmod +x process_inventory.sh
Week 2-3: Processing
bash# Run batch processing
./process_inventory.sh > processing.log 2>&1

# Monitor progress
tail -f processing.log
Week 4: Analysis
pythonimport pandas as pd
import glob

# Combine all results
all_results = []
for csv_file in glob.glob('results/batch_results/*.csv'):
    df = pd.read_csv(csv_file)
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

# Statistics
print(f"Total processed: {len(combined)}")
print(f"Success rate: {(combined['status']=='success').sum()/len(combined)*100:.1f}%")
print("\nCategory Distribution:")
print(combined['category'].value_counts())

# Export for inventory system
combined.to_csv('inventory_categories.csv', index=False)

Business Use Cases
Use Case 1: E-commerce Product Onboarding
Problem: Manual product categorization is slow and inconsistent.
Solution: Automated classification at upload time.
Implementation:
python# Product upload handler
def handle_product_upload(image_file, product_id):
    """
    Process product image on upload
    
    Args:
        image_file: Uploaded image
        product_id: Product database ID
    """
    # 1. Save image
    image_path = f'uploads/{product_id}.jpg'
    image_file.save(image_path)
    
    # 2. Classify
    result = classify_product(image_path)
    
    # 3. Update database
    if result and result['confidence'] > 0.8:
        # Auto-assign category
        db.products.update(
            {'id': product_id},
            {'category': result['category']}
        )
        print(f"Product {product_id} auto-categorized as {result['category']}")
    else:
        # Flag for manual review
        db.products.update(
            {'id': product_id},
            {'needs_review': True, 'suggested_category': result['category']}
        )
        print(f"Product {product_id} flagged for review")
    
    return result
Benefits:

80% of products auto-categorized
20% flagged for quick manual review
Saves 2-3 hours daily


Use Case 2: Quality Control
Problem: Products sometimes mislabeled or photos unclear.
Solution: Confidence scoring for quality control.
Implementation:
pythondef quality_control_check(products_folder):
    """
    Check product photo quality and correct categorization
    
    Args:
        products_folder: Folder with categorized products
        
    Returns:
        Quality report
    """
    issues = []
    
    for category in os.listdir(products_folder):
        category_path = os.path.join(products_folder, category)
        
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            
            # Classify
            result = classify_product(image_path)
            
            # Check for mismatches
            if result['category'] != category:
                issues.append({
                    'file': image_file,
                    'expected': category,
                    'predicted': result['category'],
                    'confidence': result['confidence']
                })
    
    # Generate report
    if issues:
        print(f"Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  {issue['file']}: Expected {issue['expected']}, "
                  f"predicted {issue['predicted']} ({issue['confidence']:.0%})")
    else:
        print("All products correctly categorized!")
    
    return issues

Returns Department Workflow
Complete Step-by-Step Guide for Returns Processing
Context: Fashion returns department receives 200-300 returned items daily. Items must be:

Photographed
Categorized
Quality checked
Routed to correct warehouse section


Daily Returns Processing Workflow
Morning Setup (8:00 AM)
Step 1: Prepare workspace
bash# Create today's returns folder
mkdir -p returns/$(date +%Y-%m-%d)
cd returns/$(date +%Y-%m-%d)
Step 2: Set up photo station

Light box ready
Camera/phone connected
Naming convention: return_YYYYMMDD_NNN.jpg


During Day: Photo Collection (8 AM - 4 PM)
Step 1: Receive returned item
Return ID: RET-20251013-001
Customer: John Doe
Reason: Size issue
Step 2: Photograph item
bash# Photo naming format
return_20251013_001.jpg
Step 3: Move to processing folder
bash# Copy to incoming folder for batch processing
cp *.jpg ../../data/incoming/
Progress tracking:
bash# Check how many processed today
ls -1 ../../data/incoming/ | wc -l
# 247 files

After Hours: Automated Classification (4 PM or overnight)
Option A: Manual trigger (if needed urgently)
bashcd /path/to/project
python src/batch_processor.py
Option B: Scheduled (automatic at 5 PM)
Task runs automatically via Task Scheduler
Processing takes: ~5 minutes for 300 images

Next Morning: Results Review (8:00 AM)
Step 1: Open results
bash# View latest batch results
cat results/batch_results/batch_results_20251013_170000.csv
Excel format:
filename                    | category      | confidence | status
return_20251013_001.jpg    | Tshirts       | 0.95       | success
return_20251013_002.jpg    | Casual Shoes  | 0.88       | success
return_20251013_003.jpg    | Handbags      | 0.72       | success
Step 2: Review classification
pythonimport pandas as pd

# Load results
df = pd.read_csv('results/batch_results/batch_results_20251013_170000.csv')

# Priority 1: Low confidence (needs manual review)
low_conf = df[df['confidence'] < 0.70]
print(f"\n=== MANUAL REVIEW REQUIRED: {len(low_conf)} items ===")
for idx, row in low_conf.iterrows():
    print(f"{row['filename']}: {row['category']} ({row['confidence']:.0%})")

# Priority 2: High confidence (auto-process)
high_conf = df[df['confidence'] >= 0.85]
print(f"\n=== AUTO-PROCESSED: {len(high_conf)} items ===")

# Category breakdown
print("\n=== CATEGORY SUMMARY ===")
print(df['category'].value_counts())
Step 3: Physical organization
Items already organized in folders:
bashdata/processed_batches/
├── Tshirts/
│   ├── return_20251013_001.jpg  (RET-20251013-001)
│   ├── return_20251013_015.jpg
│   └── ... (87 items)
├── Casual Shoes/
│   ├── return_20251013_002.jpg
│   └── ... (45 items)
└── Handbags/
    └── ... (32 items)
Physical routing:

Print pick lists by category
Route to warehouse sections:

Tshirts → Section A, Aisle 3
Shoes → Section B, Aisle 7
Handbags → Section C, Aisle 2




Weekly Reports (Friday)
Generate weekly summary:
pythonimport pandas as pd
import glob
from datetime import datetime, timedelta

# Get this week's results
week_ago = datetime.now() - timedelta(days=7)
week_results = []

for csv_file in glob.glob('results/batch_results/*.csv'):
    # Check if file is from this week
    file_date = datetime.fromtimestamp(os.path.getmtime(csv_file))
    if file_date >= week_ago:
        df = pd.read_csv(csv_file)
        week_results.append(df)

# Combine
weekly = pd.concat(week_results, ignore_index=True)

# Generate report
report = f"""
WEEKLY RETURNS REPORT
Week ending: {datetime.now().strftime('%Y-%m-%d')}

VOLUME:
  Total Returns Processed: {len(weekly)}
  Success Rate: {(weekly['status']=='success').sum()/len(weekly)*100:.1f}%
  
CATEGORY BREAKDOWN:
{weekly['category'].value_counts().to_string()}

CONFIDENCE LEVELS:
  High (>85%): {len(weekly[weekly['confidence'] > 0.85])} ({len(weekly[weekly['confidence'] > 0.85])/len(weekly)*100:.1f}%)
  Medium (70-85%): {len(weekly[(weekly['confidence'] >= 0.70) & (weekly['confidence'] <= 0.85)])}
  Low (<70%): {len(weekly[weekly['confidence'] < 0.70])} (needs review)

PROCESSING TIME:
  Average per item: {len(weekly) / 5 / 60:.1f} seconds
  
PRODUCTIVITY IMPROVEMENT:
  Manual categorization time saved: {len(weekly) * 2 / 60:.1f} hours
  Cost savings: ${len(weekly) * 2 / 60 * 15:.2f} (at $15/hour)
"""

print(report)

# Save report
with open(f'weekly_report_{datetime.now().strftime("%Y%m%d")}.txt', 'w') as f:
    f.write(report)

Returns Department Best Practices
1. Photo Quality Guidelines:
✅ Good photos:
   - Clean background (white/gray)
   - Item centered
   - Good lighting
   - Full item visible
   - One item per photo

❌ Avoid:
   - Multiple items in one photo
   - Dark/blurry images
   - Extreme angles
   - Item partially out of frame
2. Handling Edge Cases:
pythondef handle_uncertain_classification(row):
    """
    Decision tree for uncertain classifications
    """
    confidence = row['confidence']
    category = row['category']
    
    if confidence >= 0.85:
        return "AUTO_PROCESS"
    elif 0.70 <= confidence < 0.85:
        return "QUICK_REVIEW"  # 30 second review
    else:
        return "MANUAL_REVIEW"  # Full inspection

# Apply to results
df['action'] = df.apply(handle_uncertain_classification, axis=1)

# Route accordingly
auto = df[df['action'] == 'AUTO_PROCESS']
quick = df[df['action'] == 'QUICK_REVIEW']
manual = df[df['action'] == 'MANUAL_REVIEW']

print(f"Auto-process: {len(auto)} items")
print(f"Quick review: {len(quick)} items (15 mins)")
print(f"Manual review: {len(manual)} items (30 mins)")
3. Performance Metrics:
Track these KPIs weekly:

Total returns processed
Classification accuracy (spot checks)
Time saved vs manual processing
Items needing manual review (goal: <15%)
Processing speed (items/hour)


Integration with Existing Systems
Inventory Management System Integration:
python# Example: Update inventory system
def update_inventory_system(classification_results):
    """
    Update inventory management system with classifications
    
    Args:
        classification_results: DataFrame with classifications
    """
    for idx, row in classification_results.iterrows():
        return_id = extract_return_id(row['filename'])
        
        # Update in inventory system
        inventory_api.update_return(
            return_id=return_id,
            category=row['category'],
            confidence=row['confidence'],
            processed_date=row['timestamp'],
            auto_classified=row['confidence'] > 0.85
        )
        
        # If high confidence, auto-approve for restocking
        if row['confidence'] > 0.90:
            inventory_api.mark_for_restock(return_id)

Support and Troubleshooting
Common Issues
Issue 1: API timeout
python# Solution: Increase timeout
response = requests.post(api_url, files=files, timeout=60)  # 60 seconds
Issue 2: Large batch fails
python# Solution: Process in smaller chunks
def process_large_batch(images, chunk_size=10):
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i+chunk_size]
        process_chunk(chunk)
        time.sleep(5)  # Pause between chunks
Issue 3: Low confidence predictions
python# Solution: Retake photo with better lighting/angle
# Or flag for manual review

Resources

API Documentation: api/README.md
Batch Processing Guide: BATCH_SCHEDULING.md
Monitoring Dashboard: monitoring/dashboard.py
Support: GitHub Issues