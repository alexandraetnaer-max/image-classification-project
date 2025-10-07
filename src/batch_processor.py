"""
Batch processing system for automatic image classification
Processes all images in incoming folder and organizes them by category
"""
import os
import shutil
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import time

# Configuration
INCOMING_DIR = os.path.join('data', 'incoming')
PROCESSED_DIR = os.path.join('data', 'processed_batches')
RESULTS_DIR = os.path.join('results', 'batch_results')
LOGS_DIR = 'logs'
API_URL = "http://localhost:5000"

# Create directories
os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class BatchProcessor:
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(LOGS_DIR, f'batch_{self.timestamp}.log')
        
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                self.log("✓ API is healthy and ready")
                return True
            else:
                self.log(f"✗ API returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"✗ Cannot connect to API: {e}")
            self.log(f"  Make sure API is running: python api/app.py")
            return False
    
    def get_incoming_images(self):
        """Get list of images to process"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif',  '.webp', '.tiff', '.tif')
        images = []
        
        if not os.path.exists(INCOMING_DIR):
            self.log(f"✗ Incoming directory not found: {INCOMING_DIR}")
            return images
        
        for filename in os.listdir(INCOMING_DIR):
            if filename.lower().endswith(image_extensions):
                images.append(filename)
        
        return images
    
    def process_image(self, filename):
        """Process single image through API"""
        filepath = os.path.join(INCOMING_DIR, filename)
        
        try:
            # Send to API
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'filename': filename,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'filename': filename,
                    'category': None,
                    'confidence': 0,
                    'status': 'error',
                    'error': f"API returned {response.status_code}",
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'filename': filename,
                'category': None,
                'confidence': 0,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def organize_image(self, filename, category):
        """Move processed image to category folder"""
        source = os.path.join(INCOMING_DIR, filename)
        
        # Create category folder
        category_dir = os.path.join(PROCESSED_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Move file
        destination = os.path.join(category_dir, filename)
        
        # Handle duplicate filenames
        if os.path.exists(destination):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(destination):
                new_filename = f"{base}_{counter}{ext}"
                destination = os.path.join(category_dir, new_filename)
                counter += 1
        
        shutil.move(source, destination)
        return destination
    
    def save_results(self):
        """Save processing results to CSV"""
        if not self.results:
            self.log("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, f'batch_results_{self.timestamp}.csv')
        df.to_csv(csv_path, index=False)
        self.log(f"✓ Results saved to: {csv_path}")
        
        # Save summary
        summary = {
            'timestamp': self.timestamp,
            'total_processed': len(self.results),
            'successful': len([r for r in self.results if r['status'] == 'success']),
            'failed': len([r for r in self.results if r['status'] == 'error']),
            'categories': df['category'].value_counts().to_dict() if 'category' in df else {}
        }
        
        summary_path = os.path.join(RESULTS_DIR, f'summary_{self.timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"✓ Summary saved to: {summary_path}")
        
        return summary
    
    def process_batch(self):
        """Main batch processing function"""
        self.log("=" * 60)
        self.log("BATCH PROCESSING STARTED")
        self.log("=" * 60)
        
        # Check API
        if not self.check_api_health():
            self.log("✗ Batch processing aborted - API not available")
            return
        
        # Get images
        images = self.get_incoming_images()
        
        if not images:
            self.log(f"No images found in {INCOMING_DIR}")
            self.log("Place images in this folder for processing")
            return
        
        self.log(f"Found {len(images)} images to process")
        
        # Process each image
        for idx, filename in enumerate(images, 1):
            self.log(f"\nProcessing [{idx}/{len(images)}]: {filename}")
            
            result = self.process_image(filename)
            self.results.append(result)
            
            if result['status'] == 'success':
                self.log(f"  ✓ Category: {result['category']} (confidence: {result['confidence']*100:.1f}%)")
                
                # Organize image
                new_path = self.organize_image(filename, result['category'])
                self.log(f"  ✓ Moved to: {new_path}")
            else:
                self.log(f"  ✗ Error: {result.get('error', 'Unknown error')}")
            
            # Small delay to avoid overwhelming API
            time.sleep(0.1)
        
        # Save results
        self.log("\n" + "=" * 60)
        self.log("SAVING RESULTS")
        self.log("=" * 60)
        
        summary = self.save_results()
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("BATCH PROCESSING COMPLETE")
        self.log("=" * 60)
        self.log(f"Total processed: {summary['total_processed']}")
        self.log(f"Successful: {summary['successful']}")
        self.log(f"Failed: {summary['failed']}")
        
        if summary['categories']:
            self.log("\nCategory distribution:")
            for category, count in summary['categories'].items():
                if category:  # Skip None values
                    self.log(f"  - {category}: {count}")
        
        self.log(f"\nLog file: {self.log_file}")

if __name__ == "__main__":
    processor = BatchProcessor()
    processor.process_batch()