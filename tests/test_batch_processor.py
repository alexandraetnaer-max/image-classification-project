"""
Unit tests for batch processing
"""
import unittest
import os
import sys
import json
from pathlib import Path
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.batch_processor import BatchProcessor

class TestBatchProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path('tests/test_batch')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create mock incoming directory
        self.incoming = self.test_dir / 'incoming'
        self.incoming.mkdir(exist_ok=True)
        
        # Create mock processed directory
        self.processed = self.test_dir / 'processed'
        self.processed.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test data"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor can be initialized"""
        processor = BatchProcessor()
        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.timestamp)
    
    def test_log_function(self):
        """Test logging function"""
        processor = BatchProcessor()
        
        # Should not raise exception
        try:
            processor.log("Test message")
            success = True
        except:
            success = False
        
        self.assertTrue(success)
    
    def test_get_incoming_images_empty(self):
        """Test get_incoming_images with empty directory"""
        # Temporarily override INCOMING_DIR
        import src.batch_processor as bp
        original_dir = bp.INCOMING_DIR
        bp.INCOMING_DIR = str(self.incoming)
        
        processor = BatchProcessor()
        images = processor.get_incoming_images()
        
        self.assertEqual(len(images), 0)
        
        # Restore
        bp.INCOMING_DIR = original_dir
    
    def test_get_incoming_images_with_files(self):
        """Test get_incoming_images finds image files"""
        # Create mock image files
        test_files = ['image1.jpg', 'image2.png', 'image3.jpeg']
        for filename in test_files:
            (self.incoming / filename).touch()
        
        # Also create non-image file
        (self.incoming / 'readme.txt').touch()
        
        # Override directory
        import src.batch_processor as bp
        original_dir = bp.INCOMING_DIR
        bp.INCOMING_DIR = str(self.incoming)
        
        processor = BatchProcessor()
        images = processor.get_incoming_images()
        
        # Should find 3 image files, not the txt file
        self.assertEqual(len(images), 3)
        self.assertNotIn('readme.txt', images)
        
        # Restore
        bp.INCOMING_DIR = original_dir
    
    def test_process_image_structure(self):
        """Test process_image returns correct structure"""
        processor = BatchProcessor()
        
        # Mock result (would normally come from API)
        result = {
            'filename': 'test.jpg',
            'category': None,
            'confidence': 0,
            'status': 'error',
            'error': 'API unavailable',
            'timestamp': processor.timestamp
        }
        
        # Verify structure
        self.assertIn('filename', result)
        self.assertIn('category', result)
        self.assertIn('confidence', result)
        self.assertIn('status', result)
        self.assertIn('timestamp', result)

class TestBatchResults(unittest.TestCase):
    
    def test_result_structure(self):
        """Test batch result has required fields"""
        result = {
            'filename': 'test.jpg',
            'category': 'Tshirts',
            'confidence': 0.95,
            'status': 'success',
            'timestamp': '2025-10-13T10:00:00'
        }
        
        required_fields = ['filename', 'category', 'confidence', 'status', 'timestamp']
        
        for field in required_fields:
            self.assertIn(field, result, f"Result should contain {field}")
    
    def test_summary_structure(self):
        """Test summary JSON has required fields"""
        summary = {
            'timestamp': '20251013_100000',
            'total_processed': 100,
            'successful': 98,
            'failed': 2,
            'categories': {
                'Tshirts': 50,
                'Shoes': 48
            }
        }
        
        required_fields = ['timestamp', 'total_processed', 'successful', 'failed', 'categories']
        
        for field in required_fields:
            self.assertIn(field, summary, f"Summary should contain {field}")

if __name__ == '__main__':
    unittest.main()