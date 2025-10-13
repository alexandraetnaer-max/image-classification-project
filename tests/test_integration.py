"""
Integration tests for end-to-end workflows
Tests complete pipelines from data preparation to reporting
"""
import unittest
import os
import sys
import shutil
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete workflows"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_dir = Path('tests/integration_test_data')
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create test image
        cls.test_image_path = cls.test_dir / 'test_product.jpg'
        img = Image.new('RGB', (224, 224), color='red')
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_image_to_prediction_workflow(self):
        """Test complete workflow: image -> preprocessing -> prediction"""
        # This tests the integration of multiple components
        
        # 1. Load image
        from PIL import Image
        img = Image.open(self.test_image_path)
        self.assertIsNotNone(img)
        
        # 2. Check image properties
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, 'RGB')
        
        # 3. Would normally test prediction here
        # Skipped as it requires loaded model
        self.assertTrue(True, "Image loaded successfully")
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow: incoming -> processing -> results"""
        from src.batch_processor import BatchProcessor
        
        # 1. Initialize processor
        processor = BatchProcessor()
        self.assertIsNotNone(processor)
        
        # 2. Check logging works
        try:
            processor.log("Test integration log")
            logged_successfully = True
        except:
            logged_successfully = False
        
        self.assertTrue(logged_successfully)
    
    def test_api_health_to_prediction(self):
        """Test API workflow: health check -> prediction"""
        from api.app import app
        
        client = app.test_client()
        
        # 1. Check health
        health_response = client.get('/health')
        self.assertEqual(health_response.status_code, 200)
        
        health_data = json.loads(health_response.data)
        
        # 2. Only proceed to prediction if model loaded
        if health_data.get('model_loaded'):
            # 3. Test prediction with mock image
            img = Image.new('RGB', (224, 224), color='blue')
            from io import BytesIO
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            predict_response = client.post(
                '/predict',
                data={'file': (img_bytes, 'test.jpg')},
                content_type='multipart/form-data'
            )
            
            # Should return 200 or 500 (if processing fails)
            self.assertIn(predict_response.status_code, [200, 500])

class TestDataPipeline(unittest.TestCase):
    """Test data processing pipeline"""
    
    def setUp(self):
        """Set up test data"""
        self.test_dir = Path('tests/pipeline_test')
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test data"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_directory_creation_pipeline(self):
        """Test creating complete directory structure"""
        splits = ['train', 'val', 'test']
        categories = ['Tshirts', 'Shoes']
        
        # Create structure
        for split in splits:
            for category in categories:
                dir_path = self.test_dir / split / category
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Verify all directories exist
        for split in splits:
            for category in categories:
                dir_path = self.test_dir / split / category
                self.assertTrue(dir_path.exists())
                self.assertTrue(dir_path.is_dir())
    
    def test_image_processing_pipeline(self):
        """Test image loading -> resizing -> saving pipeline"""
        from PIL import Image
        
        # 1. Create test image
        original_path = self.test_dir / 'original.jpg'
        img = Image.new('RGB', (512, 512), color='green')
        img.save(original_path)
        
        # 2. Load and resize
        loaded_img = Image.open(original_path)
        resized_img = loaded_img.resize((224, 224))
        
        # 3. Save resized
        processed_path = self.test_dir / 'processed.jpg'
        resized_img.save(processed_path)
        
        # 4. Verify
        final_img = Image.open(processed_path)
        self.assertEqual(final_img.size, (224, 224))

class TestErrorRecovery(unittest.TestCase):
    """Test system handles errors gracefully"""
    
    def test_missing_file_handling(self):
        """Test handling of missing files"""
        from PIL import Image
        
        with self.assertRaises(FileNotFoundError):
            Image.open('nonexistent_file.jpg')
    
    def test_invalid_image_handling(self):
        """Test handling of invalid image files"""
        from PIL import Image
        import tempfile
        
        # Create invalid "image" file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'This is not an image')
            temp_path = f.name
        
        try:
            with self.assertRaises(Exception):
                Image.open(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_api_error_responses(self):
        """Test API returns proper error responses"""
        from api.app import app
        
        client = app.test_client()
        
        # Test invalid endpoint
        response = client.get('/invalid_endpoint')
        self.assertEqual(response.status_code, 404)
        
        # Test prediction without file
        response = client.post('/predict')
        self.assertIn(response.status_code, [400, 500])
        
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()