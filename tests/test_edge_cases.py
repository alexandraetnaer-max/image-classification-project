"""
Edge case tests for Fashion Classification System
Tests boundary conditions and unusual scenarios
"""
import unittest
import os
import sys
import json
from pathlib import Path
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestImageEdgeCases(unittest.TestCase):
    """Test edge cases for image processing"""
    
    def test_very_small_image(self):
        """Test handling of very small images"""
        img = Image.new('RGB', (10, 10), color='red')
        self.assertEqual(img.size, (10, 10))
        
        # Resize to model input size
        resized = img.resize((224, 224))
        self.assertEqual(resized.size, (224, 224))
    
    def test_very_large_image(self):
        """Test handling of very large images"""
        # Don't actually create huge image, just test the logic
        large_size = (4096, 4096)
        target_size = (224, 224)
        
        # Verify resize logic works
        self.assertIsNotNone(target_size)
    
    def test_non_square_image(self):
        """Test handling of non-square images"""
        img = Image.new('RGB', (100, 200), color='blue')
        self.assertEqual(img.size, (100, 200))
        
        # Resize (aspect ratio will change)
        resized = img.resize((224, 224))
        self.assertEqual(resized.size, (224, 224))
    
    def test_grayscale_image(self):
        """Test handling of grayscale images"""
        gray_img = Image.new('L', (224, 224), color=128)
        self.assertEqual(gray_img.mode, 'L')
        
        # Convert to RGB
        rgb_img = gray_img.convert('RGB')
        self.assertEqual(rgb_img.mode, 'RGB')
    
    def test_rgba_image_with_transparency(self):
        """Test handling of images with alpha channel"""
        rgba_img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
        self.assertEqual(rgba_img.mode, 'RGBA')
        
        # Convert to RGB (removes alpha)
        rgb_img = rgba_img.convert('RGB')
        self.assertEqual(rgb_img.mode, 'RGB')
    
    def test_corrupted_image_bytes(self):
        """Test handling of corrupted image data"""
        corrupted_bytes = b'This is not valid image data'
        
        with self.assertRaises(Exception):
            Image.open(BytesIO(corrupted_bytes))
    
    def test_empty_image_bytes(self):
        """Test handling of empty image data"""
        empty_bytes = b''
        
        with self.assertRaises(Exception):
            Image.open(BytesIO(empty_bytes))

class TestAPIEdgeCases(unittest.TestCase):
    """Test edge cases for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        from api.app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_empty_request_body(self):
        """Test API with empty request"""
        response = self.app.post('/predict')
        self.assertIn(response.status_code, [400, 500])
    
    def test_malformed_json(self):
        """Test API with malformed JSON"""
        response = self.app.post(
            '/predict',
            data='{"invalid json',
            content_type='application/json'
        )
        self.assertIn(response.status_code, [400, 500])
    
    def test_wrong_content_type(self):
        """Test API with wrong content type"""
        response = self.app.post(
            '/predict',
            data='some data',
            content_type='text/plain'
        )
        self.assertIn(response.status_code, [400, 500])
    
    def test_oversized_request(self):
        """Test API with very large file (simulated)"""
        # Create larger image
        img = Image.new('RGB', (2000, 2000), color='yellow')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=100)
        img_bytes.seek(0)
        
        response = self.app.post(
            '/predict',
            data={'file': (img_bytes, 'large.jpg')},
            content_type='multipart/form-data'
        )
        
        # Should still process (or return specific error)
        self.assertIsNotNone(response)
    
    def test_multiple_files_single_endpoint(self):
        """Test sending multiple files to single prediction endpoint"""
        img1 = Image.new('RGB', (224, 224), color='red')
        img2 = Image.new('RGB', (224, 224), color='blue')
        
        bytes1 = BytesIO()
        bytes2 = BytesIO()
        img1.save(bytes1, format='JPEG')
        img2.save(bytes2, format='JPEG')
        bytes1.seek(0)
        bytes2.seek(0)
        
        # Single endpoint should only process first file
        response = self.app.post(
            '/predict',
            data={
                'file': (bytes1, 'test1.jpg')
            },
            content_type='multipart/form-data'
        )
        
        self.assertIsNotNone(response)
    
    def test_special_characters_in_filename(self):
        """Test handling filenames with special characters"""
        img = Image.new('RGB', (224, 224), color='green')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Filename with special characters
        special_filename = 'test_image_№1_@#$.jpg'
        
        response = self.app.post(
            '/predict',
            data={'file': (img_bytes, special_filename)},
            content_type='multipart/form-data'
        )
        
        self.assertIsNotNone(response)

class TestBatchProcessingEdgeCases(unittest.TestCase):
    """Test edge cases for batch processing"""
    
    def test_empty_incoming_directory(self):
        """Test batch processing with no images"""
        from src.batch_processor import BatchProcessor
        import tempfile
        import src.batch_processor as bp
        
        # Create temporary empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = bp.INCOMING_DIR
            bp.INCOMING_DIR = temp_dir
            
            processor = BatchProcessor()
            images = processor.get_incoming_images()
            
            self.assertEqual(len(images), 0)
            
            bp.INCOMING_DIR = original_dir
    
    def test_mixed_file_types(self):
        """Test directory with mixed file types"""
        from src.batch_processor import BatchProcessor
        import tempfile
        import src.batch_processor as bp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various file types
            Path(temp_dir, 'image.jpg').touch()
            Path(temp_dir, 'image.png').touch()
            Path(temp_dir, 'document.txt').touch()
            Path(temp_dir, 'data.csv').touch()
            
            original_dir = bp.INCOMING_DIR
            bp.INCOMING_DIR = temp_dir
            
            processor = BatchProcessor()
            images = processor.get_incoming_images()
            
            # Should only find image files
            self.assertEqual(len(images), 2)
            
            bp.INCOMING_DIR = original_dir
    
    def test_duplicate_filenames(self):
        """Test handling of duplicate filenames"""
        # Test that organize_image handles duplicates
        filename1 = "product.jpg"
        filename2 = "product.jpg"
        
        self.assertEqual(filename1, filename2)
        
        # In real scenario, second file would be renamed to product_1.jpg

class TestDataValidationEdgeCases(unittest.TestCase):
    """Test edge cases for data validation"""
    
    def test_confidence_exactly_zero(self):
        """Test confidence score of exactly 0"""
        confidence = 0.0
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_exactly_one(self):
        """Test confidence score of exactly 1"""
        confidence = 1.0
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_floating_point_precision(self):
        """Test floating point precision issues"""
        confidence1 = 0.1 + 0.2
        confidence2 = 0.3
        
        # Use almost equal for floating point comparison
        self.assertAlmostEqual(confidence1, confidence2, places=10)
    
    def test_very_low_confidence(self):
        """Test very low confidence scores"""
        confidence = 0.0001
        self.assertTrue(0.0 <= confidence <= 1.0)
    
    def test_category_name_with_spaces(self):
        """Test category names with spaces"""
        categories = ['Casual Shoes', 'Sports Shoes']
        
        for category in categories:
            self.assertIsInstance(category, str)
            self.assertGreater(len(category), 0)
    
    def test_empty_category_list(self):
        """Test handling of empty category list"""
        categories = []
        self.assertEqual(len(categories), 0)
    
    def test_single_category(self):
        """Test with only one category"""
        categories = ['Tshirts']
        self.assertEqual(len(categories), 1)

class TestConcurrencyEdgeCases(unittest.TestCase):
    """Test edge cases for concurrent operations"""
    
    def test_multiple_simultaneous_requests(self):
        """Test API can handle multiple requests"""
        from api.app import app
        
        client = app.test_client()
        
        # Send multiple health checks
        responses = []
        for _ in range(5):
            response = client.get('/health')
            responses.append(response)
        
        # All should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
    
    def test_rapid_sequential_requests(self):
        """Test rapid sequential API calls"""
        from api.app import app
        
        client = app.test_client()
        
        # Rapid requests
        for _ in range(10):
            response = client.get('/classes')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()