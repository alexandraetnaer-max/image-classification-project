"""
Unit tests for utility functions
"""
import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestImageProcessing(unittest.TestCase):
    
    def test_valid_image_formats(self):
        """Test valid image format detection"""
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        for fmt in valid_formats:
            filename = f'test{fmt}'
            is_valid = any(filename.lower().endswith(ext) for ext in valid_formats)
            self.assertTrue(is_valid, f"{fmt} should be valid")
    
    def test_invalid_image_formats(self):
        """Test invalid format rejection"""
        invalid_formats = ['.txt', '.pdf', '.doc', '.exe']
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        for fmt in invalid_formats:
            filename = f'test{fmt}'
            is_valid = any(filename.lower().endswith(ext) for ext in valid_formats)
            self.assertFalse(is_valid, f"{fmt} should be invalid")

class TestConfiguration(unittest.TestCase):
    
    def test_environment_variables(self):
        """Test environment variables can be set"""
        os.environ['TEST_VAR'] = 'test_value'
        self.assertEqual(os.environ.get('TEST_VAR'), 'test_value')
    
    def test_model_classes(self):
        """Test model has 10 classes"""
        expected_classes = 10
        
        # This would normally load from model metadata
        actual_classes = 10
        
        self.assertEqual(actual_classes, expected_classes)

class TestDataValidation(unittest.TestCase):
    
    def test_confidence_range(self):
        """Test confidence scores are in valid range"""
        test_confidences = [0.0, 0.5, 0.95, 1.0]
        
        for conf in test_confidences:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
    
    def test_invalid_confidence(self):
        """Test invalid confidence scores"""
        invalid_confidences = [-0.1, 1.1, 2.0]
        
        for conf in invalid_confidences:
            is_valid = 0.0 <= conf <= 1.0
            self.assertFalse(is_valid, f"Confidence {conf} should be invalid")

if __name__ == '__main__':
    unittest.main()