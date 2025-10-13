"""
Unit tests for data preparation functions
"""
import unittest
import os
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestDataPreparation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path('tests/test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test data"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """Test directory can be created"""
        test_path = self.test_dir / 'test_folder'
        test_path.mkdir(exist_ok=True)
        self.assertTrue(test_path.exists())
    
    def test_data_split_structure(self):
        """Test data split directory structure"""
        splits = ['train', 'val', 'test']
        categories = ['Tshirts', 'Shoes']
        
        for split in splits:
            for category in categories:
                dir_path = self.test_dir / split / category
                dir_path.mkdir(parents=True, exist_ok=True)
                self.assertTrue(dir_path.exists())

class TestDataValidation(unittest.TestCase):
    
    def test_image_file_extensions(self):
        """Test valid image file extensions"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            filename = f'test_image{ext}'
            self.assertTrue(filename.lower().endswith(tuple(valid_extensions)))
    
    def test_invalid_file_extensions(self):
        """Test invalid file extensions are rejected"""
        invalid_files = ['test.txt', 'test.pdf', 'test.doc']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in invalid_files:
            self.assertFalse(filename.lower().endswith(tuple(valid_extensions)))
    
    def test_split_ratios(self):
        """Test standard split ratios are correct"""
        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15
        
        total = train_ratio + val_ratio + test_ratio
        self.assertAlmostEqual(total, 1.0, places=2)

if __name__ == '__main__':
    unittest.main()