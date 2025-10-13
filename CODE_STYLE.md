markdown# Code Style Guide

## Overview

This project follows Python best practices and PEP 8 style guidelines.

---

## Python Style Guidelines

### General Rules

- **PEP 8:** Follow Python Enhancement Proposal 8
- **Indentation:** 4 spaces (no tabs)
- **Line Length:** Maximum 100 characters (prefer 80)
- **Encoding:** UTF-8

### Naming Conventions

**Variables and Functions:**
```python
# Snake case for variables and functions
user_name = "Alexandra"
image_count = 100

def process_image(image_path):
    pass
Classes:
python# Pascal case for classes
class BatchProcessor:
    pass

class ImageClassifier:
    pass
Constants:
python# All caps with underscores
MAX_BATCH_SIZE = 50
DEFAULT_TIMEOUT = 300
API_VERSION = "1.0"
Private Methods:
pythonclass MyClass:
    def _private_method(self):
        """Private method (internal use)"""
        pass
    
    def public_method(self):
        """Public method (external use)"""
        pass

Documentation
Module Docstrings
python"""
Module Name: batch_processor.py

Description: Automated batch processing for fashion product classification.

Author: Alexandra Etnaer
Date: October 2025
"""
Function Docstrings
pythondef predict_image(image_path, model):
    """
    Predict category for a single image.
    
    Args:
        image_path (str): Path to image file
        model: Loaded Keras model
    
    Returns:
        dict: Prediction results with category and confidence
        
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image format invalid
        
    Example:
        >>> result = predict_image('test.jpg', model)
        >>> print(result['category'])
        'Tshirts'
    """
    pass
Class Docstrings
pythonclass BatchProcessor:
    """
    Automated batch processing for image classification.
    
    This class handles:
    - Scanning incoming directory
    - Sending images to API
    - Organizing results
    - Generating reports
    
    Attributes:
        results (list): Processing results
        timestamp (str): Batch timestamp
        
    Example:
        >>> processor = BatchProcessor()
        >>> processor.process_batch()
    """
    pass

Comments
Inline Comments
python# Good: Explains WHY, not WHAT
confidence = predictions[0][predicted_idx]  # Extract confidence for top prediction

# Bad: States the obvious
x = x + 1  # Increment x
TODO Comments
python# TODO: Add support for video files
# FIXME: Handle edge case when image is corrupted
# NOTE: This assumes images are in RGB format

Code Organization
File Structure
python# 1. Module docstring
"""Module description"""

# 2. Imports (grouped)
# Standard library
import os
import sys

# Third-party
import numpy as np
import tensorflow as tf

# Local
from src.utils import helper_function

# 3. Constants
MAX_SIZE = 1024
DEFAULT_PATH = 'data/'

# 4. Classes
class MyClass:
    pass

# 5. Functions
def my_function():
    pass

# 6. Main execution
if __name__ == '__main__':
    main()
Import Order

Standard library imports
Third-party imports
Local application imports


Error Handling
Try-Except Blocks
python# Good: Specific exceptions
try:
    image = Image.open(filepath)
except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
    raise
except PIL.UnidentifiedImageError:
    logger.error(f"Invalid image format: {filepath}")
    raise

# Bad: Bare except
try:
    image = Image.open(filepath)
except:  # Don't do this!
    pass

Testing
Test Naming
pythonclass TestBatchProcessor(unittest.TestCase):
    
    def test_initialization_succeeds(self):
        """Test that BatchProcessor initializes correctly"""
        processor = BatchProcessor()
        self.assertIsNotNone(processor)
    
    def test_empty_directory_returns_zero_images(self):
        """Test that empty directory returns no images"""
        images = processor.get_incoming_images()
        self.assertEqual(len(images), 0)

Git Commit Messages
Format
<type>: <subject>

<body>

<footer>
Types

feat: New feature
fix: Bug fix
docs: Documentation changes
style: Code style changes (formatting)
refactor: Code refactoring
test: Adding or updating tests
chore: Maintenance tasks

Examples
feat: Add batch processing for multiple images

- Implement BatchProcessor class
- Add automatic file organization
- Generate CSV reports

Closes #123
fix: Handle webp image format in batch processor

The batch processor was not detecting .webp files. Added
.webp to the list of supported formats.

File Naming
Python Files

Use lowercase with underscores: batch_processor.py
Be descriptive: generate_batch_report.py (not report.py)

Documentation

Use UPPERCASE: README.md, LICENSE
Use underscores for multi-word: CODE_STYLE.md

Directories

Lowercase: api/, src/, tests/
Descriptive: monitoring/, results/


Best Practices
1. DRY (Don't Repeat Yourself)
python# Good: Use functions
def resize_image(image, size=(224, 224)):
    return image.resize(size)

img1 = resize_image(img1)
img2 = resize_image(img2)

# Bad: Repeated code
img1 = img1.resize((224, 224))
img2 = img2.resize((224, 224))
2. Single Responsibility
python# Good: Each function does one thing
def load_image(path):
    return Image.open(path)

def preprocess_image(image):
    return image.resize((224, 224))

# Bad: Function does too much
def load_and_process_image(path):
    image = Image.open(path)
    image = image.resize((224, 224))
    image = normalize(image)
    return predict(image)
3. Meaningful Names
python# Good
user_email_address = "user@example.com"
total_image_count = 100

# Bad
x = "user@example.com"
n = 100
4. Use Type Hints
pythonfrom typing import List, Dict, Optional

def process_images(
    image_paths: List[str],
    batch_size: int = 32
) -> Dict[str, float]:
    """Process images and return predictions."""
    pass

Tools
Linting
bash# Install
pip install pylint flake8

# Run
pylint src/
flake8 api/
Formatting
bash# Install
pip install black

# Format
black src/ api/ tests/
Type Checking
bash# Install
pip install mypy

# Check
mypy src/

Resources

PEP 8: https://pep8.org/
Google Python Style: https://google.github.io/styleguide/pyguide.html
Real Python: https://realpython.com/python-pep8/

