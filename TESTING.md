```markdown
# Testing Guide

## Overview

The Fashion Classification System includes a comprehensive test suite covering all major components: data preparation, API endpoints, batch processing, and utilities.

---

## Test Structure
tests/
├── init.py                    # Test package initializer
├── test_data_preparation.py       # Data prep tests
├── test_api.py                    # API endpoint tests
├── test_batch_processor.py        # Batch processing tests
└── test_utils.py                  # Utility function tests

---

## Running Tests

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
Run All Tests
bashpython run_tests.py
Expected output:
======================================================================
FASHION CLASSIFICATION SYSTEM - TEST SUITE
======================================================================
Started: 2025-10-13 10:00:00

test_create_directory_structure ... ok
test_health_endpoint ... ok
test_predict_endpoint_with_mock_image ... ok
...

======================================================================
TEST SUMMARY
======================================================================
Tests Run: 25
Successes: 25
Failures: 0
Errors: 0
Skipped: 0

✓ ALL TESTS PASSED!
======================================================================
Run Specific Test Module
bash# Data preparation tests only
python run_tests.py test_data_preparation

# API tests only
python run_tests.py test_api

# Batch processing tests only
python run_tests.py test_batch_processor

# Utility tests only
python run_tests.py test_utils
Run Individual Test Case
bash# Specific test class
python -m unittest tests.test_api.TestFlaskAPI

# Specific test method
python -m unittest tests.test_api.TestFlaskAPI.test_health_endpoint
Verbose Output
bash# More detailed output
python -m unittest discover -s tests -p "test_*.py" -v

Test Coverage
1. Data Preparation Tests
File: tests/test_data_preparation.py
Coverage:

Directory structure creation
Data split verification (70/15/15)
File format validation
Category organization

Key Tests:

test_create_directory_structure - Verifies train/val/test folders
test_verify_data_split_ratios - Checks correct split ratios
test_image_file_extensions - Validates image formats
test_invalid_file_extensions - Rejects non-image files


2. API Tests
File: tests/test_api.py
Coverage:

All REST endpoints
Request/response formats
Error handling
JSON validation
File upload (multipart)
Base64 image encoding

Key Tests:

test_home_endpoint - Verifies service info
test_health_endpoint - Checks health status
test_classes_endpoint - Validates category list
test_stats_endpoint - Tests statistics endpoint
test_predict_endpoint_with_mock_image - Tests prediction with image
test_predict_endpoint_with_base64 - Tests base64 encoding
test_invalid_endpoint - Verifies 404 handling


3. Batch Processing Tests
File: tests/test_batch_processor.py
Coverage:

BatchProcessor initialization
Image file detection
Result structure validation
Summary generation
Directory handling

Key Tests:

test_batch_processor_initialization - Verifies setup
test_get_incoming_images_empty - Empty directory handling
test_get_incoming_images_with_files - File detection
test_process_image_structure - Result format validation
test_result_structure - CSV structure check
test_summary_structure - JSON summary validation


4. Utility Tests
File: tests/test_utils.py
Coverage:

Image format validation
Configuration management
Data validation
Confidence score ranges

Key Tests:

test_valid_image_formats - Accepts valid formats
test_invalid_image_formats - Rejects invalid formats
test_confidence_range - Validates 0-1 range
test_invalid_confidence - Rejects invalid values


Test Development Guidelines
Writing New Tests

Create test file:

python# tests/test_new_feature.py
import unittest

class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        """Prepare test environment"""
        pass
    
    def tearDown(self):
        """Clean up after test"""
        pass
    
    def test_something(self):
        """Test specific functionality"""
        result = some_function()
        self.assertEqual(result, expected_value)

Follow naming conventions:

Test files: test_*.py
Test classes: Test*
Test methods: test_*


Use descriptive names:

❌ test_1, test_function
✅ test_health_endpoint_returns_200, test_invalid_input_raises_error


Include docstrings:

pythondef test_feature(self):
    """Test that feature X behaves correctly when Y happens"""
    pass
Best Practices
1. Isolation:

Each test should be independent
Use setUp() and tearDown() for fixtures
Don't rely on test execution order

2. Mock External Dependencies:
pythonfrom unittest.mock import Mock, patch

@patch('requests.get')
def test_api_call(self, mock_get):
    mock_get.return_value.status_code = 200
    # Test code here
3. Test Edge Cases:

Empty inputs
Invalid data
Boundary conditions
Error scenarios

4. Use Assertions:
python# Equality
self.assertEqual(actual, expected)
self.assertNotEqual(actual, unexpected)

# Truthiness
self.assertTrue(condition)
self.assertFalse(condition)

# Membership
self.assertIn(item, collection)
self.assertNotIn(item, collection)

# Exceptions
self.assertRaises(ValueError, function, args)

# Numeric
self.assertGreater(a, b)
self.assertLess(a, b)
self.assertAlmostEqual(a, b, places=2)

Continuous Integration
GitHub Actions Example
yamlname: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: python run_tests.py
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test-results/

Troubleshooting
Tests Fail to Import Modules
Problem: ModuleNotFoundError
Solution:
python# Add to top of test file
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
API Tests Fail
Problem: Model not loaded
Solution:

Ensure model files exist in models/
Check api/app.py loads model correctly
API tests may skip prediction if model unavailable

Batch Processing Tests Fail
Problem: Directories not found
Solution:

Tests create temporary directories
Check setUp() and tearDown() methods
Verify write permissions

All Tests Pass Locally But Fail in CI
Problem: Environment differences
Solution:

Check Python version consistency
Verify all dependencies in requirements.txt
Check file paths (use os.path.join())
Review environment variables


Test Metrics
Coverage Goals

Overall: > 80%
Critical paths: 100%
API endpoints: 100%
Data processing: > 90%

Performance Benchmarks

All tests should complete in < 30 seconds
Individual test < 5 seconds
Setup/teardown < 1 second


Future Improvements

Add integration tests

End-to-end workflows
Database interactions
External API calls


Add performance tests

Load testing
Stress testing
API response times


Add code coverage reporting

bashpip install coverage
coverage run -m unittest discover
coverage report
coverage html

Add test fixtures

Sample images
Mock responses
Test datasets




Support

GitHub Issues: Report test failures
Documentation: See README.md for setup
Contributing: Submit tests with new features