"""
Test Runner for Fashion Classification System
Runs all unit tests and generates report
"""
import unittest
import sys
import os
from io import StringIO
from datetime import datetime

def run_all_tests():
    """Run all tests and generate report"""
    print("=" * 70)
    print("FASHION CLASSIFICATION SYSTEM - TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print()
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0
    else:
        print()
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1

def run_specific_test(test_module):
    """Run specific test module"""
    print(f"Running tests from: {test_module}")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        exit_code = run_specific_test(test_module)
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)