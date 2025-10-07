"""
Test script for API
"""
import requests

API_URL = "http://localhost:5000"

print("=" * 50)
print("TEST 1: Health Check")
print("=" * 50)

response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response:")
print(response.json())
print()

print("=" * 50)
print("TEST 2: Get Classes")
print("=" * 50)

response = requests.get(f"{API_URL}/classes")
print(f"Status: {response.status_code}")
print(f"Response:")
print(response.json())
print()

print("=" * 50)
print("TEST 3: Image Prediction")
print("=" * 50)

import os

# Use absolute path
test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')

if os.path.exists(test_image_path):
    print(f"Testing with: {test_image_path}")
    
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prediction successful!")
        print(f"  Category: {result.get('category')}")
        print(f"  Confidence: {result.get('confidence', 0)*100:.1f}%")
        print(f"\n  Top 5 predictions:")
        for cat, prob in list(result.get('all_probabilities', {}).items())[:5]:
            print(f"    {prob*100:5.1f}% - {cat}")
    else:
        print(f"❌ Error: {response.json()}")
else:
    print(f"❌ test_image.jpg not found at: {test_image_path}")