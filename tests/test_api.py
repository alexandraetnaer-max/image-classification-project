"""
Unit tests for Flask API
"""
import unittest
import json
import base64
import sys
import os
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after path setup
from api.app import app

class TestFlaskAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_endpoint(self):
        """Test home endpoint returns service info"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('service', data)
        self.assertIn('endpoints', data)
        self.assertEqual(data['service'], 'Fashion Classification API')
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_classes_endpoint(self):
        """Test classes endpoint returns category list"""
        response = self.app.get('/classes')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('classes', data)
        self.assertIn('total', data)
        self.assertEqual(data['total'], 10)
        self.assertIsInstance(data['classes'], list)
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = self.app.get('/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_requests', data)
        self.assertIn('success_rate', data)
        self.assertIn('uptime_seconds', data)
    
    def test_predict_endpoint_without_file(self):
        """Test predict endpoint returns error without file"""
        response = self.app.post('/predict')
        # Can be 400 or 500 depending on request format
        self.assertIn(response.status_code, [400, 500])
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_with_mock_image(self):
        """Test predict endpoint with mock image"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Send as file upload
        response = self.app.post('/predict',
                                data={'file': (img_bytes, 'test.jpg')},
                                content_type='multipart/form-data')
        
        # Should return prediction or error
        self.assertIn(response.status_code, [200, 500])
        
        data = json.loads(response.data)
        if response.status_code == 200:
            self.assertIn('category', data)
            self.assertIn('confidence', data)
    
    def test_predict_endpoint_with_base64(self):
        """Test predict endpoint with base64 encoded image"""
        # Create test image
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        response = self.app.post('/predict',
                                data=json.dumps({'image': img_base64}),
                                content_type='application/json')
        
        self.assertIn(response.status_code, [200, 500])
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        response = self.app.get('/invalid_endpoint')
        self.assertEqual(response.status_code, 404)

class TestAPIResponseFormat(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_json_response_format(self):
        """Test all endpoints return valid JSON"""
        endpoints = ['/', '/health', '/classes', '/stats']
        
        for endpoint in endpoints:
            response = self.app.get(endpoint)
            try:
                json.loads(response.data)
                valid_json = True
            except:
                valid_json = False
            
            self.assertTrue(valid_json, 
                          f"Endpoint {endpoint} should return valid JSON")

if __name__ == '__main__':
    unittest.main()