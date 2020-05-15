from app import app
import json
import unittest

class APITest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_successful_response(self):
        # Given
        payload = json.dumps({
            "message": "fire",
        })
        
        # When
        response = self.app.post('/predict', headers={"Content-Type": "application/json"}, data=payload)
        # Then
        self.assertEqual(list, type(response.json['prediction']))
        self.assertEqual(200, response.status_code)

    