import unittest
from fastapi.testclient import TestClient

from app import app

class TestPredictionCount(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_prediction_count_returns_integer(self):
        """Test that /prediction/count returns an integer count"""
        response = self.client.get("/predictions/count")
        self.assertEqual(response.status_code, 200)
        # If the endpoint returns a raw integer
        if isinstance(response.json(), int):
            count = response.json()
        # If the endpoint returns a dict (e.g., {"prediction_count": 5})
        elif isinstance(response.json(), dict):
            count = list(response.json().values())[0]
        else:
            self.fail("Unexpected response format")
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)