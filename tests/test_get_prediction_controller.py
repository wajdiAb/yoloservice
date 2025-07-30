# tests/test_get_prediction_controller.py

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app
from models import Base
from sqlalchemy import create_engine

def setUpModule():
   
    engine = create_engine("sqlite:///./preductions.db")  
    Base.metadata.create_all(bind=engine)

client = TestClient(app)

# Fake class to simulate a prediction model object
class FakePrediction:
    def __init__(self, uid, timestamp, original_image, predicted_image):
        self.uid = uid
        self.timestamp = timestamp
        self.original_image = original_image
        self.predicted_image = predicted_image


class TestGetPredictionByUID(unittest.TestCase):

    @patch("app.get_prediction")  # Patch where it's used (in app.py), not where it's defined
    @patch("app.get_current_username", return_value="testuser")
    def test_prediction_found(self, mock_auth, mock_get_prediction):
        # Arrange
        fake = FakePrediction(
            uid="abc123",
            timestamp="2023-01-01T00:00:00",
            original_image="uploads/original/abc123.jpg",
            predicted_image="uploads/predicted/abc123.jpg"
        )
        mock_get_prediction.return_value = fake

        # Act
        response = client.get("/prediction/abc123", auth=("testuser", "whatever"))

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "uid": "abc123",
            "timestamp": "2023-01-01T00:00:00",
            "original_image": "uploads/original/abc123.jpg",
            "predicted_image": "uploads/predicted/abc123.jpg",
            "detection_objects": []  # Since we didn't mock get_detections yet
        })

    @patch("app.get_prediction")
    @patch("app.get_current_username", return_value="testuser")
    def test_prediction_not_found(self, mock_auth, mock_get_prediction):
        mock_get_prediction.return_value = None

        response = client.get("/prediction/notfound", auth=("testuser", "whatever"))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Prediction not found or not authorized"})
