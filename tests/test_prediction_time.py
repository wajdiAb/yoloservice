import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from app import app, get_current_username, get_db


class TestProcessingTimeMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Dependency overrides: skip auth and DB
        def override_get_db():
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.save_detection")
    @patch("app.save_prediction")
    @patch("app.model")
    def test_predict_includes_processing_time(self, mock_model, mock_save_prediction, mock_save_detection):
        # Mock YOLO results
        fake_box = MagicMock()
        fake_cls = MagicMock()
        fake_cls.item.return_value = 0
        fake_box.cls = [fake_cls]
        fake_conf = MagicMock()
        fake_conf.item.return_value = 0.8
        fake_box.conf = [fake_conf]
        fake_bbox = MagicMock()
        fake_bbox.tolist.return_value = [0, 0, 100, 100]
        fake_box.xyxy = [fake_bbox]

        fake_results = MagicMock()
        fake_results.__getitem__.return_value = fake_results
        fake_results.boxes = [fake_box]
        fake_results.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        mock_model.return_value = [fake_results]
        mock_model.names = {0: "person"}

        # Send fake image bytes (we don't care about content)
        response = self.client.post(
            "/predict",
            files={"file": ("fake.jpg", b"fakebytes", "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()

        # Verify processing time field exists and is positive
        assert "time_took" in data
        assert isinstance(data["time_took"], (int, float))
        assert data["time_took"] >= 0

        # Ensure mocks were called
        mock_model.assert_called_once()
        mock_save_prediction.assert_called_once()
        mock_save_detection.assert_called()
