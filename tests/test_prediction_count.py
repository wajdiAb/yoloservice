import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestPredictionCountMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Override auth and DB dependencies
        def override_get_db():
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.count_predictions_last_week", return_value=3)
    def test_prediction_count_format(self, mock_count):
        """Check response format and status"""
        resp = self.client.get("/predictions/count")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert data["count"] == 3
        mock_count.assert_called_once()

    @patch("app.count_predictions_last_week", return_value=1)
    def test_prediction_count_last_7_days(self, mock_count):
        """Ensure count uses correct helper"""
        resp = self.client.get("/predictions/count")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1
        mock_count.assert_called_once()
