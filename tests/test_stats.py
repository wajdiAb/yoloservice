import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestStatsEndpointMocked(unittest.TestCase):
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

    @patch("app.get_user_prediction_stats")
    def test_stats_endpoint(self, mock_stats):
        # Mock stats return value
        mock_stats.return_value = {
            "total_predictions": 2,
            "average_confidence": 0.9,
            "most_frequent_labels": [
                {"label": "cat", "count": 2},
                {"label": "dog", "count": 1},
            ],
        }

        response = self.client.get("/stats")
        assert response.status_code == 200
        data = response.json()

        assert data["total_predictions"] == 2
        assert data["average_confidence"] == 0.9
        assert len(data["most_frequent_labels"]) == 2

        mock_stats.assert_called_once()
        args = mock_stats.call_args[0]
        assert args[1] == "testuser"  # username from dependency override

    @patch("app.get_user_prediction_stats", return_value={
        "total_predictions": 0,
        "average_confidence": 0.0,
        "most_frequent_labels": []
    })
    def test_stats_with_no_predictions(self, mock_stats):
        resp = self.client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_predictions"] == 0
        assert data["average_confidence"] == 0.0
        assert data["most_frequent_labels"] == []

    @patch("app.get_user_prediction_stats", return_value={
        "total_predictions": 1,
        "average_confidence": 0.0,
        "most_frequent_labels": []
    })
    def test_stats_zero_objects(self, mock_stats):
        resp = self.client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["average_confidence"] == 0.0
