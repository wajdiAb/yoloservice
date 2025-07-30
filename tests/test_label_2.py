import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestLabelFilterMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Dependency overrides
        def override_get_db():
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: "labeluser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.get_predictions_by_label")
    def test_label_match(self, mock_get_predictions_by_label):
        """At least one prediction has label 'dog'"""
        mock_get_predictions_by_label.return_value = [
            MagicMock(uid="mock-uid", timestamp="now")
        ]

        response = self.client.get("/predictions/label/dog")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["uid"] == "mock-uid"

    @patch("app.get_predictions_by_label", return_value=[])
    def test_label_not_found(self, mock_get_predictions_by_label):
        """No predictions with given label"""
        response = self.client.get("/predictions/label/notalabel")
        assert response.status_code == 200
        assert response.json() == []
