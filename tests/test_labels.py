# tests/test_labels_endpoint.py
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestLabelsEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # ---- dependency overrides: no real auth/DB ----
        def override_get_db():
            yield MagicMock()  # fake Session

        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.get_unique_labels_last_week", return_value=["person", "car", "bicycle"])
    def test_labels_endpoint_success(self, mock_get_unique):
        resp = self.client.get("/labels")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, dict)
        assert "labels" in data
        labels = data["labels"]
        assert isinstance(labels, list)
        assert labels == ["person", "car", "bicycle"]

        # ensure the route called the query with correct args
        (db_arg, username_arg) = mock_get_unique.call_args[0]
        assert username_arg == "testuser"
        assert db_arg is not None  # it's the MagicMock yielded by override_get_db

    @patch("app.get_unique_labels_last_week", return_value=[])
    def test_labels_endpoint_empty(self, mock_get_unique):
        resp = self.client.get("/labels")
        assert resp.status_code == 200
        assert resp.json() == {"labels": []}
