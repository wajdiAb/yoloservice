# tests/test_label_filter.py
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestLabelFilterMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # ---- dependency overrides: no real auth/DB ----
        def override_get_db():
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: "labeluser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.get_predictions_by_label")
    def test_label_match(self, mock_get_by_label):
        # The route returns [{"uid": row.uid, "timestamp": row.timestamp}, ...]
        mock_get_by_label.return_value = [
            MagicMock(uid="label-test-uid", timestamp="2025-07-30T10:00:00Z")
        ]

        resp = self.client.get("/predictions/label/dog")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list) and len(data) >= 1
        assert data[0]["uid"] == "label-test-uid"
        # ensure the route called the query with correct args
        mock_get_by_label.assert_called_once()
        # args: (db_session, label, username)
        _db, label, username = mock_get_by_label.call_args[0]
        assert label == "dog"
        assert username == "labeluser"

    @patch("app.get_predictions_by_label", return_value=[])
    def test_label_not_found(self, mock_get_by_label):
        resp = self.client.get("/predictions/label/notalabel")
        assert resp.status_code == 200
        assert resp.json() == []
        mock_get_by_label.assert_called_once()
        _db, label, username = mock_get_by_label.call_args[0]
        assert label == "notalabel"
        assert username == "labeluser"
