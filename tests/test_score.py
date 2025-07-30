# tests/test_score_filter_mocked.py
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db


class TestScoreFilterMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # ---- dependency overrides: no real auth/DB ----
        def override_get_db():
            yield MagicMock()  # fake Session

        app.dependency_overrides[get_current_username] = lambda: "scoreuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    @patch("app.get_predictions_by_score")
    def test_score_match(self, mock_get_by_score):
        # Route returns [{"uid": row.uid, "timestamp": row.timestamp}, ...]
        mock_get_by_score.return_value = [
            MagicMock(uid="score-test-uid", timestamp="2025-07-30T10:00:00Z")
        ]

        resp = self.client.get("/predictions/score/0.3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["uid"] == "score-test-uid"

        # Verify helper called with (db, min_score, username)
        db_arg, min_score_arg, username_arg = mock_get_by_score.call_args[0]
        assert isinstance(min_score_arg, float)  # should be converted from path param
        assert min_score_arg == 0.3
        assert username_arg == "scoreuser"
        assert db_arg is not None

    @patch("app.get_predictions_by_score", return_value=[])
    def test_score_too_high(self, mock_get_by_score):
        resp = self.client.get("/predictions/score/0.95")
        assert resp.status_code == 200
        assert resp.json() == []

        db_arg, min_score_arg, username_arg = mock_get_by_score.call_args[0]
        assert min_score_arg == 0.95
        assert username_arg == "scoreuser"
