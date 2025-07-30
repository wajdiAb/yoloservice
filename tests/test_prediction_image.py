# tests/test_prediction_image_mocked.py
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import Response

from app import app, get_current_username, get_db


class TestPredictionImageMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.uid = "img-uid"
        cls.path = f"uploads/predicted/{cls.uid}.jpg"

        # ---- dependency overrides: no real auth/DB ----
        def override_get_db():
            yield MagicMock()  # fake Session

        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    # ---------- success: PNG ----------
    @patch("app.FileResponse")
    @patch("app.os.path.exists", return_value=True)
    @patch("app.get_predicted_image_path")
    def test_get_prediction_image_png(self, mock_get_path, mock_exists, mock_file_response):
        mock_get_path.return_value = self.path
        # Return a simple Response so we don't need a real file on disk
        mock_file_response.return_value = Response(content=b"png", media_type="image/png")

        resp = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={"Accept": "image/png"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

        # Verify helper received db + username
        (db_arg, uid_arg, username_arg) = mock_get_path.call_args[0]
        assert uid_arg == self.uid
        assert username_arg == "testuser"
        assert db_arg is not None

        # Ensure FileResponse called with path + media_type=png
        called_path = mock_file_response.call_args.kwargs.get("path") or mock_file_response.call_args.args[0]
        called_media = mock_file_response.call_args.kwargs.get("media_type")
        assert called_path.endswith(self.path)
        assert called_media == "image/png"

    # ---------- success: JPEG/JPG ----------
    @patch("app.FileResponse")
    @patch("app.os.path.exists", return_value=True)
    @patch("app.get_predicted_image_path")
    def test_get_prediction_image_jpeg(self, mock_get_path, mock_exists, mock_file_response):
        mock_get_path.return_value = self.path
        mock_file_response.return_value = Response(content=b"jpeg", media_type="image/jpeg")

        for accept in ("image/jpeg", "image/jpg"):
            resp = self.client.get(
                f"/prediction/{self.uid}/image",
                headers={"Accept": accept},
            )
            assert resp.status_code == 200
            assert "image" in resp.headers["content-type"]

        called_media = mock_file_response.call_args.kwargs.get("media_type")
        assert called_media == "image/jpeg"

    # ---------- not acceptable (406) ----------
    @patch("app.os.path.exists", return_value=True)
    @patch("app.get_predicted_image_path", return_value="uploads/predicted/whatever.jpg")
    @patch("app.FileResponse")  # should NOT be called
    def test_get_prediction_image_not_acceptable(self, mock_file_response, mock_get_path, mock_exists):
        resp = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={"Accept": "application/json"},
        )
        assert resp.status_code == 406
        assert resp.json()["detail"] == "Client does not accept an image format"
        mock_file_response.assert_not_called()

    # ---------- not found: uid/ownership ----------
    @patch("app.get_predicted_image_path", return_value=None)
    def test_get_prediction_image_not_found_by_uid(self, mock_get_path):
        resp = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={"Accept": "image/png"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Prediction not found or not authorized"

    # ---------- file exists check -> missing file ----------
    @patch("app.os.path.exists", return_value=False)
    @patch("app.get_predicted_image_path", return_value="uploads/predicted/missing.jpg")
    def test_prediction_image_file_missing(self, mock_get_path, mock_exists):
        resp = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={"Accept": "image/png"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Predicted image file not found"
