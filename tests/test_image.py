import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import Response

from app import app, get_current_username, get_db


class TestGetImageEndpointMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # ---- dependency overrides ----
        def override_get_db():
            # Return a mock Session so Depends(get_db) works without a real DB
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

        cls.filename = "some-uid.jpg"  # we don't need to create it on disk

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    # ----------------- tests -----------------

    def test_invalid_type_400(self):
        # No mocks required; route rejects invalid type before IO/DB
        resp = self.client.get(f"/image/invalid/{self.filename}")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid image type"

    @patch("app.is_image_owned_by_user", return_value=True)  # not reached but harmless
    @patch("app.os.path.exists", return_value=False)
    def test_not_found_image_404(self, mock_exists, mock_own):
        resp = self.client.get(f"/image/original/{self.filename}")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Image not found"

    @patch("app.os.path.exists", return_value=True)
    @patch("app.is_image_owned_by_user", return_value=False)
    def test_not_owned_image_403(self, mock_owned, mock_exists):
        resp = self.client.get(f"/image/original/{self.filename}")
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Not authorized to access this image"

    @patch("app.os.path.exists", return_value=True)
    @patch("app.is_image_owned_by_user", return_value=True)
    @patch("app.FileResponse")
    def test_valid_image_access_original_200(self, mock_file_response, mock_owned, mock_exists):
        # Make FileResponse return a simple Response so we don't need a real file
        mock_file_response.return_value = Response(content=b"fake-bytes", media_type="image/jpeg")

        resp = self.client.get(f"/image/original/{self.filename}")
        assert resp.status_code == 200
        # Starlette may send default media type for Response; assert that FileResponse was used correctly
        mock_file_response.assert_called_once()
        # Ensure path assembled as 'uploads/original/<filename>'
        called_path = mock_file_response.call_args.args[0]
        assert called_path.endswith(f"uploads/original/{self.filename}")

    @patch("app.os.path.exists", return_value=True)
    @patch("app.is_image_owned_by_user", return_value=True)
    @patch("app.FileResponse")
    def test_valid_image_access_predicted_200(self, mock_file_response, mock_owned, mock_exists):
        mock_file_response.return_value = Response(content=b"fake-bytes", media_type="image/jpeg")

        resp = self.client.get(f"/image/predicted/{self.filename}")
        assert resp.status_code == 200
        mock_file_response.assert_called_once()
        called_path = mock_file_response.call_args.args[0]
        assert called_path.endswith(f"uploads/predicted/{self.filename}")
