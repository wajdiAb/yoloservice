import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, get_current_username, get_db

AUTH_USER = "testuser"
UID = "test-delete-uid"
ORIG = f"uploads/original/{UID}.jpg"
PRED = f"uploads/predicted/{UID}.jpg"


class TestDeletePredictionEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # ---- dependency overrides ----
        def override_get_db():
            # yield a mock Session so Depends(get_db) works
            yield MagicMock()

        app.dependency_overrides[get_current_username] = lambda: AUTH_USER
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides = {}

    # ------------------- tests -------------------

    @patch("app.safe_delete_file")
    @patch("app.delete_prediction_and_detections")
    @patch("app.get_prediction_file_paths", return_value=(ORIG, PRED))
    def test_delete_prediction_success(self, mock_paths, mock_delete_db, mock_safe_delete):
        """Happy path: paths found -> delete DB rows and both files"""
        resp = self.client.delete(f"/prediction/{UID}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "deleted")
        self.assertEqual(resp.json()["uid"], UID)

        # DB deletion must happen
        mock_delete_db.assert_called_once()
        # Both files should be scheduled for deletion
        mock_safe_delete.assert_any_call(ORIG)
        mock_safe_delete.assert_any_call(PRED)
        self.assertEqual(mock_safe_delete.call_count, 2)

    @patch("app.get_prediction_file_paths", return_value=None)
    def test_delete_prediction_not_found(self, mock_paths):
        """If no paths for uid/user -> 404"""
        resp = self.client.delete("/prediction/nonexistent-id")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "Prediction not found")

    @patch("app.safe_delete_file")
    @patch("app.delete_prediction_and_detections")
    @patch("app.get_prediction_file_paths", return_value=(ORIG, PRED))
    @patch("app.os.path.exists", return_value=False)  # make safe_delete_file do nothing internally
    def test_delete_prediction_files_already_deleted(
        self, mock_exists, mock_paths, mock_delete_db, mock_safe_delete
    ):
        """Files already gone -> still 200 and DB deleted"""
        resp = self.client.delete(f"/prediction/{UID}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "deleted")
        # safe_delete_file still called, but it will check exists (False) and skip os.remove
        self.assertEqual(mock_safe_delete.call_count, 2)

    def test_safe_delete_file_exception_logged(self):
        """Unit test the helper to ensure it logs (doesn't raise) on deletion failure."""
        from app import safe_delete_file

        with patch("app.os.path.exists", return_value=True), \
             patch("app.os.remove", side_effect=OSError("fail")), \
             patch("logging.getLogger") as get_logger:

            logger = MagicMock()
            get_logger.return_value = logger

            # should not raise even though os.remove fails
            safe_delete_file("uploads/bad_dir")

            # verify we log a warning with the failing path
            self.assertTrue(logger.warning.called)
            args, kwargs = logger.warning.call_args
            self.assertIn("Failed to delete file", args[0])
            self.assertIn("uploads/bad_dir", args[0])
