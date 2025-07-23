import unittest
import os
import sqlite3
import time
from fastapi.testclient import TestClient
from unittest.mock import patch

AUTH = ("testuser", "testpass")

class TestDeletePredictionEndpoint(unittest.TestCase):
    def setUp(self):
        from app import app, DB_PATH
        self.DB_PATH = DB_PATH
        self.AUTH = AUTH
        self.client = TestClient(app)
        self.uid = "test-delete-uid"
        self.original_image = f"uploads/original/{self.uid}.jpg"
        self.predicted_image = f"uploads/predicted/{self.uid}.jpg"

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        with open(self.original_image, "wb") as f:
            f.write(b"dummy")
        with open(self.predicted_image, "wb") as f:
            f.write(b"dummy")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, original_image, predicted_image, username) VALUES (?, ?, ?, ?)",
                (self.uid, self.original_image, self.predicted_image, AUTH[0])
            )
            conn.execute(
                "INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                (self.uid, "cat", 0.9, "[0,0,10,10]")
            )

    def test_delete_prediction_success(self):
        response = self.client.delete(f"/prediction/{self.uid}", auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")

        # Ensure DB is clean
        with sqlite3.connect(self.DB_PATH) as conn:
            self.assertIsNone(conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (self.uid,)).fetchone())
            self.assertIsNone(conn.execute("SELECT * FROM detection_objects WHERE prediction_uid = ?", (self.uid,)).fetchone())

        # Ensure files are removed
        self.assertFalse(os.path.exists(self.original_image))
        self.assertFalse(os.path.exists(self.predicted_image))

    def test_delete_prediction_not_found(self):
        response = self.client.delete("/prediction/nonexistent-id", auth=AUTH)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_delete_prediction_files_already_deleted(self):
        # Manually delete files before calling endpoint
        if os.path.exists(self.original_image):
            os.remove(self.original_image)
        if os.path.exists(self.predicted_image):
            os.remove(self.predicted_image)

        response = self.client.delete(f"/prediction/{self.uid}", auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")

    def test_delete_prediction_file_remove_fails(self):
        # Patch os.remove to simulate error
        with patch("os.remove", side_effect=OSError("Simulated deletion error")):
            response = self.client.delete(f"/prediction/{self.uid}", auth=AUTH)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "deleted")

    def tearDown(self):
        for path in [self.original_image, self.predicted_image]:
            if os.path.exists(path):
                os.remove(path)
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))

    def test_safe_delete_file_exception_logged(self):
        import os
        from app import safe_delete_file
        import tempfile

        path = tempfile.NamedTemporaryFile(delete=False).name

        # Monkey-patch os.remove to raise error
        original_remove = os.remove
        os.remove = lambda p: (_ for _ in ()).throw(OSError("fail"))

        try:
            safe_delete_file(path)
        finally:
            os.remove = original_remove
            if os.path.exists(path):
                os.unlink(path)

if __name__ == "__main__":
    unittest.main()
