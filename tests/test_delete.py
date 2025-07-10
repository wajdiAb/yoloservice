import unittest
import os
import sqlite3
import time
from fastapi.testclient import TestClient
from app import app, DB_PATH

class TestDeletePredictionEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.uid = "test-delete-uid"
        self.original_image = f"uploads/original/{self.uid}.jpg"
        self.predicted_image = f"uploads/predicted/{self.uid}.jpg"

        # Ensure upload directories exist
        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        # Create dummy image files
        with open(self.original_image, "wb") as f:
            f.write(b"dummy")
        with open(self.predicted_image, "wb") as f:
            f.write(b"dummy")

        # Insert into DB
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, original_image, predicted_image) VALUES (?, ?, ?)",
                (self.uid, self.original_image, self.predicted_image)
            )
            conn.execute(
                "INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                (self.uid, "cat", 0.9, "[0,0,10,10]")
            )

    def test_delete_prediction_with_delay(self):
        # Confirm files exist before deletion
        self.assertTrue(os.path.exists(self.original_image))
        self.assertTrue(os.path.exists(self.predicted_image))
        print(f"Original image exists: {self.original_image}")
        print(f"Predicted image exists: {self.predicted_image}")

        # Wait for 3 seconds so you can see the files in the uploads directory
        time.sleep(3)

        # Delete the prediction
        response = self.client.delete(f"/prediction/{self.uid}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(data["uid"], self.uid)

        # Check DB cleanup
        with sqlite3.connect(DB_PATH) as conn:
            session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (self.uid,)).fetchone()
            self.assertIsNone(session)
            obj = conn.execute("SELECT * FROM detection_objects WHERE prediction_uid = ?", (self.uid,)).fetchone()
            self.assertIsNone(obj)

        # Check file cleanup
        self.assertFalse(os.path.exists(self.original_image))
        self.assertFalse(os.path.exists(self.predicted_image))

    def tearDown(self):
        # Clean up in case test fails
        for path in [self.original_image, self.predicted_image]:
            if os.path.exists(path):
                os.remove(path)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))

if __name__ == "__main__":
    unittest.main()