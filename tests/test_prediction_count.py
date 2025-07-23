import unittest
from fastapi.testclient import TestClient
import sqlite3
from datetime import datetime, timedelta, UTC

DB_PATH = "predictions.db"

class TestPredictionCount(unittest.TestCase):
    def setUp(self):
        from app import app

        self.client = TestClient(app)

        # Clean and insert controlled test data with username
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions")

            # Insert a recent prediction (within 7 days)
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "recent-id", 
                "testuser",
                datetime.now(UTC).isoformat(), 
                "recent_original.jpg", 
                "recent_predicted.jpg"
            ))

            # Insert an old prediction (more than 7 days ago)
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "old-id", 
                "testuser",
                (datetime.now(UTC) - timedelta(days=10)).isoformat(), 
                "old_original.jpg", 
                "old_predicted.jpg"
            ))

    def test_prediction_count_format(self):
        """Check response format and status"""
        response = self.client.get("/predictions/count", auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("count", data)
        self.assertIsInstance(data["count"], int)
        self.assertGreaterEqual(data["count"], 0)

    def test_prediction_count_last_7_days(self):
        """Ensure only recent predictions are counted"""
        response = self.client.get("/predictions/count", auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)  # Only the recent one should be counted

if __name__ == "__main__":
    unittest.main()
