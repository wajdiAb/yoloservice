import unittest
import sqlite3
from fastapi.testclient import TestClient


class TestStatsEndpoint(unittest.TestCase):
    def setUp(self):
        from app import app, DB_PATH
        self.DB_PATH = DB_PATH
        client = TestClient(app)
        self.client = TestClient(app)

        # Clean and insert controlled test data
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects")
            conn.execute("DELETE FROM prediction_sessions")
            
            # Insert prediction sessions WITH username for authentication
            conn.execute(
                "INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image) VALUES (?, ?, datetime('now', '-2 days'), ?, ?)",
                ("uid1", "testuser", "orig1.jpg", "pred1.jpg")
            )
            conn.execute(
                "INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image) VALUES (?, ?, datetime('now', '-1 days'), ?, ?)",
                ("uid2", "testuser", "orig2.jpg", "pred2.jpg")
            )

            # Insert detection objects
            conn.execute(
                "INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                ("uid1", "cat", 0.9, "[0,0,10,10]")
            )
            conn.execute(
                "INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                ("uid1", "dog", 0.8, "[10,10,20,20]")
            )
            conn.execute(
                "INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                ("uid2", "cat", 0.95, "[5,5,15,15]")
            )

    def test_stats_endpoint(self):
        response = self.client.get("/stats", auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("total_predictions", data)
        self.assertIn("average_confidence", data)
        self.assertIn("most_frequent_labels", data)

        self.assertEqual(data["total_predictions"], 2)
        self.assertAlmostEqual(data["average_confidence"], (0.9 + 0.8 + 0.95) / 3, places=3)

        labels = {item["label"]: item["count"] for item in data["most_frequent_labels"]}
        self.assertEqual(labels["cat"], 2)
        self.assertEqual(labels["dog"], 1)

    def test_stats_with_no_predictions(self):
        username = "emptyuser"
        password = "emptypass"

        # Register new user
        response = self.client.get("/stats", auth=(username, password))
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_predictions"], 0)
        self.assertEqual(data["average_confidence"], 0.0)
        self.assertEqual(data["most_frequent_labels"], [])
        
    def test_stats_zero_objects(self):
        username = "noobjects"
        password = "nopass"

        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE username = ?", (username,))
            conn.execute("DELETE FROM detection_objects")

            conn.execute(
                "INSERT INTO prediction_sessions (uid, original_image, predicted_image, username) VALUES (?, ?, ?, ?)",
                ("zzz", "a.jpg", "b.jpg", username)
            )
            # No detection_objects inserted

        response = self.client.get("/stats", auth=(username, password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["average_confidence"], 0.0)

if __name__ == "__main__":
    unittest.main()
