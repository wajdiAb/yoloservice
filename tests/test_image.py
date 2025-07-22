import unittest
import os
import sqlite3
from fastapi.testclient import TestClient
from app import app, DB_PATH

AUTH = ("testuser", "testpass")
client = TestClient(app)

class TestGetImageEndpoint(unittest.TestCase):
    def setUp(self):
        self.uid = "test-img-uid"
        self.original = f"uploads/original/{self.uid}.jpg"
        self.predicted = f"uploads/predicted/{self.uid}.jpg"

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        with open(self.original, "wb") as f:
            f.write(b"dummy")
        with open(self.predicted, "wb") as f:
            f.write(b"dummy")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, original_image, predicted_image, username) VALUES (?, ?, ?, ?)",
                (self.uid, self.original, self.predicted, AUTH[0])
            )

    def test_invalid_type(self):
        response = client.get(f"/image/invalid/{self.uid}.jpg", auth=AUTH)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image type")

    def test_not_found_image(self):
        response = client.get(f"/image/original/not_exists.jpg", auth=AUTH)
        self.assertEqual(response.status_code, 404)

    def test_not_owned_image(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE prediction_sessions SET username = ? WHERE uid = ?", ("otheruser", self.uid))

        response = client.get(f"/image/original/{self.uid}.jpg", auth=AUTH)
        self.assertEqual(response.status_code, 403)

    def test_valid_image_access(self):
        response = client.get(f"/image/original/{self.uid}.jpg", auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.headers["content-type"], ["image/jpeg", "application/octet-stream"])

    def tearDown(self):
        for path in [self.original, self.predicted]:
            if os.path.exists(path):
                os.remove(path)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))

if __name__ == "__main__":
    unittest.main()
