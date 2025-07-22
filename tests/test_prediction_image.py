import unittest
import os
import sqlite3
from fastapi.testclient import TestClient
from app import app, DB_PATH

client = TestClient(app)
AUTH = ("testuser", "testpass")

class TestPredictionImage(unittest.TestCase):
    def setUp(self):
        self.uid = "img-uid"
        self.predicted_image = f"uploads/predicted/{self.uid}.jpg"

        os.makedirs("uploads/predicted", exist_ok=True)

        with open(self.predicted_image, "wb") as f:
            f.write(b"fake")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (self.uid, self.predicted_image, AUTH[0])
            )

    def test_get_prediction_image_png(self):
        response = client.get(f"/prediction/{self.uid}/image", headers={"Accept": "image/png"}, auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_get_prediction_image_jpeg(self):
        response = client.get(f"/prediction/{self.uid}/image", headers={"Accept": "image/jpeg"}, auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertIn("image", response.headers["content-type"])

    def test_get_prediction_image_not_acceptable(self):
        response = client.get(f"/prediction/{self.uid}/image", headers={"Accept": "application/json"}, auth=AUTH)
        self.assertEqual(response.status_code, 406)
        self.assertEqual(response.json()["detail"], "Client does not accept an image format")

    def test_get_prediction_image_not_found(self):
        response = client.get("/prediction/nonexistent-uid/image", headers={"Accept": "image/png"}, auth=AUTH)
        self.assertEqual(response.status_code, 404)

    def tearDown(self):
        if os.path.exists(self.predicted_image):
            os.remove(self.predicted_image)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
    
    def test_prediction_image_file_missing(self):
        os.remove(self.predicted_image)  # Delete manually
        response = client.get(f"/prediction/{self.uid}/image", headers={"Accept": "image/png"}, auth=AUTH)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Predicted image file not found")


if __name__ == "__main__":
    unittest.main()
