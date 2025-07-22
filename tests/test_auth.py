import unittest
from fastapi.testclient import TestClient
from app import app, DB_PATH
from PIL import Image
import io
import sqlite3

client = TestClient(app)

class TestAuth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure test user is created by sending a predict request."""
        image_bytes = cls._generate_test_image()
        client.post("/predict", files={"file": image_bytes}, auth=("testuser", "testpass"))

    @staticmethod
    def _generate_test_image():
        """Generate a simple in-memory PNG image."""
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return ("test.png", buf, "image/png")

    def test_protected_endpoint_requires_auth(self):
        response = client.get("/prediction/test-uid")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_get_prediction_unauthenticated(self):
        response = client.get("/prediction/some-uid")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_get_prediction_wrong_credentials(self):
        response = client.get("/prediction/some-uid", auth=("testuser", "wrongpass"))
        self.assertIn(response.status_code, (401, 403))

    def test_get_prediction_nonexistent_uid(self):
        response = client.get("/prediction/nonexistent-uid", auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found or not authorized")

    def test_get_prediction_valid(self):
        image_bytes = self._generate_test_image()
        predict_resp = client.post("/predict", files={"file": image_bytes}, auth=("testuser", "testpass"))
        self.assertEqual(predict_resp.status_code, 200)
        uid = predict_resp.json()["prediction_uid"]

        get_resp = client.get(f"/prediction/{uid}", auth=("testuser", "testpass"))
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["uid"], uid)

    def test_predict_wrong_password_after_registration(self):
        image_bytes = self._generate_test_image()
        response = client.post("/predict", files={"file": image_bytes}, auth=("testuser", "wrongpass"))
        self.assertEqual(response.status_code, 401)

    def test_get_predictions_by_label_requires_auth(self):
        response = client.get("/predictions/label/person")
        self.assertEqual(response.status_code, 401)

    def test_status_endpoint_no_auth(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    # ----------------------- New Tests for Full Coverage -----------------------

    def test_malformed_authorization_header(self):
        """Simulate malformed Authorization header (not Basic)."""
        headers = {
            "Authorization": "Bearer sometoken"
        }
        response = client.post("/predict", headers=headers)
        self.assertEqual(response.status_code, 422)  # FastAPI throws 422 on malformed headers

    def test_first_time_user_registration(self):
        """Register a new user by calling predict."""
        image_bytes = self._generate_test_image()
        response = client.post("/predict", files={"file": image_bytes}, auth=("newuser123", "newpass"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "newuser123")

    def test_existing_user_correct_password(self):
        """Test login with valid password after registration."""
        response = client.get("/labels", auth=("testuser", "testpass"))
        self.assertIn(response.status_code, (200, 204))
        self.assertIn("labels", response.json())

    def test_predict_without_auth_optional(self):
        """Check that prediction works without auth (username should be null)."""
        image_bytes = self._generate_test_image()
        response = client.post("/predict", files={"file": image_bytes})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["username"])
    
    def test_get_prediction_other_user(self):
        # Create a prediction with another user
        uid = "foreign-uid"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, original_image, predicted_image, username) VALUES (?, ?, ?, ?)",
                (uid, "x.jpg", "y.jpg", "someoneelse")
            )

        response = client.get(f"/prediction/{uid}", auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
