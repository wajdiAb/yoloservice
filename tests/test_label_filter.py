import unittest
import sqlite3
from fastapi.testclient import TestClient
from app import app, DB_PATH

client = TestClient(app)
AUTH = ("labeluser", "labelpass")

class TestLabelFilter(unittest.TestCase):
    def setUp(self):
        self.uid = "label-test-uid"

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("""
                INSERT INTO prediction_sessions (uid, original_image, predicted_image, username) 
                VALUES (?, ?, ?, ?)
            """, (self.uid, "a.jpg", "b.jpg", AUTH[0]))
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (self.uid, "dog", 0.99, "[0,0,10,10]"))

    def test_label_match(self):
        response = client.get("/predictions/label/dog", auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()), 1)

    def test_label_not_found(self):
        response = client.get("/predictions/label/notalabel", auth=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def tearDown(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (self.uid,))
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.uid,))

if __name__ == "__main__":
    unittest.main()
