import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app
import numpy as np
import os

class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Create required directories
        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

    def tearDown(self):
        # Clean up test directories if they're empty
        try:
            os.rmdir("uploads/original")
            os.rmdir("uploads/predicted")
            os.rmdir("uploads")
        except OSError:
            pass  # Directory not empty or doesn't exist

    def create_mock_yolo_result(self):
        # Create a simple mock result with one detected object
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 0)]  # 0 = person
        mock_box.conf = [MagicMock(item=lambda: 0.95)]  # 95% confidence
        mock_box.xyxy = [MagicMock(tolist=lambda: [10, 20, 30, 40])]  # bounding box

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        # Create a mock numpy array for the plot result
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        return [mock_result]

    @patch('app.model')
    def test_predict_with_file(self, mock_model):
        # Setup mock model
        mock_model.return_value = self.create_mock_yolo_result()
        mock_model.names = {0: "person"}

        # Test file upload prediction
        files = {"file": ("test.jpg", b"fake_image_content", "image/jpeg")}
        response = self.client.post("/predict", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction_uid", data)
        self.assertEqual(data["detection_count"], 1)
        self.assertEqual(data["labels"], ["person"])

    @patch('app.model')
    @patch('app.AWS_S3_BUCKET', 'test-bucket')
    @patch('app.download_file')
    @patch('app.s3_key_exists')
    @patch('app.upload_file')
    @patch('app.copy_object')
    def test_predict_with_s3(self, mock_copy, mock_upload, mock_exists, mock_download, mock_model):
        # Setup mock model
        mock_model.return_value = self.create_mock_yolo_result()
        mock_model.names = {0: "person"}
        
        # Setup S3 mocks
        mock_exists.return_value = False
        mock_download.return_value = None
        mock_upload.return_value = None
        mock_copy.return_value = None

        # Test S3 image prediction
        response = self.client.post("/predict?img=test/image.jpg&chat_id=test-chat")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction_uid", data)
        self.assertEqual(data["detection_count"], 1)
        self.assertEqual(data["labels"], ["person"])
        self.assertEqual(data["s3"]["bucket"], "test-bucket")

    def test_predict_errors(self):
        # Test no input provided
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Provide a file upload or ?img=<s3_key>")

        # Test both inputs provided
        files = {"file": ("test.jpg", b"fake_image_content", "image/jpeg")}
        response = self.client.post("/predict?img=test.jpg", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Provide only one of: file OR img")

if __name__ == '__main__':
    unittest.main()
