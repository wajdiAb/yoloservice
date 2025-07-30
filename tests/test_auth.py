import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import numpy as np
import bcrypt

from app import app, get_current_username, get_optional_username, get_db


class TestAuthWithMocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup TestClient and default dependency overrides (no real DB/auth)."""
        cls.client = TestClient(app)

        def override_get_db():
            # Mock SQLAlchemy Session for every test
            yield MagicMock()

        # Default dependencies: treat the user as authenticated "testuser"
        app.dependency_overrides[get_current_username] = lambda: "testuser"
        app.dependency_overrides[get_optional_username] = lambda: "testuser"
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        """Reset overrides after tests."""
        app.dependency_overrides = {}

    # -----------------------------
    # Utility helpers
    # -----------------------------
    def _patch_yolo_with_results(self, detections=None):
        """
        Patch YOLO model and DB writers used in /predict.
        detections: list of MagicMock boxes; default is one box with label 'person'.
        Returns (patchers, mocks) so caller can stop them in finally.
        """
        if detections is None:
            fake_box = MagicMock()
            fake_cls, fake_conf, fake_bbox = MagicMock(), MagicMock(), MagicMock()
            fake_cls.item.return_value = 0          # -> label index
            fake_conf.item.return_value = 0.8       # -> confidence
            fake_bbox.tolist.return_value = [0, 0, 100, 100]

            fake_box.cls = [fake_cls]
            fake_box.conf = [fake_conf]
            fake_box.xyxy = [fake_bbox]
            detections = [fake_box]

        patchers = [
            patch("app.model"),            # YOLO runner
            patch("app.save_prediction"),  # DB insert (session)
            patch("app.save_detection"),   # DB insert (detections)
        ]
        mocks = [p.start() for p in patchers]

        mock_model = mocks[0]
        fake_results = MagicMock()
        fake_results.__getitem__.return_value = fake_results
        fake_results.boxes = detections
        fake_results.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        mock_model.return_value = [fake_results]
        mock_model.names = {0: "person"}  # map label index -> name

        return patchers, mocks

    # -----------------------------
    # Endpoint tests (mocked)
    # -----------------------------

    @patch("app.get_detections")
    @patch("app.get_prediction")
    def test_get_prediction_valid(self, mock_get_prediction, mock_get_detections):
        """GET /prediction/{uid} returns prediction when found for user."""
        mock_obj = MagicMock(
            uid="mocked-uid",
            timestamp="now",
            original_image="x.jpg",
            predicted_image="y.jpg"
        )
        mock_get_prediction.return_value = mock_obj
        mock_get_detections.return_value = [
            MagicMock(id=1, label="person", score=0.9, box=[0, 0, 100, 100])
        ]

        resp = self.client.get("/prediction/mocked-uid")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["uid"], "mocked-uid")
        self.assertEqual(len(data["detection_objects"]), 1)

    @patch("app.get_prediction", return_value=None)
    def test_get_prediction_not_found(self, _):
        """GET /prediction/{uid} -> 404 when not found or not authorized."""
        resp = self.client.get("/prediction/invalid-uid")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "Prediction not found or not authorized")

    def test_predict_with_detection(self):
        """/predict with mocked YOLO returning one detection."""
        patchers, _ = self._patch_yolo_with_results()
        try:
            resp = self.client.post(
                "/predict",
                files={"file": ("test.png", b"fakebytes", "image/png")}
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["detection_count"], 1)
            self.assertIn("person", data["labels"])
            self.assertIn("time_took", data)
            self.assertIsInstance(data["time_took"], (int, float))
        finally:
            for p in patchers:
                p.stop()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_predict_no_authentication(self):
        """
        POST /predict with NO Authorization header.
        Temporarily override get_optional_username -> None so username is null.
        """
        from app import app as _app, get_optional_username as _get_opt_user
        orig_override = _app.dependency_overrides.get(_get_opt_user)
        _app.dependency_overrides[_get_opt_user] = lambda: None

        try:
            patchers, _ = self._patch_yolo_with_results(detections=[])
            try:
                resp = self.client.post(
                    "/predict",
                    files={"file": ("img.jpg", b"fake", "image/jpeg")}
                )
                self.assertEqual(resp.status_code, 200)
                self.assertIsNone(resp.json()["username"])
                self.assertEqual(resp.json()["detection_count"], 0)
            finally:
                for p in patchers:
                    p.stop()
        finally:
            # restore original override
            if orig_override is None:
                _app.dependency_overrides.pop(_get_opt_user, None)
            else:
                _app.dependency_overrides[_get_opt_user] = orig_override

    def test_predict_invalid_auth_header(self):
        """
        POST /predict with malformed Basic header. Remove override so HTTPBasic runs.
        """
        from app import app as _app, get_optional_username as _get_opt_user
        orig_override = _app.dependency_overrides.pop(_get_opt_user, None)
        try:
            resp = self.client.post(
                "/predict",
                headers={"Authorization": "Basic invalid_base64"},
                files={"file": ("img.jpg", b"x", "image/jpeg")}
            )
            self.assertEqual(resp.status_code, 401)  # HTTPBasic rejects invalid base64
        finally:
            if orig_override is not None:
                _app.dependency_overrides[_get_opt_user] = orig_override

    # -----------------------------
    # Direct function tests to cover auth branches (mocked DB & bcrypt)
    # -----------------------------

    def test_get_current_username_new_user_created(self):
        """
        Covers: user is None => create_user(...) and return username.
        (Lines ~52-54)
        """
        fake_db = MagicMock()
        fake_credentials = MagicMock(username="newuser", password="testpass")

        with patch("app.get_user", return_value=None), \
             patch("app.create_user") as mock_create:
            username = get_current_username(fake_credentials, fake_db)
            self.assertEqual(username, "newuser")
            mock_create.assert_called_once()

    def test_get_current_username_invalid_password(self):
        """
        Covers: user exists but bcrypt.checkpw -> False => HTTP 401.
        (Line ~57)
        """
        fake_user = MagicMock()
        # It's enough to present a password string; we patch bcrypt.checkpw below.
        fake_user.password = "hash_doesnt_matter_here"
        fake_credentials = MagicMock(username="testuser", password="wrong")

        with patch("app.get_user", return_value=fake_user), \
             patch("bcrypt.checkpw", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                get_current_username(fake_credentials, MagicMock())
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("Invalid username or password", ctx.exception.detail)

    def test_get_optional_username_no_auth_header(self):
        """
        Covers: no Authorization header => return None.
        (Line ~76)
        """
        # Temporarily remove override to call the real function
        from app import app as _app, get_optional_username as _get_opt_user
        orig_override = _app.dependency_overrides.pop(_get_opt_user, None)
        try:
            fake_request = MagicMock()
            fake_request.headers = {}
            result = asyncio.run(get_optional_username(fake_request, MagicMock()))
            self.assertIsNone(result)
        finally:
            if orig_override is not None:
                _app.dependency_overrides[_get_opt_user] = orig_override

    def test_get_optional_username_http_exception(self):
        """
        Covers: security(request) raises HTTPException => propagate.
        """
        # Temporarily remove override to call the real function
        from app import app as _app, get_optional_username as _get_opt_user
        orig_override = _app.dependency_overrides.pop(_get_opt_user, None)
        try:
            fake_request = MagicMock()
            fake_request.headers = {"authorization": "Basic something"}
            with patch("app.security", side_effect=HTTPException(status_code=401)):
                with self.assertRaises(HTTPException):
                    asyncio.run(get_optional_username(fake_request, MagicMock()))
        finally:
            if orig_override is not None:
                _app.dependency_overrides[_get_opt_user] = orig_override

    def test_get_optional_username_valid_auth(self):
        """Covers: security(request) succeeds -> calls get_current_username and returns username."""
        from app import get_optional_username

        fake_request = MagicMock()
        fake_request.headers = {"authorization": "Basic goodtoken"}

        # Use AsyncMock for awaitable security
        fake_credentials = MagicMock()
        with patch("app.security", AsyncMock(return_value=fake_credentials)), \
            patch("app.get_current_username", return_value="validuser") as mock_get_user:
            result = asyncio.run(get_optional_username(fake_request, MagicMock()))

        self.assertEqual(result, "validuser")
        mock_get_user.assert_called_once_with(fake_credentials, unittest.mock.ANY)
