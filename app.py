from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import time
import logging
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
import bcrypt

from sqlalchemy.orm import Session
from db import get_db, init_db
from queries import save_prediction, save_detection, get_prediction,  get_detections
from models import User, PredictionSession, DetectionObject
from queries import get_user, create_user, get_predictions_by_label, get_predictions_by_score, is_image_owned_by_user
from queries import get_predicted_image_path, count_predictions_last_week, get_unique_labels_last_week, get_prediction_file_paths, delete_prediction_and_detections
from queries import get_user_prediction_stats

security = HTTPBasic()

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()
init_db()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB).
model = YOLO("yolov8n.pt")  


def get_current_username(
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> str:
    username = credentials.username
    password = credentials.password.encode()

    user = get_user(db, username)

    if user is None:
        # Register new user
        hashed_pw = bcrypt.hashpw(password, bcrypt.gensalt()).decode()
        create_user(db, username, hashed_pw)
        return username

    if not bcrypt.checkpw(password, user.password.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return username


async def get_optional_username(
    request: Request,
    db: Session = Depends(get_db)
) -> str | None:
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Basic "):
        return None

    try:
        credentials = await security(request)
        return get_current_username(credentials, db)
    except HTTPException:
        raise



@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    username: str = Depends(get_optional_username),
    db: Session = Depends(get_db)  # ✅ Inject the database session
):
    """
    Predict objects in an image — optional authentication (username = null if not provided)
    """
    start_time = time.time()
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu")

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    # ✅ Save using SQLAlchemy
    save_prediction(db, uid, original_path, predicted_path, username)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        # ✅ Save detection using SQLAlchemy
        save_detection(db, uid, label, score, bbox)
        detected_labels.append(label)

    processing_time = time.time() - start_time

    return {
        "prediction_uid": uid,
        "username": username,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time
    }


@app.get("/prediction/{uid}")
def get_prediction_by_uid(
    uid: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get prediction session by uid with all detected objects (only if it belongs to the user)
    """
    session = get_prediction(db, uid, username)
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found or not authorized")

    objects = get_detections(db, uid)

    return {
        "uid": session.uid,
        "timestamp": session.timestamp,
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box
            } for obj in objects
        ]
    }

@app.get("/predictions/label/{label}")
def predictions_by_label(
    label: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get prediction sessions containing objects with specified label
    """
    rows = get_predictions_by_label(db, label, username)
    return [{"uid": row.uid, "timestamp": row.timestamp} for row in rows]



@app.get("/predictions/score/{min_score}")
def predictions_by_score(
    min_score: float,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get the authenticated user's prediction sessions containing objects with score >= min_score
    """
    rows = get_predictions_by_score(db, min_score, username)
    return [{"uid": row.uid, "timestamp": row.timestamp} for row in rows]


@app.get("/image/{type}/{filename}")
def get_image(
    type: str,
    filename: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get image by type and filename — only if it belongs to the authenticated user
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    path = os.path.join("uploads", type, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    # ✅ Check ownership using SQLAlchemy
    if not is_image_owned_by_user(db, path, username):
        raise HTTPException(status_code=403, detail="Not authorized to access this image")

    return FileResponse(path)


@app.get("/prediction/{uid}/image")
def get_prediction_image(
    uid: str,
    request: Request,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get prediction image by uid (only if it belongs to the authenticated user)
    """
    accept = request.headers.get("accept", "")

    image_path = get_predicted_image_path(db, uid, username)

    if not image_path:
        raise HTTPException(status_code=404, detail="Prediction not found or not authorized")

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=406, detail="Client does not accept an image format")
    

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/predictions/count")
def get_prediction_count_last_week(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get the number of predictions made in the last 7 days
    """
    count = count_predictions_last_week(db, username)
    return {"count": count}

@app.get("/labels")
def get_unique_labels_last_week_route(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get all unique object labels detected in the last 7 days
    """
    labels = get_unique_labels_last_week(db, username)
    return {"labels": labels}

def safe_delete_file(path: str):
    logger = logging.getLogger(__name__)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to delete file: {path}. Error: {e}")


@app.delete("/prediction/{uid}")
def delete_prediction(
    uid: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Delete a specific prediction and clean up associated files.
    Removes prediction from database and deletes original and predicted image files.
    """
    paths = get_prediction_file_paths(db, uid, username)
    if not paths:
        raise HTTPException(status_code=404, detail="Prediction not found")

    original_image, predicted_image = paths

    # Delete from DB
    delete_prediction_and_detections(db, uid, username)

    # Delete associated files
    for path in [original_image, predicted_image]:
        safe_delete_file(path)

    return {"status": "deleted", "uid": uid}


@app.get("/stats")
def get_stats_last_week(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Get analytics about predictions made by the authenticated user in the last 1 week:
    - Total predictions
    - Average confidence score
    - Most frequently detected labels
    """
    return get_user_prediction_stats(db, username)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
