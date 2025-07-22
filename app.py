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

security = HTTPBasic()

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB).
model = YOLO("yolov8n.pt")  

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT,
                username TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Users table for basic authentication
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)

        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")


init_db()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password.encode()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row is None:
            # User does not exist → Register new user
            hashed_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            return username

        # User exists → Check password
        stored_hashed_pw = row[0].encode()
        if not bcrypt.checkpw(password, stored_hashed_pw):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

        return username

def save_prediction_session(uid, original_image, predicted_image, username=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image, username)
            VALUES (?, ?, ?, ?)
        """, (uid, original_image, predicted_image, username))


def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))


async def get_optional_username(request: Request):
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Basic "):
        return None

    try:
        credentials = await security(request)
        # Sync function in thread-safe way
        return get_current_username(credentials)
    except HTTPException:
        raise



@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    username: str = Depends(get_optional_username)
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

    # Save prediction session with username (can be None)
    save_prediction_session(uid, original_path, predicted_path, username)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
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
def get_prediction_by_uid(uid: str, username: str = Depends(get_current_username)):
    """
    Get prediction session by uid with all detected objects (only if it belongs to the user)
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Get user-owned prediction session
        session = conn.execute(
            "SELECT * FROM prediction_sessions WHERE uid = ? AND username = ?",
            (uid, username)
        ).fetchone()

        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found or not authorized")

        # Get all detection objects
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()

        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, username: str = Depends(get_current_username)):
    """
    Get prediction sessions containing objects with specified label
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ? AND ps.username = ?
        """, (label, username)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float, username: str = Depends(get_current_username)):
    """
    Get the authenticated user's prediction sessions containing objects with score >= min_score
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ? AND ps.username = ?
        """, (min_score, username)).fetchall()

        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str, username: str = Depends(get_current_username)):
    """
    Get image by type and filename — only if it belongs to the authenticated user
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    path = os.path.join("uploads", type, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if this image belongs to the current user
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM prediction_sessions 
            WHERE (original_image = ? OR predicted_image = ?) AND username = ?
        """, (path, path, username))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Not authorized to access this image")

    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request, username: str = Depends(get_current_username)):
    """
    Get prediction image by uid (only if it belongs to the authenticated user)
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ? AND username = ?",
            (uid, username)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found or not authorized")

        image_path = row[0]

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
def get_prediction_count_last_week(username: str = Depends(get_current_username)):
    """
    Get the number of predictions made in the last 7 days
    """
    with sqlite3.connect(DB_PATH) as conn:
        
        cursor = conn.execute("""
            SELECT COUNT(*) FROM prediction_sessions
            WHERE timestamp >= datetime('now', '-7 days')
            AND username = ?
        """, (username,))
        count = cursor.fetchone()[0]
    return {"count": count}

@app.get("/labels")
def get_unique_labels_last_week(username: str = Depends(get_current_username)):
    """
    Get all unique object labels detected in the last 7 days
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT DISTINCT do.label
            FROM detection_objects do
            JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
            WHERE ps.timestamp >= datetime('now', '-7 days')
            AND username = ?
        """, (username,))
        labels = [row["label"] for row in cursor.fetchall()]
    return {"labels": labels}


def safe_delete_file(path: str):
    logger = logging.getLogger(__name__)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to delete file: {path}. Error: {e}")


@app.delete("/prediction/{uid}")
def delete_prediction(uid: str, username: str = Depends(get_current_username)):
    """
    Delete a specific prediction and clean up associated files.
    Removes prediction from database and deletes original and predicted image files.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # Get image file paths before deleting
        row = conn.execute(
            "SELECT original_image, predicted_image FROM prediction_sessions WHERE uid = ? AND username = ?",
            (uid, username)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        original_image, predicted_image = row

        # Delete detection objects
        conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (uid,))
        # Delete prediction session
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ? AND username = ?", (uid, username))


    # Remove image files if they exist
    for path in [original_image, predicted_image]:
        safe_delete_file(path)

    return {"status": "deleted", "uid": uid}

@app.get("/stats")
def get_stats_last_week(username: str = Depends(get_current_username)):
    """
    Get analytics about predictions made by the authenticated user in the last 1 week:
    - Total predictions
    - Average confidence score
    - Most frequently detected labels
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Total predictions by this user
        total_predictions = conn.execute("""
            SELECT COUNT(*) as count
            FROM prediction_sessions
            WHERE timestamp >= datetime('now', '-7 days') AND username = ?
        """, (username,)).fetchone()["count"]

        # Average confidence score for this user's predictions
        avg_conf_row = conn.execute("""
            SELECT AVG(do.score) as avg_score
            FROM detection_objects do
            JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
            WHERE ps.timestamp >= datetime('now', '-7 days') AND ps.username = ?
        """, (username,)).fetchone()
        avg_confidence = avg_conf_row["avg_score"] if avg_conf_row and avg_conf_row["avg_score"] is not None else 0.0

        # Most frequent labels for this user
        freq_labels = conn.execute("""
            SELECT do.label, COUNT(*) as count
            FROM detection_objects do
            JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
            WHERE ps.timestamp >= datetime('now', '-7 days') AND ps.username = ?
            GROUP BY do.label
            ORDER BY count DESC
            LIMIT 5
        """, (username,)).fetchall()
        most_frequent_labels = [{"label": row["label"], "count": row["count"]} for row in freq_labels]

    return {
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence,
        "most_frequent_labels": most_frequent_labels
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
