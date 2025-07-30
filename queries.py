from sqlalchemy.orm import Session
from models import PredictionSession, DetectionObject, User
from sqlalchemy import func, distinct
from datetime import datetime, timedelta

def save_prediction(db: Session, uid: str, original_img: str, predicted_img: str, username: str):
    row = PredictionSession(
        uid=uid,
        original_image=original_img,
        predicted_image=predicted_img,
        username=username
    )
    db.add(row)
    db.commit()

def save_detection(db: Session, uid: str, label: str, score: float, box: str):
    obj = DetectionObject(prediction_uid=uid, label=label, score=score, box=str(box))
    db.add(obj)
    db.commit()

def get_prediction(db: Session, uid: str, username: str):
    return db.query(PredictionSession).filter_by(uid=uid, username=username).first()

def get_detections(db: Session, uid: str):
    return db.query(DetectionObject).filter_by(prediction_uid=uid).all()


def get_user(db: Session, username: str) -> User | None:
    return db.query(User).filter_by(username=username).first()

def create_user(db: Session, username: str, password_hash: str) -> None:
    user = User(username=username, password=password_hash)
    db.add(user)
    db.commit()

def get_predictions_by_label(db: Session, label: str, username: str):
    return (
        db.query(PredictionSession.uid, PredictionSession.timestamp)
        .join(DetectionObject, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(
            DetectionObject.label == label,
            PredictionSession.username == username
        )
        .distinct()
        .all()
    )

def get_predictions_by_score(db: Session, min_score: float, username: str):
    return (
        db.query(PredictionSession.uid, PredictionSession.timestamp)
        .join(DetectionObject, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(
            DetectionObject.score >= min_score,
            PredictionSession.username == username
        )
        .distinct()
        .all()
    )

def is_image_owned_by_user(db: Session, path: str, username: str) -> bool:
    return db.query(PredictionSession).filter(
        PredictionSession.username == username,
        (PredictionSession.original_image == path) | (PredictionSession.predicted_image == path)
    ).first() is not None

def get_predicted_image_path(db: Session, uid: str, username: str) -> str | None:
    result = db.query(PredictionSession.predicted_image).filter_by(uid=uid, username=username).first()
    return result[0] if result else None

def count_predictions_last_week(db: Session, username: str) -> int:
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    return db.query(func.count()).select_from(PredictionSession).filter(
        PredictionSession.username == username,
        PredictionSession.timestamp >= one_week_ago
    ).scalar()

def get_unique_labels_last_week(db: Session, username: str) -> list[str]:
    one_week_ago = datetime.utcnow() - timedelta(days=7)

    results = (
        db.query(distinct(DetectionObject.label))
        .join(PredictionSession, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(
            PredictionSession.timestamp >= one_week_ago,
            PredictionSession.username == username
        )
        .all()
    )

    return [row[0] for row in results]  # row is a tuple like ('label',)

def get_prediction_file_paths(db: Session, uid: str, username: str):
    result = db.query(PredictionSession).filter_by(uid=uid, username=username).first()
    if not result:
        return None
    return result.original_image, result.predicted_image

def delete_prediction_and_detections(db: Session, uid: str, username: str):
    # First delete associated detection objects
    db.query(DetectionObject).filter_by(prediction_uid=uid).delete()
    
    # Then delete the prediction session itself
    db.query(PredictionSession).filter_by(uid=uid, username=username).delete()
    
    db.commit()

def get_user_prediction_stats(db: Session, username: str):
    one_week_ago = datetime.utcnow() - timedelta(days=7)

    # 1. Total predictions
    total_predictions = (
        db.query(func.count())
        .select_from(PredictionSession)
        .filter(
            PredictionSession.timestamp >= one_week_ago,
            PredictionSession.username == username
        )
        .scalar()
    )

    # 2. Average confidence score
    avg_confidence = (
        db.query(func.avg(DetectionObject.score))
        .join(PredictionSession, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(
            PredictionSession.timestamp >= one_week_ago,
            PredictionSession.username == username
        )
        .scalar()
    ) or 0.0

    # 3. Most frequent labels
    label_counts = (
        db.query(DetectionObject.label, func.count().label("count"))
        .join(PredictionSession, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(
            PredictionSession.timestamp >= one_week_ago,
            PredictionSession.username == username
        )
        .group_by(DetectionObject.label)
        .order_by(desc("count"))
        .limit(5)
        .all()
    )

    frequent_labels = [{"label": row.label, "count": row.count} for row in label_counts]

    return {
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence,
        "most_frequent_labels": frequent_labels
    }

    