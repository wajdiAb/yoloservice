from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, UTC
# from db import Base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password = Column(String, nullable=False)

class PredictionSession(Base):
    __tablename__ = "prediction_sessions"
    uid = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    original_image = Column(String)
    predicted_image = Column(String)
    username = Column(String, ForeignKey("users.username"))

class DetectionObject(Base):
    __tablename__ = "detection_objects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)
