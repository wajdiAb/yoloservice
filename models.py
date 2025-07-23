# models.py

from sqlalchemy import Column, String, DateTime, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# All models inherit from this base class
Base = declarative_base()


class PredictionSession(Base):
    """
    Model for prediction_sessions table
    
    This replaces: CREATE TABLE prediction_sessions (...)
    """
    __tablename__ = 'prediction_sessions'
    
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)