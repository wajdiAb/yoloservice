# db.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from models import User, PredictionSession, DetectionObject
from models import Base
from dotenv import load_dotenv

load_dotenv()

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")

if DB_BACKEND == "postgres":
    # prefer env if provided; otherwise default to the docker service name 'postgres'
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:pass@postgres:5432/predictions")

else:
    DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    if DB_BACKEND == "postgres":
        print("Creating tables in Postgres...")
        Base.metadata.create_all(bind=engine)