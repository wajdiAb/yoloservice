# init_db.py

from db import engine
from models import Base

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

print("✅ Database initialized!")
