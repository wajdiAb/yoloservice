import sqlite3
import bcrypt
import sys

DB_PATH = "predictions.db"

# Check for correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python add_user.py <username> <password>")
    sys.exit(1)

username = sys.argv[1]
password = sys.argv[2]

# Hash the password securely
hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Connect and insert the user
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
if cursor.fetchone():
    print(f"❌ Error: Username '{username}' already exists.")
else:
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    print(f"✅ User '{username}' created successfully.")
