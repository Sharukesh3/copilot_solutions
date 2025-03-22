import sqlite3

DATABASE = "tokens.db"

def init_db():
    """Initialize the database and create tokens table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_token(token: str):
    """Save a generated token into the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tokens (token) VALUES (?)", (token,))
    conn.commit()
    conn.close()

def get_tokens():
    """Retrieve all tokens from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, token, created_at FROM tokens")
    tokens = cursor.fetchall()
    conn.close()
    return tokens

def delete_token(token_id: int):
    """Delete a token by its ID."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()
