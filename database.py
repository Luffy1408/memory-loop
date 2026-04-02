"""
Database module for DigiMemoir face recognition and conversation storage.
Uses SQLite for structured storage of known faces and conversation history.
"""

import sqlite3
import pickle
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "digimemoir.db")


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create known_faces table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            transcription TEXT NOT NULL,
            video_path TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES known_faces (id)
        )
    """)

    conn.commit()
    conn.close()


def save_known_face(name: str, face_encoding: np.ndarray, image_path: Optional[str] = None) -> int:
    """
    Save a known face to the database.

    Args:
        name: Person's name
        face_encoding: numpy array of face encoding
        image_path: Optional path to saved face image

    Returns:
        The ID of the newly inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Serialize face encoding as pickle
    face_encoding_blob = pickle.dumps(face_encoding)

    cursor.execute(
        "INSERT INTO known_faces (name, face_encoding, image_path) VALUES (?, ?, ?)",
        (name, face_encoding_blob, image_path)
    )

    person_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return person_id


def get_all_known_faces() -> List[Dict[str, Any]]:
    """
    Retrieve all known faces from the database.

    Returns:
        List of dicts with keys: id, name, face_encoding (numpy array), image_path, created_at
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM known_faces ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    faces = []
    for row in rows:
        face_encoding = pickle.loads(row["face_encoding"])
        faces.append({
            "id": row["id"],
            "name": row["name"],
            "face_encoding": face_encoding,
            "image_path": row["image_path"],
            "created_at": row["created_at"]
        })

    return faces


def get_person_by_id(person_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific person by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM known_faces WHERE id = ?", (person_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        face_encoding = pickle.loads(row["face_encoding"])
        return {
            "id": row["id"],
            "name": row["name"],
            "face_encoding": face_encoding,
            "image_path": row["image_path"],
            "created_at": row["created_at"]
        }
    return None


def get_person_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a person by name (case-insensitive)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM known_faces WHERE LOWER(name) = LOWER(?)", (name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        face_encoding = pickle.loads(row["face_encoding"])
        return {
            "id": row["id"],
            "name": row["name"],
            "face_encoding": face_encoding,
            "image_path": row["image_path"],
            "created_at": row["created_at"]
        }
    return None


def save_conversation(person_id: int, transcription: str, video_path: Optional[str] = None) -> int:
    """
    Save a conversation associated with a person.

    Args:
        person_id: ID of the person in known_faces
        transcription: The transcribed conversation text
        video_path: Optional path to saved video file

    Returns:
        The ID of the newly inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (person_id, transcription, video_path) VALUES (?, ?, ?)",
        (person_id, transcription, video_path)
    )

    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return conversation_id


def get_conversations_for_person(person_id: int, limit: int = 1) -> List[Dict[str, Any]]:
    """
    Get conversations for a specific person.

    Args:
        person_id: ID of the person
        limit: Maximum number of conversations to return (default 1 for most recent)

    Returns:
        List of dicts with keys: id, person_id, transcription, video_path, recorded_at
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM conversations WHERE person_id = ? ORDER BY recorded_at DESC LIMIT ?",
        (person_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    conversations = []
    for row in rows:
        conversations.append({
            "id": row["id"],
            "person_id": row["person_id"],
            "transcription": row["transcription"],
            "video_path": row["video_path"],
            "recorded_at": row["recorded_at"]
        })

    return conversations


def get_last_conversation_for_person(person_id: int) -> Optional[Dict[str, Any]]:
    """Get the most recent conversation for a person."""
    conversations = get_conversations_for_person(person_id, limit=1)
    return conversations[0] if conversations else None


def update_face_image(person_id: int, image_path: str):
    """Update the face image path for a person."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE known_faces SET image_path = ? WHERE id = ?",
        (image_path, person_id)
    )

    conn.commit()
    conn.close()


def delete_known_face(person_id: int):
    """Delete a known face and all associated conversations."""
    conn = get_connection()
    cursor = conn.cursor()

    # Delete conversations first (foreign key constraint)
    cursor.execute("DELETE FROM conversations WHERE person_id = ?", (person_id,))
    cursor.execute("DELETE FROM known_faces WHERE id = ?", (person_id,))

    conn.commit()
    conn.close()


# Initialize database when module is imported
init_db()
