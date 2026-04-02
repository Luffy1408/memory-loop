"""
Database module for Memory Loop face recognition and conversation storage.
Uses SQLite for structured storage of known faces and conversation history.
"""

import sqlite3
import pickle
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "memoryloop.db")


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

    # Create medical_routines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            medicine_name TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT,
            time_of_day TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

    # Order by id DESC to get most recent conversations first (id is auto-incrementing)
    cursor.execute(
        "SELECT * FROM conversations WHERE person_id = ? ORDER BY id DESC LIMIT ?",
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


def update_person_name(person_id: int, new_name: str) -> bool:
    """
    Update the name of a person in the database.

    Args:
        person_id: ID of the person to update
        new_name: New name for the person

    Returns:
        True if successful, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "UPDATE known_faces SET name = ? WHERE id = ?",
            (new_name, person_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating person name: {e}")
        conn.close()
        return False


def get_all_conversations_with_persons() -> List[Dict[str, Any]]:
    """
    Get all conversations with associated person data.

    Returns:
        List of dicts with conversation and person data
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.id, c.transcription, c.video_path, c.recorded_at,
               p.id as person_id, p.name as person_name, p.image_path as person_image
        FROM conversations c
        JOIN known_faces p ON c.person_id = p.id
        ORDER BY c.recorded_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    conversations = []
    for row in rows:
        conversations.append({
            "id": row["id"],
            "transcription": row["transcription"],
            "video_path": row["video_path"],
            "recorded_at": row["recorded_at"],
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "person_image": row["person_image"]
        })

    return conversations


def get_all_persons_with_conversation_count() -> List[Dict[str, Any]]:
    """
    Get all persons with their conversation count.

    Returns:
        List of dicts with person data and conversation count
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.id, p.name, p.image_path, p.created_at,
               COUNT(c.id) as conversation_count
        FROM known_faces p
        LEFT JOIN conversations c ON p.id = c.person_id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    persons = []
    for row in rows:
        persons.append({
            "id": row["id"],
            "name": row["name"],
            "image_path": row["image_path"],
            "created_at": row["created_at"],
            "conversation_count": row["conversation_count"]
        })

    return persons


def get_all_conversations() -> List[Dict[str, Any]]:
    """
    Get all conversations from the database.

    Returns:
        List of dicts with keys: id, person_id, transcription, video_path, recorded_at
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM conversations ORDER BY recorded_at DESC")
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


# Initialize database when module is imported
init_db()


# ============ Medical Routines Functions ============

def add_medical_routine(person_id: int, medicine_name: str, dosage: str = "",
                        frequency: str = "", time_of_day: str = "", notes: str = "") -> int:
    """
    Add a medical routine for a person.

    Args:
        person_id: ID of the person
        medicine_name: Name of the medicine
        dosage: Dosage information (e.g., "1 pill", "5ml")
        frequency: How often to take (e.g., "Daily", "Twice daily")
        time_of_day: When to take (e.g., "Morning", "After lunch", "8:00 AM")
        notes: Additional notes

    Returns:
        The ID of the newly inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO medical_routines (person_id, medicine_name, dosage, frequency, time_of_day, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (person_id, medicine_name, dosage, frequency, time_of_day, notes))

    routine_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return routine_id


def get_medical_routines(person_id: int) -> List[Dict[str, Any]]:
    """
    Get all medical routines for a person.

    Args:
        person_id: ID of the person

    Returns:
        List of dicts with medical routine data
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM medical_routines
        WHERE person_id = ?
        ORDER BY time_of_day, medicine_name
    """, (person_id,))

    rows = cursor.fetchall()
    conn.close()

    routines = []
    for row in rows:
        routines.append({
            "id": row["id"],
            "person_id": row["person_id"],
            "medicine_name": row["medicine_name"],
            "dosage": row["dosage"],
            "frequency": row["frequency"],
            "time_of_day": row["time_of_day"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        })

    return routines


def get_all_medical_routines() -> List[Dict[str, Any]]:
    """
    Get all medical routines with person names.

    Returns:
        List of dicts with medical routine and person data
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT m.id, m.person_id, m.medicine_name, m.dosage, m.frequency,
               m.time_of_day, m.notes, m.created_at, m.updated_at,
               p.name as person_name
        FROM medical_routines m
        JOIN known_faces p ON m.person_id = p.id
        ORDER BY p.name, m.time_of_day
    """)

    rows = cursor.fetchall()
    conn.close()

    routines = []
    for row in rows:
        routines.append({
            "id": row["id"],
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "medicine_name": row["medicine_name"],
            "dosage": row["dosage"],
            "frequency": row["frequency"],
            "time_of_day": row["time_of_day"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        })

    return routines


def update_medical_routine(routine_id: int, medicine_name: str = None, dosage: str = None,
                           frequency: str = None, time_of_day: str = None, notes: str = None) -> bool:
    """
    Update a medical routine.

    Args:
        routine_id: ID of the routine to update
        Other args: New values (None to keep existing)

    Returns:
        True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get current values
    cursor.execute("SELECT * FROM medical_routines WHERE id = ?", (routine_id,))
    current = cursor.fetchone()

    if not current:
        conn.close()
        return False

    # Update with new values or keep existing
    new_values = {
        "medicine_name": medicine_name if medicine_name is not None else current["medicine_name"],
        "dosage": dosage if dosage is not None else current["dosage"],
        "frequency": frequency if frequency is not None else current["frequency"],
        "time_of_day": time_of_day if time_of_day is not None else current["time_of_day"],
        "notes": notes if notes is not None else current["notes"]
    }

    cursor.execute("""
        UPDATE medical_routines
        SET medicine_name = ?, dosage = ?, frequency = ?, time_of_day = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (new_values["medicine_name"], new_values["dosage"], new_values["frequency"],
          new_values["time_of_day"], new_values["notes"], routine_id))

    conn.commit()
    conn.close()

    return True


def delete_medical_routine(routine_id: int) -> bool:
    """
    Delete a medical routine.

    Args:
        routine_id: ID of the routine to delete

    Returns:
        True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM medical_routines WHERE id = ?", (routine_id,))
    conn.commit()
    conn.close()

    return True
