import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import logging
from pathlib import Path


@dataclass
class Dataset:
    """Represents a dataset in the system"""

    name: str
    index_name: str
    system_prompt: str
    created_at: datetime = None


class DatasetRepository:
    """Repository for managing dataset metadata in SQLite"""

    def __init__(self, db_path: str = "datasets.db"):
        """Initialize the repository with the database path

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        name TEXT PRIMARY KEY,
                        index_name TEXT NOT NULL,
                        system_prompt TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize database: {e}")
            raise

    def upsert_dataset(self, dataset: Dataset) -> None:
        """Add or update a dataset in the repository
        
        Args:
            dataset: Dataset to add or update
            
        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO datasets (name, index_name, system_prompt)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        index_name = excluded.index_name,
                        system_prompt = excluded.system_prompt,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (dataset.name, dataset.index_name, dataset.system_prompt)
                )
        except sqlite3.Error as e:
            logging.error(f"Failed to upsert dataset: {e}")
            raise

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Retrieve a dataset by name

        Args:
            name: Name of the dataset to retrieve

        Returns:
            Dataset if found, None otherwise

        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT name, index_name, system_prompt, created_at
                    FROM datasets WHERE name = ?
                    """,
                    (name,),
                ).fetchone()

                if row:
                    return Dataset(
                        name=row["name"],
                        index_name=row["index_name"],
                        system_prompt=row["system_prompt"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                return None
        except sqlite3.Error as e:
            logging.error(f"Failed to get dataset: {e}")
            raise

    def list_datasets(self) -> Dict[str, Dataset]:
        """Retrieve all datasets

        Returns:
            List of all datasets

        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT name, index_name, system_prompt, created_at FROM datasets"
                ).fetchall()

                if not rows:
                    return {}

                return {
                    row["name"]: Dataset(
                        name=row["name"],
                        index_name=row["index_name"],
                        system_prompt=row["system_prompt"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    for row in rows
                }
        except sqlite3.Error as e:
            logging.error(f"Failed to list datasets: {e}")
            raise

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset by name

        Args:
            name: Name of the dataset to delete

        Returns:
            True if dataset was deleted, False if not found

        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Failed to delete dataset: {e}")
            raise
