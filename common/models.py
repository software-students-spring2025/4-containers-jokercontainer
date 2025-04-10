"""
Module for database models and connection handling with MongoDB.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


class MongoDBConnection:
    """
    Singleton class to handle MongoDB connection and provide access to collections.
    """
    _instance = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self.connect()

    def connect(self) -> None:
        """
        Establish connection to MongoDB using environment variables.
        """
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/mydb")
        self._client = MongoClient(mongo_uri)
        # Extract database name from URI
        db_name = mongo_uri.split("/")[-1]
        self._db = self._client[db_name]

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection by name.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            Collection object for the specified collection
        """
        if self._db is None:
            self.connect()
        return self._db[collection_name]


class AudioTranscription:
    """
    Model representing audio transcription data with MongoDB storage.
    """
    collection_name = "transcriptions"

    @classmethod
    def get_collection(cls) -> Collection:
        """
        Get the MongoDB collection for audio transcriptions.

        Returns:
            Collection object for transcriptions
        """
        conn = MongoDBConnection()
        return conn.get_collection(cls.collection_name)

    @classmethod
    def create_indexes(cls) -> None:
        """
        Create necessary indexes for the transcriptions collection.
        """
        collection = cls.get_collection()
        collection.create_index("filename")
        collection.create_index("created_at")

    @classmethod
    def create(cls, filename: str, transcription: str) -> str:
        """
        Create a new transcription document in the database.

        Args:
            filename: Name of the audio file
            transcription: Text transcription of the audio

        Returns:
            ID of the created document
        """
        collection = cls.get_collection()
        document = {
            "filename": filename,
            "transcription": transcription,
            "created_at": datetime.now()
        }
        result = collection.insert_one(document)
        return str(result.inserted_id)

    @classmethod
    def find_all(cls) -> List[Dict[str, Any]]:
        """
        Retrieve all transcription documents from the database.

        Returns:
            List of all transcription documents
        """
        collection = cls.get_collection()
        return list(collection.find().sort("created_at", -1))

    @classmethod
    def find_by_filename(cls, filename: str) -> Dict[str, Any]:
        """
        Find a transcription document by filename.

        Args:
            filename: Name of the file to search for

        Returns:
            Transcription document or None if not found
        """
        collection = cls.get_collection()
        return collection.find_one({"filename": filename})


# Create indexes when module is imported
try:
    AudioTranscription.create_indexes()
except Exception:
    # Connection might not be available at import time
    pass