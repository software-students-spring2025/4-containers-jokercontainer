"""
Module for database models and connection handling with MongoDB.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from bson.objectid import ObjectId


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
    Model representing Q&A pairs from audio transcription with MongoDB storage.
    """
    collection_name = "Q&A"

    @classmethod
    def get_collection(cls) -> Collection:
        """
        Get the MongoDB collection for Q&A.

        Returns:
            Collection object for Q&A
        """
        conn = MongoDBConnection()
        return conn.get_collection(cls.collection_name)

    @classmethod
    def create_indexes(cls) -> None:
        """
        Create necessary indexes for the Q&A collection.
        """
        collection = cls.get_collection()
        collection.create_index("chatid")
        collection.create_index("created_at")
        collection.create_index([("chatid", ASCENDING), ("created_at", DESCENDING)])

    @classmethod
    def create(cls, chatid: str, user_question: str = "", answer: str = "") -> str:
        """
        Create a new Q&A document in the database.

        Args:
            chatid: ID of the conversation this translation belongs to
            user_question: Transcribed question from user audio
            answer: Response generated by LLM

        Returns:
            ID of the created document
        """
        current_time = datetime.now()
        collection = cls.get_collection()
        
        # Create document with new schema
        document = {
            "chatid": chatid,
            "user_question": user_question,
            "answer": answer,
            "created_at": current_time,
            "updated_at": current_time
        }
        
        
        result = collection.insert_one(document)
        return str(result.inserted_id)

    @classmethod
    def update(cls, doc_id: str, user_question: str = None, answer: str = None) -> bool:
        """
        Update an existing Q&A document.

        Args:
            doc_id: ID of the document to update
            user_question: New user question content
            answer: New LLM response content

        Returns:
            True if update was successful, False otherwise
        """
        collection = cls.get_collection()
        update_data = {
            "updated_at": datetime.now()
        }
        
        if user_question is not None:
            update_data["user_question"] = user_question
            
        if answer is not None:
            update_data["answer"] = answer
            
        result = collection.update_one(
            {"_id": ObjectId(doc_id)}, 
            {"$set": update_data}
        )
        
        return result.modified_count > 0

    @classmethod
    def find_all(cls) -> List[Dict[str, Any]]:
        """
        Retrieve all Q&A documents from the database.

        Returns:
            List of all Q&A documents
        """
        collection = cls.get_collection()
        return list(collection.find().sort("created_at", -1))

    @classmethod
    def find_by_chatid(cls, chatid: str) -> List[Dict[str, Any]]:
        """
        Find all Q&A pairs for a specific chat.

        Args:
            chatid: ID of the chat to search for

        Returns:
            List of Q&A documents for the chat
        """
        collection = cls.get_collection()
        return list(collection.find({"chatid": chatid}).sort("created_at", -1))

    @classmethod
    def find_by_id(cls, doc_id: str) -> Dict[str, Any]:
        """
        Find a Q&A document by its ID.

        Args:
            doc_id: ID of the document to search for

        Returns:
            Q&A document or None if not found
        """
        collection = cls.get_collection()
        return collection.find_one({"_id": ObjectId(doc_id)})


# Create indexes when module is imported
try:
    AudioTranscription.create_indexes()
except Exception:
    # Connection might not be available at import time
    pass