# common/models.py
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class AudioTranscription(Base):
    __tablename__ = 'transcriptions'

    id = Column(Integer, primary_key=True)
    filename = Column(String(100))
    transcription = Column(String(500))
    created_at = Column(DateTime, default=datetime.now)