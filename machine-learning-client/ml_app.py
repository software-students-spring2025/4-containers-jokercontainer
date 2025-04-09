"""
Module for processing audio transcription via a Flask web service.
"""

import os
from time import sleep
from sqlite3 import OperationalError
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common.models import Base, AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/data'


# Database setup
def create_db_engine():
    """
        Create and return a SQLAlchemy database engine with retry logic.

        The function attempts to connect up to 5 times before failing.

        Returns:
            engine: A SQLAlchemy engine object connected to the database.

        Raises:
            RuntimeError: If unable to connect after multiple attempts.
    """
    db_url = os.getenv('DB_URL')
    local_engine = create_engine(db_url)

    # Retry connection
    retries = 5
    while retries > 0:
        try:
            with engine.connect() as _:
                return local_engine
        except OperationalError:
            retries -= 1
            sleep(5)

    raise RuntimeError("Failed to connect to database after multiple attempts")


# Use this instead of direct create_engine
engine = create_db_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def process_audio(filename):
    """
    Process the audio file by reading its content and converting it to uppercase.

    Args:
        filename (str): The name of the file to process.

    Returns:
        str: The processed text in uppercase.
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.upper()


@app.route('/process', methods=['POST'])
def process():
    """
        Process a file and store its transcription in the database.

        Expects a JSON payload with a 'filename' key.

        Returns:
            JSON response indicating success.
    """
    data = request.get_json()
    filename = data['filename']

    # Process the file
    transcription = process_audio(filename)

    # Save to database
    session = Session()
    entry = AudioTranscription(filename=filename, transcription=transcription)
    session.add(entry)
    session.commit()

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
