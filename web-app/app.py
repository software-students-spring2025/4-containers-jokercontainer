"""
Module for the web application that handles file uploads,
triggers ML processing, and provides results retrieval.
"""

import os
from datetime import datetime
from time import sleep
from sqlite3 import OperationalError
from flask import Flask, request, render_template, jsonify
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common.models import Base, AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/data'


# Database setup
def create_db_engine():
    """
        Create and return a SQLAlchemy database engine with retry logic.

        Attempts to connect to the database up to 5 times before failing.

        Returns:
            engine: A SQLAlchemy engine object.

        Raises:
            RuntimeError: If unable to connect after multiple attempts.
    """
    db_url = os.getenv('DB_URL')
    local_engine  = create_engine(db_url)

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


@app.route('/')
def index():
    """
        Render the main page of the web application.

        Returns:
            Rendered index.html template.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
        Handle file uploads, save the file, and trigger ML processing.

        Expects form data with a 'text' field. The file is saved with a unique
        timestamp-based filename, and a POST request is sent to the ML processing service.

        Returns:
            JSON response indicating that the file is being processed.
    """
    text = request.form.get('text')
    filename = f"audio_{datetime.now().timestamp()}.txt"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the file with explicit encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Trigger ML processing
    ml_service_url = os.getenv('ML_SERVICE_URL')
    requests.post(ml_service_url, json={'filename': filename})

    return jsonify({'status': 'processing'})


@app.route('/results')
def results():
    """
        Retrieve processed transcription results from the database.

        Returns:
            JSON response containing the filename, transcription text, and creation timestamp for each entry.
    """
    session = Session()
    data = session.query(AudioTranscription).all()
    return jsonify([
        {
            'filename': item.filename,
            'text': item.transcription,
            'time': item.created_at.isoformat() if item.created_at else None
        }
        for item in data
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
