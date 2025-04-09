
import os
from datetime import time
from sqlite3 import OperationalError
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common.models import Base, AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/data'


# Database setup
def create_db_engine():
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)

    # Retry connection
    retries = 5
    while retries > 0:
        try:
            with engine.connect() as conn:
                return engine
        except OperationalError:
            retries -= 1
            time.sleep(5)

    raise RuntimeError("Failed to connect to database after multiple attempts")


# Use this instead of direct create_engine
engine = create_db_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def process_audio(filename):
    """Mock ML processing (uppercase conversion)"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'r') as f:
        text = f.read()
    return text.upper()


@app.route('/process', methods=['POST'])
def process():
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
