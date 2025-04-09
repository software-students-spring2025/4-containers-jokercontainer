import os
from datetime import datetime, time
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    text = request.form.get('text')
    filename = f"audio_{datetime.now().timestamp()}.txt"

    # Save mock audio file (text)
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'w') as f:
        f.write(text)

    # Trigger ML processing
    requests.post(os.getenv('ML_SERVICE_URL'), json={'filename': filename})

    return jsonify({'status': 'processing'})


@app.route('/results')
def results():
    session = Session()
    data = session.query(AudioTranscription).all()
    return jsonify([{
        'filename': item.filename,
        'text': item.transcription,
        'time': item.created_at.isoformat()
    } for item in data])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
