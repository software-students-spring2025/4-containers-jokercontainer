"""
Module for the web application that handles file uploads,
triggers ML processing, and provides results retrieval.
"""

import os
from datetime import datetime
from time import sleep
from flask import Flask, request, render_template, jsonify
import requests
from pymongo.errors import ConnectionFailure
from common.models import AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/data'


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
    try:
        data = AudioTranscription.find_all()
        results_list = [
            {
                'filename': item['filename'],
                'text': item['transcription'],
                'time': item['created_at'].isoformat() if 'created_at' in item else None
            }
            for item in data
        ]
        return jsonify(results_list)
    except Exception as e:
        app.logger.error(f"Error fetching results: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Wait for a few seconds to ensure MongoDB is ready
    sleep(5)
    
    # Try to initialize database connection and indexes
    retries = 5
    while retries > 0:
        try:
            AudioTranscription.create_indexes()
            app.logger.info("Successfully connected to MongoDB and created indexes")
            break
        except ConnectionFailure as e:
            app.logger.warning(f"Failed to connect to MongoDB, retrying... ({retries} attempts left)")
            retries -= 1
            if retries == 0:
                app.logger.error(f"Could not connect to MongoDB: {e}")
            sleep(5)
    
    app.run(host='0.0.0.0', port=5001)
