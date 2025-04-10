"""
Module for processing audio transcription via a Flask web service.
"""

import os
from time import sleep
from flask import Flask, request, jsonify
from pymongo.errors import ConnectionFailure
from common.models import AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/data'


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
    try:
        transcript_id = AudioTranscription.create(
            filename=filename, 
            transcription=transcription
        )
        return jsonify({'status': 'success', 'id': transcript_id})
    except Exception as e:
        app.logger.error(f"Error saving to database: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
