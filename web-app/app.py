"""
Module for the web application that handles audio recording,
triggers ML translation, and provides results retrieval.
"""

import os
import uuid
import base64
from datetime import datetime
from time import sleep
from flask import Flask, request, render_template, jsonify
import requests
from pymongo.errors import ConnectionFailure
from common.models import AudioTranscription
import tempfile

app = Flask(__name__)


@app.route('/')
def index():
    """
        Render the main page of the web application.

        Returns:
            Rendered index.html template.
    """
    return render_template('index.html')


@app.route('/api/record', methods=['POST'])
def process_recording():
    """
    Process an audio recording from the browser.
    
    Receives base64 encoded audio data, saves it to a temporary file,
    and sends it to the ML service for processing.
    
    Returns:
        JSON response with processing status and chatid
    """
    # Get JSON data from request
    data = request.json
    
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Get or generate chat ID
    chatid = data.get('chatid') or str(uuid.uuid4())
    
    try:
        # Extract the base64 audio data (remove the data URL prefix if present)
        base64_audio = data['audio_data']
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',', 1)[1]
        
        # Decode the base64 data
        audio_data = base64.b64decode(base64_audio)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
        
        # Send the audio file to the ML service
        try:
            files = {'audio_file': ('recording.webm', open(temp_file_path, 'rb'), 'audio/webm')}
            data = {'chatid': chatid}
            
            response = requests.post(f"{os.getenv('ML_SERVICE_URL')}/process_audio", files=files, data=data)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # TODO: real processing
            app.logger.info(f"ML response: {response.text}")
            
            # insert the record into the database
            doc_id = AudioTranscription.create(
                chatid=chatid,
                translated_content="received"
            )
            app.logger.info(f"Created record, ID: {doc_id}, content: received")
            
            return jsonify({
                'success': True,
                'message': 'Audio sent to ML service and logged',
                'chatid': chatid,
            })
                
        except requests.RequestException as e:
            # Clean up temporary file if request failed
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
            return jsonify({
                'success': False,
                'message': f'Error communicating with ML service: {str(e)}',
                'chatid': chatid
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing audio: {str(e)}',
            'chatid': chatid
        }), 500


@app.route('/results')
def results():
    """
        Retrieve all translation results from the database.

        Returns:
            JSON response containing all translations with their metadata.
    """
    try:
        data = AudioTranscription.find_all()
        results_list = [
            {
                'id': str(item['_id']),
                'chatid': item['chatid'],
                'translated_content': item['translated_content'],
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in data
        ]
        return jsonify(results_list)
    except Exception as e:
        app.logger.error(f"Error fetching results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<chatid>')
def chat_results(chatid):
    """
        Retrieve translation results for a specific chat.

        Args:
            chatid: ID of the chat to retrieve translations for

        Returns:
            JSON response containing all translations for the specified chat.
    """
    try:
        data = AudioTranscription.find_by_chatid(chatid)
        results_list = [
            {
                'id': str(item['_id']),
                'chatid': item['chatid'],
                'translated_content': item['translated_content'],
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in data
        ]
        return jsonify(results_list)
    except Exception as e:
        app.logger.error(f"Error fetching chat results: {e}")
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
