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
        app.logger.warning("No audio data provided in request")
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Get or generate chat ID
    chatid = data.get('chatid') or str(uuid.uuid4())
    app.logger.info(f"Processing recording for chatid: {chatid}")
    
    try:
        # Extract the base64 audio data (remove the data URL prefix if present)
        base64_audio = data['audio_data']
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',', 1)[1]
        
        app.logger.info(f"Received base64 audio data, length: {len(base64_audio)} chars")
        
        # Decode the base64 data
        audio_data = base64.b64decode(base64_audio)
        app.logger.info(f"Decoded audio data, size: {len(audio_data)} bytes")
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
            app.logger.info(f"Saved audio data to temporary file: {temp_file_path}")
        
        # Send the audio file to the ML service
        try:
            app.logger.info(f"Preparing to send audio to ML service at: {os.getenv('ML_SERVICE_URL')}/process_audio")
            
            files = {'audio_file': ('recording.webm', open(temp_file_path, 'rb'), 'audio/webm')}
            data = {'chatid': chatid}
            
            app.logger.info(f"Sending request to ML service with chatid: {chatid}")
            response = requests.post(f"{os.getenv('ML_SERVICE_URL')}/process_audio", files=files, data=data)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            app.logger.info(f"Removed temporary file: {temp_file_path}")
            
            # Check response status
            if response.status_code == 200:
                app.logger.info(f"ML service response: {response.status_code} {response.text[:100]}...")
                
                # Parse the response JSON to see if the ML service was successful
                ml_response = response.json()
                if ml_response.get('status') == 'success':
                    app.logger.info(f"ML service successfully processed audio: {ml_response.get('question', '')[:50]}...")
                    
                    # Don't create a duplicate database entry - the ML service already created one
                    return jsonify({
                        'success': True,
                        'message': 'Audio processed successfully by ML service',
                        'chatid': chatid,
                        'question': ml_response.get('question'),
                        'answer': ml_response.get('answer')
                    })
                else:
                    app.logger.warning(f"ML service returned error: {ml_response.get('message', 'Unknown error')}")
                    # In case of ML error, create a record to show the attempt
                    doc_id = AudioTranscription.create(
                        chatid=chatid,
                        user_question="[Audio processing failed]",
                        answer=f"Error: {ml_response.get('message', 'Unknown error')}"
                    )
                    app.logger.info(f"Created error record, ID: {doc_id}")
            else:
                app.logger.error(f"ML service returned error status code: {response.status_code}, response: {response.text}")
                # In case of HTTP error, create a record to show the attempt
                doc_id = AudioTranscription.create(
                    chatid=chatid,
                    user_question="[Error processing audio]",
                    answer=f"HTTP Error: {response.status_code}"
                )
                app.logger.info(f"Created error record, ID: {doc_id}")
            
            return jsonify({
                'success': False,
                'message': 'Error processing audio with ML service',
                'chatid': chatid,
            })
                
        except requests.RequestException as e:
            app.logger.error(f"Error communicating with ML service: {str(e)}", exc_info=True)
            
            # Clean up temporary file if request failed
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                app.logger.info(f"Removed temporary file after error: {temp_file_path}")
            
            # Create error record in database
            doc_id = AudioTranscription.create(
                chatid=chatid,
                user_question="[Connection error]",
                answer=f"Error connecting to ML service: {str(e)}"
            )
            app.logger.info(f"Created error record, ID: {doc_id}")
                
            return jsonify({
                'success': False,
                'message': f'Error communicating with ML service: {str(e)}',
                'chatid': chatid
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error processing audio: {str(e)}", exc_info=True)
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
                'question': item.get('user_question', ''),
                'answer': item.get('answer', ''),
                'translated_content': item.get('translated_content', ''),  # For backward compatibility
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in data
        ]
        return jsonify(results_list)
    except Exception as e:
        app.logger.error(f"Error fetching results: {e}", exc_info=True)
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
                'question': item.get('user_question', ''),
                'answer': item.get('answer', ''),
                'translated_content': item.get('translated_content', ''),  # For backward compatibility
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in data
        ]
        return jsonify(results_list)
    except Exception as e:
        app.logger.error(f"Error fetching chat results: {e}", exc_info=True)
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
