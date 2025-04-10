"""
Module for processing audio recordings and generating translations via a Flask web service.
"""

import os
import uuid
import tempfile
from time import sleep
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo.errors import ConnectionFailure
from common.models import AudioTranscription

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/process', methods=['POST'])
def process():
    """
        Legacy endpoint for text processing.
        Will be deprecated in favor of /process_audio.

        Returns:
            JSON response indicating success.
    """
    data = request.get_json()
    text = data.get('text', '')
    
    # Use provided chatid or generate a new one
    chatid = data.get('chatid')
    if not chatid:
        chatid = str(uuid.uuid4())

    # Process the text
    translated_content = translate_text(text)

    # Save to database
    try:
        doc_id = AudioTranscription.create(
            chatid=chatid,
            translated_content=translated_content
        )
        return jsonify({
            'status': 'success', 
            'id': doc_id,
            'chatid': chatid,
            'translated_content': translated_content
        })
    except Exception as e:
        app.logger.error(f"Error saving to database: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
        Process audio file and store its transcription in the database.

        Expects a multipart form with:
        - 'audio_file': The audio file to process
        - 'chatid': (Optional) ID of the chat this recording belongs to

        Returns:
            JSON response indicating success and containing the translated content.
    """
    # Check if audio file is in the request
    if 'audio_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
    audio_file = request.files['audio_file']
    
    # If user submits an empty file
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty audio file provided'}), 400
    
    # Use provided chatid or generate a new one
    chatid = request.form.get('chatid')
    if not chatid:
        chatid = str(uuid.uuid4())
    
    try:
        # Save the file temporarily
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Process the audio file - in a real app, this would use speech recognition
        translated_content = transcribe_audio(filepath)
        
        # Clean up the temporary file
        os.remove(filepath)
        
        # Save result to database
        doc_id = AudioTranscription.create(
            chatid=chatid,
            translated_content=translated_content
        )
        
        # Return success response
        return jsonify({
            'status': 'success',
            'id': doc_id,
            'chatid': chatid,
            'translated_content': translated_content
        })
    
    except Exception as e:
        app.logger.error(f"Error processing audio: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def translate_text(text):
    """
    Process text by converting it to uppercase.
    In a real application, this would be replaced with actual ML processing.

    Args:
        text (str): The text to process.

    Returns:
        str: The processed text in uppercase.
    """
    # Simple example - in a real app, this would use ML models
    return text.upper()


def transcribe_audio(file_path):
    """
    Transcribe audio file to text.
    In a real application, this would use speech recognition ML models.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Transcribed text from the audio.
    """
    # MOCK IMPLEMENTATION
    # In a real app, this would use speech-to-text ML models like:
    # - OpenAI Whisper
    # - Google Speech-to-Text
    # - Mozilla DeepSpeech
    # - Hugging Face Transformers (Wav2Vec2, etc.)
    
    # For demo purposes, we'll return a mock response
    return f"[AUDIO TRANSCRIPTION] This is a simulated transcription from audio file {os.path.basename(file_path)}. In a real implementation, we would use speech recognition here."


@app.route('/results/<chatid>', methods=['GET'])
def get_chat_results(chatid):
    """
    Get all translations for a specific chat.
    
    Args:
        chatid: ID of the chat to retrieve translations for
        
    Returns:
        JSON response with all translations for the chat
    """
    try:
        translations = AudioTranscription.find_by_chatid(chatid)
        result = [
            {
                'id': str(item['_id']),
                'chatid': item['chatid'],
                'translated_content': item['translated_content'],
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in translations
        ]
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error retrieving chat results: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
