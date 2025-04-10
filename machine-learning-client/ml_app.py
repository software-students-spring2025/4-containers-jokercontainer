"""
Module for processing audio recordings, transcribing with OpenAI's Whisper, 
and generating responses with OpenAI's GPT models via a Flask web service.
"""

import os
import uuid
import tempfile
import base64
from time import sleep
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo.errors import ConnectionFailure
from openai import OpenAI
from common.models import AudioTranscription
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=openai_api_key)


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
        Process audio file, transcribe with OpenAI, and generate a response with GPT.

        Expects either:
        1. A multipart form with:
           - 'audio_file': The audio file to process
           - 'chatid': (Optional) ID of the chat this recording belongs to
        2. Or a JSON payload with:
           - 'audio': Base64 encoded audio data
           - 'chatid': (Optional) ID of the chat
           - 'llm_settings': (Optional) Settings for the LLM

        Returns:
            JSON response with the transcribed question and LLM response.
    """
    chatid = None
    audio_data = None
    temp_file_path = None
    
    # Check if this is a JSON request with base64 audio
    if request.is_json:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
        
        # Get chat ID
        chatid = data.get('chatid')
        
        # Decode base64 audio
        base64_audio = data['audio']
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',', 1)[1]
        
        # Save to temp file
        try:
            audio_data = base64.b64decode(base64_audio)
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(audio_data)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error decoding audio: {str(e)}'}), 400
    
    # Check if this is a multipart form with an audio file
    elif 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        
        # If user submits an empty file
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty audio file provided'}), 400
        
        # Get chat ID
        chatid = request.form.get('chatid')
        
        # Save the file temporarily
        filename = secure_filename(audio_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_file_path)
    
    else:
        return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
    
    # Generate chat ID if not provided
    if not chatid:
        chatid = str(uuid.uuid4())
    
    try:
        # Process the audio file with OpenAI
        transcribed_text = transcribe_audio(temp_file_path)
        app.logger.info(f"Transcribed text: {transcribed_text}")
        
        # Generate LLM response to the transcribed text
        llm_response = process_text_with_llm(transcribed_text)
        app.logger.info(f"LLM response: {llm_response}")
        
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # Save result to database
        doc_id = AudioTranscription.create(
            chatid=chatid,
            user_question=transcribed_text,
            llm_response=llm_response
        )
        
        # Return success response
        return jsonify({
            'status': 'success',
            'id': doc_id,
            'chatid': chatid,
            'question': transcribed_text,
            'answer': llm_response,
            'model': 'gpt-4'
        })
    
    except Exception as e:
        app.logger.error(f"Error processing audio: {e}")
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({'status': 'error', 'message': str(e)}), 500


def process_text_with_llm(text):
    """
    Process text using OpenAI's GPT models.

    Args:
        text (str): The text to process.

    Returns:
        str: The response from the LLM.
    """
    try:
        # Call OpenAI API to process the text
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Using the model specified in project plan
            temperature=0.7,
            max_tokens=1000,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Answer the user's question concisely and accurately."
                },
                {
                    "role": "user", 
                    "content": text
                }
            ]
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    
    except Exception as e:
        app.logger.error(f"Error from OpenAI API: {str(e)}")
        return f"Sorry, I encountered an error processing your request: {str(e)}"


def transcribe_audio(file_path):
    """
    Transcribe audio file to text using OpenAI's Whisper API.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Transcribed text from the audio.
    """
    try:
        with open(file_path, "rb") as audio_file:
            # Using gpt-4o-transcribe per the project plan
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text"
            )
            
        # In newer versions, the response is an object with a text attribute
        if hasattr(transcription, 'text'):
            return transcription.text
        # In case it's just a string (for older versions)
        return transcription
    
    except Exception as e:
        app.logger.error(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Error transcribing audio: {str(e)}")


@app.route('/results/<chatid>', methods=['GET'])
def get_chat_results(chatid):
    """
    Get all Q&A pairs for a specific chat.
    
    Args:
        chatid: ID of the chat to retrieve results for
        
    Returns:
        JSON response with all Q&A pairs for the chat
    """
    try:
        items = AudioTranscription.find_by_chatid(chatid)
        result = [
            {
                'id': str(item['_id']),
                'chatid': item['chatid'],
                'question': item.get('user_question', ''),
                'answer': item.get('llm_response', ''),
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in items
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
