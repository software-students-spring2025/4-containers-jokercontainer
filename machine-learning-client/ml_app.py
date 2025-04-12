"""
Module for processing audio recordings, transcribing with OpenAI's Whisper, 
and generating responses with OpenAI's GPT models via a Flask web service.
"""

import os
import uuid
import tempfile
import base64
import asyncio
import logging
from time import sleep
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pymongo.errors import ConnectionFailure
from openai import OpenAI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
import requests

# Try-except for common.models import
try:
    from common.models import AudioTranscription
except ImportError:
    # Create a placeholder for the AudioTranscription class if import fails
    class AudioTranscription:
        """Placeholder for AudioTranscription class if import fails"""
        @staticmethod
        def find_all():
            """Placeholder method"""
            return []
        
        @staticmethod
        def find_by_chatid(chatid):
            """Placeholder method"""
            return []
        
        @staticmethod
        def create(**kwargs):
            """Placeholder method"""
            return None
        
        @staticmethod
        def get_collection():
            """Placeholder method"""
            return None
        
        @staticmethod
        def create_indexes():
            """Placeholder method"""
            return None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=openai_api_key)
logger.info("OpenAI client initialized successfully")

# Configure browser automation
config = BrowserConfig(
    headless=True,
)

# Initialize LLM models
plannerllm = ChatOpenAI(model="o3-mini", api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)


class UserQuery(BaseModel):
    """Model for extracting user queries from audio transcriptions."""
    is_query: bool
    user_query: str


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

    Returns:
        JSON response with the transcribed question and LLM response.
    """
    chatid = None
    audio_data = None
    temp_file_path = None
    
    logger.info("Received /process_audio request: %s", request.content_type)
    
    # Check if this is a JSON request with base64 audio
    if request.is_json:
        temp_file_path = _handle_json_request(request)
        if isinstance(temp_file_path, tuple):  # Error response
            return temp_file_path
        chatid = request.json.get('chatid')
    
    # Check if this is a multipart form with an audio file
    elif 'audio_file' in request.files:
        temp_file_path = _handle_multipart_request(request)
        if isinstance(temp_file_path, tuple):  # Error response
            return temp_file_path
        chatid = request.form.get('chatid')
    
    else:
        logger.warning("Request received with no audio data")
        return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
    
    # Generate chat ID if not provided
    if not chatid:
        chatid = str(uuid.uuid4())
        logger.info("Generated new chatid: %s", chatid)
    
    try:
        # Process the audio file with OpenAI
        logger.info("Starting audio transcription for file: %s", temp_file_path)
        transcribed_text = transcribe_audio(temp_file_path)
        logger.info("Audio successfully transcribed, text length: %s chars", len(transcribed_text))
        logger.info("Transcribed text: %s", transcribed_text)
        
        # Generate LLM response to the transcribed text
        logger.info("Processing transcribed text with LLM")
        userquery = process_text_with_llm(transcribed_text)
        logger.info("User query received, length: %s chars", len(userquery.user_query))
        logger.info("User query: %s", userquery.user_query)
        
        if userquery.is_query:
            # Notify web-app that we have a query and are processing it
            _notify_web_app(chatid, userquery.user_query)
            
            # Use browser to get the answer
            logger.info("Using browser to get the answer")
            answer = asyncio.run(browser_use(userquery.user_query))
            logger.info("Browser response received, length: %s chars", len(answer))
            logger.info("Browser response: %s", answer)
            
            # Save the answer to the database via the web app
            _save_answer_via_web_app(chatid, userquery.user_query, answer)
        else:
            logger.info("User did not ask a question")
            transcribed_text = 'Please ask a question'
            answer = 'I am sorry, it seems like you are not asking a question. Please try again.'
            
            # Notify web-app with the non-question and response
            _notify_web_app(chatid, 'Not a question')
            _save_answer_via_web_app(chatid, 'Not a question', answer)
        
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Removed temporary file: %s", temp_file_path)
        
        # Return success response
        return jsonify({
            'status': 'success',
            'chatid': chatid,
            'question': transcribed_text,
            'answer': answer,
            'model': 'gpt-4o'
        })
    
    except Exception as e:
        logger.error("Error processing audio: %s", str(e), exc_info=True)
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Removed temporary file after error: %s", temp_file_path)
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _handle_json_request(request):
    """
    Process a JSON request with base64 audio data.
    
    Args:
        request: The Flask request object
        
    Returns:
        str: Path to the temporary audio file or tuple(response, status_code) on error
    """
    data = request.get_json()
    if not data or 'audio' not in data:
        logger.warning("JSON request received without audio data")
        return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
    
    # Decode base64 audio
    base64_audio = data['audio']
    if ',' in base64_audio:
        base64_audio = base64_audio.split(',', 1)[1]
    
    # Save to temp file
    try:
        logger.info("Decoding base64 audio data")
        audio_data = base64.b64decode(base64_audio)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
        logger.info("Audio data saved to temporary file: %s", temp_file_path)
        return temp_file_path
    except Exception as e:
        logger.error("Error decoding base64 audio: %s", str(e))
        return jsonify({'status': 'error', 'message': f'Error decoding audio: {str(e)}'}), 400


def _handle_multipart_request(request):
    """
    Process a multipart form request with an audio file.
    
    Args:
        request: The Flask request object
        
    Returns:
        str: Path to the temporary audio file or tuple(response, status_code) on error
    """
    audio_file = request.files['audio_file']
    
    # If user submits an empty file
    if audio_file.filename == '':
        logger.warning("Empty audio file provided in multipart form")
        return jsonify({'status': 'error', 'message': 'Empty audio file provided'}), 400
    
    logger.info("Processing multipart form with audio file: %s", audio_file.filename)
    
    # Save the file temporarily
    try:
        filename = secure_filename(audio_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_file_path)
        logger.info("Audio file saved to: %s", temp_file_path)
        return temp_file_path
    except Exception as e:
        logger.error("Error saving audio file: %s", str(e))
        return jsonify({'status': 'error', 'message': f'Error saving audio file: {str(e)}'}), 500


def _notify_web_app(chatid, query):
    """
    Notify the web app that a query is being processed.
    
    Args:
        chatid: The ID of the chat
        query: The user's query text
    """
    try:
        logger.info("Notifying web-app that we have a query and are processing it")
        web_app_url = os.getenv('WEB_APP_URL', 'http://web:5001')
        notification_response = requests.post(
            f"{web_app_url}/api/processing_notification",
            json={
                'chatid': chatid,
                'query': query,
                'status': 'processing'
            },
            timeout=30
        )
        logger.info("Notification response: %s", notification_response.status_code)
    except Exception as e:
        logger.error("Error notifying web-app: %s", str(e))


def _save_answer_via_web_app(chatid, query, answer):
    """
    Save the answer to the web app database directly.
    
    Args:
        chatid: The ID of the chat
        query: The user's query
        answer: The answer to save
    """
    try:
        logger.info("Saving answer to web-app for chatid: %s", chatid)
        web_app_url = os.getenv('WEB_APP_URL', 'http://web:5001')
        response = requests.post(
            f"{web_app_url}/api/save_answer",
            json={
                'chatid': chatid,
                'question': query,
                'answer': answer
            },
            timeout=30
        )
        if response.status_code == 200:
            logger.info("Successfully saved answer via web-app API: %s", response.status_code)
        else:
            logger.error("Error saving answer via web-app: %s, %s", response.status_code, response.text)
    except Exception as e:
        logger.error("Error saving answer via web-app: %s", str(e))


def process_text_with_llm(text):
    """
    Process text using OpenAI's GPT models to extract the user query.

    Args:
        text (str): The text to process.

    Returns:
        UserQuery: Object containing the extracted query and whether it's a valid question.
    """
    try:
        logger.info("Calling OpenAI API with text: '%s...' (truncated)", text[:50])
        # Call OpenAI API to process the text
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a given some text converted from audio, you need to identify the query of user. "
                                "Note the text might be incomplete, so you should do your best to infer the query based on the existing information. "
                                "Make sure the query you put in response is a valid question."
                                "You should not answer the question, extract the query and use it as output"
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            response_format=UserQuery
        )
        
        # Extract and return the response text
        result = response.choices[0].message.parsed
        logger.info("OpenAI API response successful, model: %s, tokens used: %s", response.model, response.usage.total_tokens)
        return result
    
    except Exception as e:
        logger.error("Error from OpenAI API: %s", str(e), exc_info=True)
        raise Exception(f"Error processing text with LLM: {str(e)}")


async def browser_use(task):
    """
    Use browser automation to find answers to user queries.
    
    Args:
        task (str): The user query to answer
        
    Returns:
        str: The answer from the browser automation
    """
    browser = Browser(config=config)
    try:
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            planner_llm=plannerllm,
            use_vision_for_planner=False,
            planner_interval=4 
        )
        result = await agent.run()
        await browser.close()
        return result.final_result()
    except Exception as e:
        logger.error("Error from Browser Use: %s", str(e), exc_info=True)
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
        logger.info("Opening audio file for transcription: %s", file_path)
        with open(file_path, "rb") as audio_file:
            # Get the file size for logging
            audio_file.seek(0, os.SEEK_END)
            file_size = audio_file.tell()
            audio_file.seek(0)
            logger.info("Audio file size: %s bytes", file_size)
            
            # Using gpt-4o-transcribe per the project plan
            logger.info("Sending audio to OpenAI Whisper API for transcription")
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text"
            )
            
        # In newer versions, the response is an object with a text attribute
        if hasattr(transcription, 'text'):
            logger.info("Transcription successful, received response object")
            return transcription.text
        # In case it's just a string (for older versions)
        logger.info("Transcription successful, received text string")
        return transcription
    
    except Exception as e:
        logger.error("Error transcribing audio: %s", str(e), exc_info=True)
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
    logger.info("Received request for chat results, chatid: %s", chatid)
    try:
        items = AudioTranscription.find_by_chatid(chatid)
        logger.info("Found %s results for chatid: %s", len(items), chatid)
        result = [
            {
                'id': str(item['_id']),
                'chatid': item['chatid'],
                'question': item.get('user_question', ''),
                'answer': item.get('answer', ''),
                'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
                'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
            }
            for item in items
        ]
        return jsonify(result)
    except Exception as e:
        logger.error("Error retrieving chat results: %s", str(e), exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Wait for a few seconds to ensure MongoDB is ready
    logger.info("Starting ML application, waiting for MongoDB to be ready...")
    sleep(5)
    
    # Try to initialize database connection and indexes
    MAX_RETRIES = 5
    retries = MAX_RETRIES
    while retries > 0:
        try:
            AudioTranscription.create_indexes()
            logger.info("Successfully connected to MongoDB and created indexes")
            break
        except ConnectionFailure as e:
            logger.warning("Failed to connect to MongoDB, retrying... (%s attempts left)", retries)
            retries -= 1
            if retries == 0:
                logger.error("Could not connect to MongoDB: %s", e)
            sleep(5)
    
    # Convert the environment variable to the correct type
    port = int(os.getenv('PORT', '5001'))
    logger.info("Starting Flask server on port %s", port)
    app.run(host='0.0.0.0', port=port)
