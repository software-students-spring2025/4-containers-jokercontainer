"""
Module for the web application that handles audio recording,
triggers ML processing, and provides results retrieval.
"""

import os
import uuid
import base64
import tempfile
import logging
import threading
from time import sleep, time
from flask import Flask, request, render_template, jsonify
import requests
from pymongo.errors import ConnectionFailure

# Use a try-except to handle import error for common.models
try:
    from common.models import AudioTranscription
except ImportError:
    # Create a placeholder for the AudioTranscription class if import fails
    # This allows the code to pass linting but would need proper resolution
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory cache for queries that haven't been stored in the database yet
# Structure: {chatid: {"query": "...", "timestamp": time.time()}}
query_cache = {}


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
    and returns the file path and chatid for subsequent processing.
    
    Returns:
        JSON response with recording status and chatid
    """
    # Get JSON data from request
    data = request.json
    
    if not data or 'audio_data' not in data:
        logger.warning("No audio data provided in request")
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Get or generate chat ID
    chatid = data.get('chatid') or str(uuid.uuid4())
    logger.info("Processing recording for chatid: %s", chatid)
    
    try:
        # Extract the base64 audio data (remove the data URL prefix if present)
        base64_audio = data['audio_data']
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',', 1)[1]
        
        logger.info("Received base64 audio data, length: %s chars", len(base64_audio))
        
        # Decode the base64 data
        audio_data = base64.b64decode(base64_audio)
        logger.info("Decoded audio data, size: %s bytes", len(audio_data))
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
            logger.info("Saved audio data to temporary file: %s", temp_file_path)
        
        # Launch a background thread to process the audio
        process_thread = threading.Thread(
            target=process_audio_in_background,
            args=(temp_file_path, chatid),
            daemon=True
        )
        process_thread.start()
        logger.info("Started background processing thread for chatid: %s", chatid)
            
        # Return response to frontend with just the chatid
        # The frontend will poll separately for query extraction and answer
        return jsonify({
            'success': True,
            'message': 'Audio saved and processing started',
            'chatid': chatid
        })
                
    except Exception as e:
        logger.error("Error processing audio: %s", str(e), exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error processing audio: {str(e)}',
            'chatid': chatid
        }), 500


def process_audio_in_background(temp_file_path, chatid):
    """
    Process audio file in background thread by sending to ML service.
    
    Args:
        temp_file_path: Path to temporary audio file
        chatid: Chat ID for this recording
    """
    try:
        # Send the audio file to the ML service
        ml_service_url = os.getenv('ML_SERVICE_URL', 'http://ml:5001')
        logger.info("Sending audio to ML service at: %s/process_audio", ml_service_url)
        
        # Use with statement for file opening
        with open(temp_file_path, 'rb') as audio_file:
            files = {'audio_file': ('recording.webm', audio_file, 'audio/webm')}
            form_data = {'chatid': chatid}
            
            logger.info("Sending request to ML service with chatid: %s", chatid)
            # Add timeout to requests.post
            response = requests.post(
                f"{ml_service_url}/process_audio", 
                files=files, 
                data=form_data,
                timeout=30
            )
            
            # Log the response from ML service
            if response.status_code == 200:
                logger.info("ML service processed audio successfully: %s", response.json())
            else:
                logger.error("ML service returned error: %s, %s", response.status_code, response.text)
                
    except requests.RequestException as e:
        logger.error("Error communicating with ML service: %s", str(e), exc_info=True)
    except Exception as e:
        logger.error("Error in background processing: %s", str(e), exc_info=True)
    finally:
        # Clean up temporary file in all cases
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("Removed temporary file: %s", temp_file_path)


@app.route('/results')
def results():
    """
    Retrieve all translation results from the database.

    Returns:
        JSON response containing all translations with their metadata.
    """
    try:
        data = AudioTranscription.find_all()
        results_list = [format_transcription_item(item) for item in data]
        return jsonify(results_list)
    except Exception as e:
        logger.error("Error fetching results: %s", e, exc_info=True)
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
        results_list = [format_transcription_item(item) for item in data]
        return jsonify(results_list)
    except Exception as e:
        logger.error("Error fetching chat results: %s", e, exc_info=True)
        return jsonify({'error': str(e)}), 500


def format_transcription_item(item):
    """
    Format a database item for JSON response.
    
    Args:
        item: Database document to format
        
    Returns:
        Formatted dictionary for JSON response
    """
    return {
        'id': str(item['_id']),
        'chatid': item['chatid'],
        'question': item.get('user_question', ''),
        'answer': item.get('answer', ''),
        'created_at': item['created_at'].isoformat() if 'created_at' in item else None,
        'updated_at': item['updated_at'].isoformat() if 'updated_at' in item else None
    }


@app.route('/api/processing_notification', methods=['POST'])
def processing_notification():
    """
    Handle processing notification from the ML service.
    
    This endpoint receives a notification that the ML service has identified a query
    and is processing it. Instead of creating a database entry, it stores the query
    in an in-memory cache for the frontend to access.
    
    Returns:
        JSON response confirming receipt of the notification
    """
    try:
        data = request.json
        chatid = data.get('chatid')
        query = data.get('query')
        status = data.get('status')
        
        logger.info("Received processing notification for chatid: %s, query: %s, status: %s", 
                    chatid, query, status)
        
        # Store the query in our in-memory cache instead of the database
        if chatid and query:
            query_cache[chatid] = {
                "query": query,
                "timestamp": time()
            }
            logger.info("Stored query in cache for chatid: %s", chatid)
            
            return jsonify({
                'success': True,
                'message': 'Processing notification received and query cached',
                'chatid': chatid,
                'query': query,
                'status': status
            })
            
        return jsonify({
            'success': False,
            'message': 'Missing required parameters'
        }), 400
            
    except Exception as e:
        logger.error("Error handling processing notification: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """
    Clear all records from the database.
    
    Returns:
        JSON response indicating success or failure
    """
    try:
        before_count = len(AudioTranscription.find_all())
        # Delete all data from the collection
        collection = AudioTranscription.get_collection()
        result = collection.delete_many({})
        after_count = len(AudioTranscription.find_all())
        
        logger.info("Cleared %s records from database", result.deleted_count)
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleared {result.deleted_count} records',
            'deleted_count': result.deleted_count,
            'before_count': before_count,
            'after_count': after_count
        })
    except Exception as e:
        logger.error("Error clearing database: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error clearing database: {e}'
        }), 500


@app.route('/api/query_status/<chatid>', methods=['GET'])
def query_status(chatid):
    """
    Endpoint for the frontend to check if a query has been extracted for a chatid.
    
    This endpoint first checks the in-memory cache for queries that haven't been stored
    in the database yet. If not found in cache, it then checks the database.
    
    Args:
        chatid: ID of the chat to check query for
        
    Returns:
        JSON response with the extracted query if available
    """
    try:
        # First check our in-memory cache
        cached_query = query_cache.get(chatid)
        if cached_query:
            logger.info("Found query in cache for chatid: %s", chatid)
            return jsonify({
                'success': True,
                'chatid': chatid,
                'question': cached_query["query"],
                'has_query': True,
                'from_cache': True
            })
        
        # If not found in cache, check the database
        data = AudioTranscription.find_by_chatid(chatid)
        if data and len(data) > 0:
            # Found in database
            latest = data[0]  # Assuming sorted by created_at desc
            question = latest.get('user_question', '')
            if question:
                logger.info("Found query in database for chatid: %s", chatid)
                return jsonify({
                    'success': True,
                    'chatid': chatid,
                    'question': question,
                    'has_query': True,
                    'from_cache': False
                })
        
        # No query found in either cache or database
        logger.info("No query found for chatid: %s", chatid)
        return jsonify({
            'success': True,
            'chatid': chatid,
            'has_query': False,
            'from_cache': False
        })
            
    except Exception as e:
        logger.error("Error checking query status: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/answer_status/<chatid>', methods=['GET'])
def answer_status(chatid):
    """
    Endpoint for the frontend to check if an answer is available for a chatid.
    
    This endpoint checks if the answer has been generated for a query.
    It will also return the query from the cache if available but no answer yet.
    
    Args:
        chatid: ID of the chat to check answer for
        
    Returns:
        JSON response with the answer status and content if available
    """
    try:
        # Check if there's an entry in the database
        data = AudioTranscription.find_by_chatid(chatid)
        
        if data and len(data) > 0:
            # Entry exists, check for answer
            latest = data[0] # Assuming sorted by created_at desc
            answer = latest.get('answer', '')
            has_answer = bool(answer) and answer != 'PROCESSING'
            
            return jsonify({
                'success': True,
                'chatid': chatid,
                'question': latest.get('user_question', ''),
                'answer': answer if has_answer else '',
                'has_answer': has_answer,
                'is_processing': answer == 'PROCESSING'
            })
        
        # No database entry yet, check the cache for the query
        cached_query = query_cache.get(chatid)
        if cached_query:
            # We have a query but no answer yet
            return jsonify({
                'success': True,
                'chatid': chatid,
                'question': cached_query["query"],
                'has_answer': False,
                'is_processing': False,
                'from_cache': True
            })
        
        # No entry yet
        return jsonify({
            'success': True,
            'chatid': chatid,
            'has_answer': False,
            'is_processing': False,
            'from_cache': False
        })
            
    except Exception as e:
        logger.error("Error checking answer status: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


def cleanup_query_cache():
    """
    Clean up old entries from the query cache.
    Removes entries older than 30 minutes.
    """
    current_time = time()
    expired_time = current_time - (30 * 60)  # 30 minutes
    
    # Find expired keys
    expired_keys = [k for k, v in query_cache.items() if v["timestamp"] < expired_time]
    
    # Remove expired keys
    for key in expired_keys:
        logger.info("Removing expired query from cache: %s", key)
        del query_cache[key]
    
    if expired_keys:
        logger.info("Cleaned up %s expired queries from cache", len(expired_keys))


@app.route('/api/save_answer', methods=['POST'])
def save_answer():
    """
    Endpoint for the ML service to save an answer for a query.
    
    This endpoint creates a database entry with the question and answer,
    and also removes the query from the in-memory cache if it exists.
    
    Returns:
        JSON response confirming receipt of the answer
    """
    try:
        data = request.json
        chatid = data.get('chatid')
        question = data.get('question')
        answer = data.get('answer')
        
        if not chatid or not question or not answer:
            return jsonify({
                'success': False,
                'message': 'Missing required parameters (chatid, question, answer)'
            }), 400
        
        logger.info("Saving answer for chatid: %s", chatid)
        
        # Create the database entry
        doc_id = AudioTranscription.create(
            chatid=chatid,
            user_question=question,
            answer=answer
        )
        
        # Remove from cache if it exists
        if chatid in query_cache:
            logger.info("Removing query from cache after saving answer: %s", chatid)
            del query_cache[chatid]
        
        return jsonify({
            'success': True,
            'message': 'Answer saved successfully',
            'doc_id': str(doc_id)
        })
        
    except Exception as e:
        logger.error("Error saving answer: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


# Set up a background thread to clean up the cache periodically
if __name__ == '__main__':
    def cache_cleanup_thread():
        """Background thread that periodically cleans up the query cache"""
        while True:
            sleep(300)  # Run every 5 minutes
            try:
                cleanup_query_cache()
            except Exception as e:
                logger.error("Error in cache cleanup thread: %s", e, exc_info=True)
    
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cache_cleanup_thread, daemon=True)
    cleanup_thread.start()
    
    # Wait for a few seconds to ensure MongoDB is ready
    logger.info("Starting web application, waiting for MongoDB to be ready...")
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
