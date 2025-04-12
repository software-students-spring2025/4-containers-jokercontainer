"""
Module for the web application that handles audio recording,
triggers ML processing, and provides results retrieval.
"""

import os
import uuid
import base64
import tempfile
import logging
from time import sleep
from flask import Flask, request, render_template, jsonify
import requests
from pymongo.errors import ConnectionFailure
from common.models import AudioTranscription
import threading

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
    logger.info(f"Processing recording for chatid: {chatid}")
    
    try:
        # Extract the base64 audio data (remove the data URL prefix if present)
        base64_audio = data['audio_data']
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',', 1)[1]
        
        logger.info(f"Received base64 audio data, length: {len(base64_audio)} chars")
        
        # Decode the base64 data
        audio_data = base64.b64decode(base64_audio)
        logger.info(f"Decoded audio data, size: {len(audio_data)} bytes")
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
            logger.info(f"Saved audio data to temporary file: {temp_file_path}")
        
        # Launch a background thread to process the audio
        process_thread = threading.Thread(
            target=process_audio_in_background,
            args=(temp_file_path, chatid),
            daemon=True
        )
        process_thread.start()
        logger.info(f"Started background processing thread for chatid: {chatid}")
            
        # Return response to frontend with just the chatid
        # The frontend will poll separately for query extraction and answer
        return jsonify({
            'success': True,
            'message': 'Audio saved and processing started',
            'chatid': chatid
        })
                
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
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
        logger.info(f"Sending audio to ML service at: {ml_service_url}/process_audio")
        
        files = {'audio_file': ('recording.webm', open(temp_file_path, 'rb'), 'audio/webm')}
        form_data = {'chatid': chatid}
        
        logger.info(f"Sending request to ML service with chatid: {chatid}")
        response = requests.post(f"{ml_service_url}/process_audio", files=files, data=form_data)
        
        # Log the response from ML service
        if response.status_code == 200:
            logger.info(f"ML service processed audio successfully: {response.json()}")
        else:
            logger.error(f"ML service returned error: {response.status_code}, {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Error communicating with ML service: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}", exc_info=True)
    finally:
        # Clean up temporary file in all cases
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")


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
        logger.error(f"Error fetching results: {e}", exc_info=True)
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
        logger.error(f"Error fetching chat results: {e}", exc_info=True)
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
        
        logger.info(f"Received processing notification for chatid: {chatid}, query: {query}, status: {status}")
        
        # Store the query in our in-memory cache instead of the database
        if chatid and query:
            from time import time
            query_cache[chatid] = {
                "query": query,
                "timestamp": time()
            }
            logger.info(f"Stored query in cache for chatid: {chatid}")
            
            return jsonify({
                'success': True,
                'message': 'Processing notification received and query cached',
                'chatid': chatid,
                'query': query,
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Missing required parameters'
            }), 400
            
    except Exception as e:
        logger.error(f"Error handling processing notification: {e}", exc_info=True)
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
        
        logger.info(f"Cleared {result.deleted_count} records from database")
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleared {result.deleted_count} records',
            'deleted_count': result.deleted_count,
            'before_count': before_count,
            'after_count': after_count
        })
    except Exception as e:
        logger.error(f"Error clearing database: {e}", exc_info=True)
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
            logger.info(f"Found query in cache for chatid: {chatid}")
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
                logger.info(f"Found query in database for chatid: {chatid}")
                return jsonify({
                    'success': True,
                    'chatid': chatid,
                    'question': question,
                    'has_query': True,
                    'from_cache': False
                })
                
        # No query found in either cache or database
        logger.info(f"No query found for chatid: {chatid}")
        return jsonify({
            'success': True,
            'chatid': chatid,
            'has_query': False,
            'from_cache': False
        })
            
    except Exception as e:
        logger.error(f"Error checking query status: {e}", exc_info=True)
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
        else:
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
            else:
                # No entry yet
                return jsonify({
                    'success': True,
                    'chatid': chatid,
                    'has_answer': False,
                    'is_processing': False,
                    'from_cache': False
                })
            
    except Exception as e:
        logger.error(f"Error checking answer status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


def cleanup_query_cache():
    """
    Clean up old entries from the query cache.
    Removes entries older than 30 minutes.
    """
    from time import time
    current_time = time()
    expired_time = current_time - (30 * 60)  # 30 minutes
    
    # Find expired keys
    expired_keys = [k for k, v in query_cache.items() if v["timestamp"] < expired_time]
    
    # Remove expired keys
    for key in expired_keys:
        logger.info(f"Removing expired query from cache: {key}")
        del query_cache[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired queries from cache")


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
        
        logger.info(f"Saving answer for chatid: {chatid}")
        
        # Create the database entry
        doc_id = AudioTranscription.create(
            chatid=chatid,
            user_question=question,
            answer=answer
        )
        
        # Remove from cache if it exists
        if chatid in query_cache:
            logger.info(f"Removing query from cache after saving answer: {chatid}")
            del query_cache[chatid]
        
        return jsonify({
            'success': True,
            'message': 'Answer saved successfully',
            'doc_id': str(doc_id)
        })
        
    except Exception as e:
        logger.error(f"Error saving answer: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


# Set up a background thread to clean up the cache periodically
if __name__ == '__main__':
    import threading
    
    def cache_cleanup_thread():
        while True:
            sleep(300)  # Run every 5 minutes
            try:
                cleanup_query_cache()
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}", exc_info=True)
    
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cache_cleanup_thread, daemon=True)
    cleanup_thread.start()
    
    # Wait for a few seconds to ensure MongoDB is ready
    logger.info("Starting web application, waiting for MongoDB to be ready...")
    sleep(5)
    
    # Try to initialize database connection and indexes
    retries = 5
    while retries > 0:
        try:
            AudioTranscription.create_indexes()
            logger.info("Successfully connected to MongoDB and created indexes")
            break
        except ConnectionFailure as e:
            logger.warning(f"Failed to connect to MongoDB, retrying... ({retries} attempts left)")
            retries -= 1
            if retries == 0:
                logger.error(f"Could not connect to MongoDB: {e}")
            sleep(5)
    
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)
