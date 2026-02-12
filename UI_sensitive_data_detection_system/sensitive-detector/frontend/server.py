import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Backend API configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')


@app.route('/')
def index():
    """Serve the main SPA page."""
    return render_template('index.html')


@app.route('/api/proxy/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_to_backend(endpoint):
    """Proxy API calls to FastAPI backend."""
    try:
        backend_url = f"{BACKEND_URL}/api/v1/{endpoint}"
        
        # Forward the request to backend
        if request.method == 'GET':
            response = requests.get(backend_url, params=request.args)
        elif request.method == 'POST':
            if request.is_json:
                response = requests.post(backend_url, json=request.json)
            else:
                # Handle file uploads and form data
                files = {}
                data = {}
                
                for key, value in request.form.items():
                    data[key] = value
                
                for key, file in request.files.items():
                    files[key] = (file.filename, file.stream, file.content_type)
                
                response = requests.post(backend_url, files=files, data=data)
        else:
            return jsonify({'error': 'Method not supported'}), 405
        
        # Return response
        if response.headers.get('content-type', '').startswith('application/json'):
            return jsonify(response.json()), response.status_code
        else:
            # Handle binary responses (images, PDFs)
            return response.content, response.status_code, {
                'Content-Type': response.headers.get('content-type', 'application/octet-stream')
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Backend request failed: {e}")
        return jsonify({'error': 'Backend service unavailable'}), 503
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')


@socketio.on('process_document')
def handle_process_document(data):
    """Handle document processing requests with real-time updates."""
    try:
        emit('processing_status', {'status': 'started', 'message': 'Processing document...'})
        
        # This would be expanded to handle actual processing
        # For now, just emit status updates
        emit('processing_status', {'status': 'ocr', 'message': 'Running OCR...'})
        
        # Simulate processing steps
        import time
        time.sleep(1)
        
        emit('processing_status', {'status': 'detection', 'message': 'Detecting sensitive data...'})
        time.sleep(1)
        
        emit('processing_status', {'status': 'completed', 'message': 'Processing completed'})
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        emit('processing_status', {'status': 'error', 'message': str(e)})


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html')  # SPA handles routing


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Backend API URL: {BACKEND_URL}")
    
    # In containerized/prod-like environments, Werkzeug is blocked by default.
    # Allow it explicitly since this UI is low-load.
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)

