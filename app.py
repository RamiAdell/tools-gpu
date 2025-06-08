import os
import uuid
import json
import logging
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS # type: ignore
import cv2
import tempfile
import time
from werkzeug.utils import secure_filename
import threading
import shutil
from rembg import remove

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
PROCESSED_FOLDER = '/tmp/processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm'}
API_KEY = "GPukTcc2FXcAo32U6j6y5rOK8LJW5QAf"

# Make sure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Progress tracking
progress_data = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_background_from_video(input_video_path, output_video_path, progress_callback):
    """Remove background from video using rembg"""
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    processed_frames = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to BGRA for rembg processing
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
            # Remove background using rembg
            output = remove(frame_rgba)
            
            # Convert back to BGR for video output
            output_bgr = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
            
            # Write frame to output video
            out.write(output_bgr)
            
            processed_frames += 1
            progress_percentage = (processed_frames / frame_count) * 100
            progress_callback(progress_percentage)
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise e
    finally:
        cap.release()
        out.release()

@app.before_request
def verify_api_key():
    # Skip API key check for progress endpoint which is used for SSE
    if request.path == '/progress':
        return
        
    # Check API key in headers
    api_key = request.headers.get('X-Api-Key')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
        
    file = request.files['file']
    user_id = request.headers.get('X-User-ID', 'unknown')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    try:
        # Create a unique filename
        unique_id = uuid.uuid4().hex
        filename = f"{user_id}_{unique_id}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(file_path)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            os.remove(file_path)
            return jsonify({'success': False, 'error': 'File too large'}), 400
            
        # Check duration
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os.remove(file_path)
            return jsonify({'success': False, 'error': 'Could not open video file'}), 400
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration > 300:  # 5 minutes max
            os.remove(file_path)
            return jsonify({'success': False, 'error': 'Video too long (max 5 minutes)'}), 400
        
        # Initialize progress tracking for this file
        progress_data[filename] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'Upload complete'
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'duration': int(duration),
            'original_name': file.filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

def process_video_background(filename, user_id):
    """Background removal processing function"""
    try:
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Update progress
        progress_data[filename] = {
            'status': 'processing',
            'progress': 5,
            'message': 'Starting background removal'
        }
        
        # Define progress callback function
        def update_progress(progress_percentage):
            progress_data[filename] = {
                'status': 'processing',
                'progress': min(99, int(progress_percentage)),
                'message': f'Processing: {int(progress_percentage):.1f}% complete'
            }
        
        # Process video using the remove_background_from_video function
        remove_background_from_video(input_path, output_path, update_progress)
        
        # Mark as complete
        progress_data[filename] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete',
            'output_path': output_path
        }
        
        # Clean up original upload
        try:
            os.remove(input_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        progress_data[filename] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/process', methods=['POST'])
def process_video():
    """Start video processing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        filename = data.get('filename')
        user_id = data.get('user_id', 'unknown')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
            
        # Check if file exists
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Start processing in background thread
        threading.Thread(
            target=process_video_background,
            args=(filename, user_id)
        ).start()
        
        # Poll for completion
        max_wait = 600  # 10 minutes max wait (increased for AI processing)
        wait_time = 0
        sleep_interval = 1
        
        while wait_time < max_wait:
            progress = progress_data.get(filename, {})
            if progress.get('status') == 'complete':
                # Return processed file
                output_path = progress.get('output_path')
                return send_file(output_path, as_attachment=True, 
                                download_name=f"processed_{filename}")
            elif progress.get('status') == 'error':
                return jsonify({'error': progress.get('message')}), 500
                
            time.sleep(sleep_interval)
            wait_time += sleep_interval
            
        return jsonify({'error': 'Processing timeout'}), 408
        
    except Exception as e:
        logger.error(f"Process request error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/progress', methods=['GET'])
def progress_stream():
    """Server-sent events endpoint for progress updates"""
    def generate():
        last_data = {}
        
        while True:
            # Copy to avoid modification during iteration
            current_data = progress_data.copy()
            
            # Only send if there's new data
            if current_data != last_data:
                data_str = json.dumps(current_data)
                yield f"data: {data_str}\n\n"
                last_data = current_data.copy()
                
            time.sleep(1)
    
    return Response(generate(), content_type='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = 5550
    
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)