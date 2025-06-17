import os
import uuid
import json
import logging
import tempfile 
import time
import threading # Not used for main path in single endpoint, but run_captioning_pipeline could still be structured for clarity
import shutil

from flask import Flask, request, jsonify, Response, send_file, make_response
from flask_cors import CORS # type: ignore
import numpy as np
from werkzeug.utils import secure_filename

# Video/Audio Processing (ensure all imports from previous version are here)
import cv2 
from moviepy.editor import VideoFileClip # type: ignore
from PIL import Image, ImageDraw, ImageFont # type: ignore
from pysrt import SubRipFile # type: ignore
import arabic_reshaper # type: ignore
from bidi.algorithm import get_display # type: ignore
import whisper # type: ignore
from deep_translator import GoogleTranslator # type: ignore
from pydub.utils import mediainfo # type: ignore
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
# GPU Support
import torch

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GPU Configuration ---
# Check for GPU availability and set device
if torch.cuda.is_available():
    device = "cuda"
    gpu_info = torch.cuda.get_device_properties(0)
    logger.info(f"GPU detected: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.1f} GB memory")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    device = "cpu"
    logger.warning("No GPU detected, using CPU for processing")

# Set memory allocation strategy for GPU
if device == "cuda":
    torch.cuda.empty_cache()
    # Set memory fraction to avoid OOM errors
    torch.cuda.set_per_process_memory_fraction(0.8)

# --- Configuration (same as before) ---
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/caption_uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', '/tmp/caption_processed')
FONT_FOLDER = os.getenv('FONT_FOLDER', '/app/fonts')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'mkv', 'avi'}
SERVICE_API_KEY = os.getenv('CAPTION_SERVICE_API_KEY', "YourSecretApiKeyForCaptionService123")
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100')) * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = int(os.getenv('MAX_VIDEO_DURATION_SECONDS', '300'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FONT_FOLDER, exist_ok=True)

# Update your font paths to include extensions
font_paths_to_try = [
    os.path.join(FONT_FOLDER, "Poppins-Bold.ttf"),  # Added .ttf
    "Poppins-Bold.ttf",  # System font with extension
    os.path.join(FONT_FOLDER, "Arial.ttf"),
    "arial.ttf",
]

# Add debug logging
logger.debug(f"Looking for fonts in: {FONT_FOLDER}")
logger.debug(f"Font files present: {os.listdir(FONT_FOLDER)}")

# Global Whisper model - load once and reuse
whisper_model = None
whisper_model_size = os.getenv('WHISPER_MODEL_SIZE', 'small')  # Can be: tiny, base, small, medium, large

def load_whisper_model():
    """Load Whisper model with GPU support if available"""
    global whisper_model
    if whisper_model is None:
        try:
            logger.info(f"Loading Whisper model '{whisper_model_size}' on device: {device}")
            whisper_model = whisper.load_model(whisper_model_size, device=device)
            logger.info(f"Whisper model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model on {device}: {e}")
            # Fallback to CPU if GPU loading fails
            if device == "cuda":
                logger.info("Attempting to load Whisper model on CPU as fallback")
                whisper_model = whisper.load_model(whisper_model_size, device="cpu")
            else:
                raise e
    return whisper_model

# --- Helper Functions (same as before) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_duration(file_path):
    try:
        info = mediainfo(file_path)
        return float(info['duration'])
    except Exception as e:
        logger.error(f"Pydub mediainfo error for {file_path}: {e}")
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): return 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e_cv2:
            logger.error(f"OpenCV duration error for {file_path}: {e_cv2}")
            return 0

def format_whisper_timestamp(seconds): # Copied for completeness
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000; milliseconds %= 3_600_000
    minutes = milliseconds // 60_000; milliseconds %= 60_000
    seconds = milliseconds // 1_000; milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Core processing functions with GPU optimization
def _extract_audio(video_path, audio_path, job_id_log_prefix=""):
    logger.info(f"{job_id_log_prefix} Extracting audio from {video_path} to {audio_path}")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    video.close()
    logger.info(f"{job_id_log_prefix} Audio extracted.")

def _audio_to_text(wav_path, srt_path, job_id_log_prefix=""):
    logger.info(f"{job_id_log_prefix} Transcribing {wav_path} to {srt_path} using {device}")
    
    try:
        # Load model if not already loaded
        model = load_whisper_model()
        
        # Transcribe with GPU acceleration
        start_time = time.time()
        
        # Set FP16 based on device - GPU can use FP16 for faster processing
        use_fp16 = device == "cuda"
        
        result = model.transcribe(
            wav_path, 
            fp16=use_fp16,
            verbose=False  # Reduce logging noise
        )
        
        processing_time = time.time() - start_time
        logger.info(f"{job_id_log_prefix} Transcription completed in {processing_time:.2f}s using {device}")
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start, end, text = segment['start'], segment['end'], segment['text'].strip()
                f.write(f"{i+1}\n{format_whisper_timestamp(start)} --> {format_whisper_timestamp(end)}\n{text}\n\n")
        
        logger.info(f"{job_id_log_prefix} Transcription complete. SRT saved to {srt_path}")
        
        # Clear GPU cache after transcription
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error during transcription: {str(e)}", exc_info=True)
        raise

def _translate_srt(original_srt_path, target_lang, translated_srt_path, job_id_log_prefix=""):
    if target_lang.lower() in ['en', 'english']:
        logger.info(f"{job_id_log_prefix} Target is English, skipping translation.")
        shutil.copyfile(original_srt_path, translated_srt_path)
        return
    logger.info(f"{job_id_log_prefix} Translating {original_srt_path} to {target_lang}")
    subs = SubRipFile.open(original_srt_path, encoding='utf-8')
    translator = GoogleTranslator(source='auto', target=target_lang)
    for sub in subs:
        try:
            translated = translator.translate(sub.text)
            sub.text = translated if translated else sub.text
        except Exception as e_trans:
            logger.warning(f"{job_id_log_prefix} Translation failed for '{sub.text[:30]}...': {e_trans}")
    subs.save(translated_srt_path, encoding='utf-8')
    logger.info(f"{job_id_log_prefix} Translation complete.")
    
def _add_captions_to_video(video_path, srt_path, output_path, font_opts, job_id_log_prefix=""):
    """Add captions to video with enhanced error handling and text rendering."""
    try:
        logger.info(f"{job_id_log_prefix} Adding captions from {srt_path} to {video_path}")
        
        # Validate input files
        if not all(os.path.exists(f) for f in [video_path, srt_path]):
            raise FileNotFoundError("Input video or SRT file not found")
        
        # Load video and subtitles
        video = VideoFileClip(video_path)
        subs = SubRipFile.open(srt_path, encoding='utf-8')
        
        # Process font options with robust fallback
        font_name = font_opts.get('family', 'Arial.ttf')
        font_size = font_opts.get('size', 24)
        font_color = font_opts.get('color', '#FFFFFF')
        
        # Font loading with multiple fallbacks
        font = None
        font_paths_to_try = [
            os.path.join(FONT_FOLDER, font_name),
            font_name,  # Try system font
            os.path.join(FONT_FOLDER, "Arial.ttf"),
            "arial.ttf",  # Common system fallback
            os.path.join(FONT_FOLDER, "LiberationSans-Regular.ttf")
        ]
        
        for fp in font_paths_to_try:
            try:
                font = ImageFont.truetype(fp, font_size)
                logger.info(f"{job_id_log_prefix} Using font: {fp}")
                break
            except (IOError, OSError):
                continue
                
        if font is None:
            raise RuntimeError(f"Could not load any fallback font from: {font_paths_to_try}")

        def text_overlay_func(get_frame, t):
            """Process each frame to add captions."""
            try:
                frame_array = get_frame(t)
                img = Image.fromarray(frame_array)
                draw = ImageDraw.Draw(img)
                
                # Get active subtitles for current time
                active_texts = []
                for sub in subs:
                    start = sub.start.ordinal / 1000.0
                    end = sub.end.ordinal / 1000.0
                    if start <= t <= end:
                        text = sub.text
                        # Handle RTL languages like Arabic
                        if any('\u0600' <= char <= '\u06FF' for char in text):
                            text = get_display(arabic_reshaper.reshape(text))
                        active_texts.append(text)
                
                if not active_texts:
                    return np.array(img)
                
                full_caption = " ".join(active_texts)
                
                # Improved text wrapping
                max_width = img.width * 0.9
                lines = []
                words = full_caption.split()
                current_line = ""
                
                for word in words:
                    test_line = f"{current_line} {word}" if current_line else word
                    text_width = draw.textlength(test_line, font=font)
                    
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Calculate text position
                line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 5
                total_text_height = len(lines) * line_height
                margin = img.height * 0.05
                base_y = img.height - total_text_height - margin
                
                # Draw each line with stroke/border
                stroke_width = 2
                stroke_color = (0, 0, 0)  # Black stroke
                text_color = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Hex to RGB
                
                for i, line in enumerate(lines):
                    if not line:
                        continue
                        
                    text_width = draw.textlength(line, font=font)
                    x = (img.width - text_width) / 2
                    y = base_y + (i * line_height)
                    
                    # Draw stroke (outline)
                    for dx in [-stroke_width, stroke_width]:
                        for dy in [-stroke_width, stroke_width]:
                            draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
                    
                    # Draw main text
                    draw.text((x, y), line, font=font, fill=text_color)
                
                return np.array(img)
            
            except Exception as e:
                logger.error(f"{job_id_log_prefix} Error processing frame at {t}s: {str(e)}")
                return get_frame(t)  # Return original frame on error
        
        # Process video with progress tracking
        logger.info(f"{job_id_log_prefix} Starting video processing...")
        captioned_clip = video.fl(text_overlay_func, apply_to=['video'])
        
        # Write output with optimized settings
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        captioned_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset='medium',
            ffmpeg_params=["-crf", "23"],
            logger='bar' if logger.level <= logging.INFO else None
        )
        
        logger.info(f"{job_id_log_prefix} Successfully created captioned video: {output_path}")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error in _add_captions_to_video: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure resources are cleaned up
        if 'video' in locals():
            video.close()
        if 'captioned_clip' in locals():
            captioned_clip.close()

# --- Flask Middleware (API Key Check) ---
@app.before_request
def verify_api_key_middleware():
    if request.path in ['/health', '/gpu-status']: # Allow health check and GPU status without key
        return
    api_key = request.headers.get('X-Api-Key')
    if api_key != SERVICE_API_KEY:
        logger.warning(f"Unauthorized API key from {request.remote_addr} to {request.path}")
        return jsonify({'error': 'Unauthorized: Invalid API Key'}), 401

# --- Flask Routes ---
@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """Get GPU status and information"""
    status = {
        'gpu_available': torch.cuda.is_available(),
        'device': device,
        'whisper_model_loaded': whisper_model is not None,
        'whisper_model_size': whisper_model_size
    }

    if device == "cuda":
        status.update({
            'gpu_name': gpu_info.name,
            'gpu_memory_total': gpu_info.total_memory,
            'gpu_memory_allocated': torch.cuda.memory_allocated(0),
            'gpu_memory_cached': torch.cuda.memory_reserved(0),
            'cuda_version': torch.version.cuda
        })

    return jsonify(status), 200

@app.route('/process_direct', methods=['POST'])
def process_video_directly():
    """
    Single endpoint to receive video and parameters, process it, and stream back the result.
    """
    user_id = request.headers.get('X-User-ID', 'unknown_user')
    job_id_log_prefix = f"[DirectProcess-{user_id}-{uuid.uuid4().hex[:8]}]" # Unique ID for logging this specific job
    logger.info(f"{job_id_log_prefix} Direct processing request received for user: {user_id}")

    if 'video_file' not in request.files:
        return jsonify({'success': False, 'error': 'No video_file part in request.'}), 400
    
    video_file_storage = request.files['video_file']

    if video_file_storage.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(video_file_storage.filename):
        return jsonify({'success': False, 'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400

    # Get caption parameters from form data
    try:
        language = request.form.get('language', 'en')
        font_family = request.form.get('font_family', 'Arial.ttf')
        font_size = int(request.form.get('font_size', 24))
        font_color = request.form.get('font_color', 'white')
        font_options = {'family': font_family, 'size': font_size, 'color': font_color}
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid font size parameter.'}), 400
    except Exception as e_form:
        logger.error(f"{job_id_log_prefix} Error parsing form data: {e_form}", exc_info=True)
        return jsonify({'success': False, 'error': 'Error parsing caption parameters.'}), 400

    original_secure_filename = secure_filename(video_file_storage.filename)
    # Use a unique name for the temporary uploaded file
    temp_input_filename = f"{user_id}_{uuid.uuid4().hex}_temp_{original_secure_filename}"
    temp_input_filepath = os.path.join(UPLOAD_FOLDER, temp_input_filename)

    # Define paths for intermediate and final files for this job
    base_name_for_job = temp_input_filename.rsplit('.',1)[0]
    audio_file_path = os.path.join(PROCESSED_FOLDER, f"{base_name_for_job}_audio.wav")
    srt_file_path = os.path.join(PROCESSED_FOLDER, f"{base_name_for_job}_original.srt")
    translated_srt_file_path = os.path.join(PROCESSED_FOLDER, f"{base_name_for_job}_translated.srt")
    final_output_video_path = os.path.join(PROCESSED_FOLDER, f"captioned_{temp_input_filename}")

    files_to_cleanup = [temp_input_filepath, audio_file_path, srt_file_path, translated_srt_file_path, final_output_video_path]

    try:
        video_file_storage.save(temp_input_filepath)
        logger.info(f"{job_id_log_prefix} File saved temporarily to {temp_input_filepath}")

        file_size = os.path.getsize(temp_input_filepath)
        if file_size > MAX_FILE_SIZE_MB:
            raise ValueError(f'File too large (max {MAX_FILE_SIZE_MB // (1024*1024)}MB).')
        
        duration_seconds = get_video_duration(temp_input_filepath)
        if duration_seconds == 0 and file_size > 1000: # Check for valid video
             raise ValueError('Invalid video file or could not determine duration.')
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f'Video too long (max {MAX_VIDEO_DURATION_SECONDS // 60} minutes).')

        logger.info(f"{job_id_log_prefix} Video validated. Duration: {duration_seconds:.2f}s. Starting pipeline on {device}.")

        # --- Execute processing pipeline synchronously ---
        _extract_audio(temp_input_filepath, audio_file_path, job_id_log_prefix)
        _audio_to_text(audio_file_path, srt_file_path, job_id_log_prefix)
        _translate_srt(srt_file_path, language, translated_srt_file_path, job_id_log_prefix)
        
        srt_to_use = translated_srt_file_path if os.path.exists(translated_srt_file_path) and os.path.getsize(translated_srt_file_path) > 0 else srt_file_path
        _add_captions_to_video(temp_input_filepath, srt_to_use, final_output_video_path, font_options, job_id_log_prefix)
        # --- End of pipeline ---

        logger.info(f"{job_id_log_prefix} Processing complete. Output: {final_output_video_path}")
        
        response = make_response(send_file(
            final_output_video_path,
            as_attachment=True,
            download_name=f"captioned_{original_secure_filename}" # Suggest a nice name for client
        ))
        # Send video duration back to Django for usage tracking
        response.headers['X-Video-Duration-Seconds'] = str(int(duration_seconds))
        response.headers['X-GPU-Used'] = str(device)  # Indicate which device was used
        response.headers['Content-Type'] = 'video/mp4' # Explicitly set content type

        return response

    except ValueError as ve: # Catch specific validation errors
        logger.error(f"{job_id_log_prefix} Validation error: {str(ve)}", exc_info=True)
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error during direct processing: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'An internal error occurred: {str(e)}'}), 500
    finally:
        # Cleanup all temporary files for this job
        for f_path in files_to_cleanup:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logger.info(f"{job_id_log_prefix} Cleaned up: {f_path}")
                except OSError as e_clean:
                    logger.warning(f"{job_id_log_prefix} Could not clean up file {f_path}: {e_clean}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'message': 'Video Captioning Service is running.',
        'gpu_available': torch.cuda.is_available(),
        'device': device
    }), 200

@app.route('/progress', methods=['GET'])
def stream_progress_sse():
    """Server-Sent Events endpoint for real-time progress updates (if used)."""
    target_filename = request.args.get('filename') # This would need to be a job_id now

    def generate_progress_events():
        yield f"event: connection\ndata: {json.dumps({'message': 'Connected to progress stream.'})}\n\n"
        while True:
            yield f"data: {json.dumps({'time': time.time(), 'status': 'polling for general progress...', 'device': device})}\n\n"
            time.sleep(5)
    return Response(generate_progress_events(), content_type='text/event-stream')


if __name__ == '__main__':
    # Preload Whisper model on startup
    try:
        load_whisper_model()
        logger.info("Whisper model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}")
    
    port = 5003
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true', threaded=True)