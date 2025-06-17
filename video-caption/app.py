import os
import uuid
import json
import logging
import tempfile 
import time
import threading
import shutil

from flask import Flask, request, jsonify, Response, send_file, make_response
from flask_cors import CORS
import numpy as np
from werkzeug.utils import secure_filename

# Video/Audio Processing
import cv2 
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from pysrt import SubRipFile
import arabic_reshaper
from bidi.algorithm import get_display
import whisper
from deep_translator import GoogleTranslator
from pydub.utils import mediainfo
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

# --- Enhanced GPU Configuration ---
def setup_gpu():
    """Enhanced GPU setup with better error handling"""
    global device
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_props = torch.cuda.get_device_properties(current_device)
            
            logger.info(f"GPU Count: {gpu_count}")
            logger.info(f"Current GPU: {current_device}")
            logger.info(f"GPU Name: {gpu_props.name}")
            logger.info(f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            
            # Test GPU functionality
            test_tensor = torch.randn(10, 10).cuda()
            test_result = torch.matmul(test_tensor, test_tensor)
            logger.info(f"GPU Test: {test_result.shape} tensor created successfully")
            
            # Configure memory management
            torch.cuda.empty_cache()
            
            # Set memory fraction to prevent OOM
            if gpu_props.total_memory > 8 * 1024**3:  # > 8GB
                memory_fraction = 0.8
            else:
                memory_fraction = 0.7
            
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"Set GPU memory fraction to {memory_fraction}")
            
            device = "cuda"
            return True
            
        except Exception as e:
            logger.error(f"GPU setup failed: {e}")
            logger.warning("Falling back to CPU")
            device = "cpu"
            return False
    else:
        logger.warning("No GPU detected, using CPU")
        device = "cpu"
        return False

# Initialize GPU
gpu_available = setup_gpu()

# --- Configuration ---
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

# Font configuration
font_paths_to_try = [
    os.path.join(FONT_FOLDER, "Poppins-Bold.ttf"),
    "Poppins-Bold.ttf",
    os.path.join(FONT_FOLDER, "Arial.ttf"),
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Common Linux font
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "C:\\Windows\\Fonts\\arial.ttf",  # Windows
]

logger.debug(f"Looking for fonts in: {FONT_FOLDER}")
if os.path.exists(FONT_FOLDER):
    logger.debug(f"Font files present: {os.listdir(FONT_FOLDER)}")

# Global Whisper model
whisper_model = None
whisper_model_size = os.getenv('WHISPER_MODEL_SIZE', 'small')

def load_whisper_model():
    """Load Whisper model with enhanced GPU support"""
    global whisper_model
    
    if whisper_model is None:
        try:
            logger.info(f"Loading Whisper model '{whisper_model_size}' on device: {device}")
            
            if device == "cuda":
                # Ensure GPU is available
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                
                # Clear cache before loading
                torch.cuda.empty_cache()
                
                # Load model on GPU
                whisper_model = whisper.load_model(whisper_model_size, device="cuda")
                
                # Verify model is on GPU
                model_device = next(whisper_model.parameters()).device
                logger.info(f"Whisper model loaded on device: {model_device}")
                
                if model_device.type != 'cuda':
                    logger.warning(f"Model unexpectedly on {model_device}, moving to GPU")
                    whisper_model = whisper_model.cuda()
                    model_device = next(whisper_model.parameters()).device
                    logger.info(f"Model moved to: {model_device}")
                
                # Log memory usage
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
            else:
                whisper_model = whisper.load_model(whisper_model_size, device="cpu")
                logger.info("Whisper model loaded on CPU")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model on {device}: {e}")
            
            # Fallback to CPU if GPU fails
            if device == "cuda":
                logger.info("Attempting CPU fallback for Whisper model")
                try:
                    whisper_model = whisper.load_model(whisper_model_size, device="cpu")
                    logger.info("Whisper model loaded on CPU as fallback")
                except Exception as e_cpu:
                    logger.error(f"CPU fallback also failed: {e_cpu}")
                    raise e_cpu
            else:
                raise e
    
    return whisper_model

def ensure_model_on_gpu():
    """Ensure model is on GPU and log status"""
    global whisper_model
    
    if whisper_model is not None and device == "cuda":
        try:
            model_device = next(whisper_model.parameters()).device
            
            if model_device.type != 'cuda':
                logger.info(f"Moving model from {model_device} to GPU")
                whisper_model = whisper_model.cuda()
                model_device = next(whisper_model.parameters()).device
                logger.info(f"Model now on: {model_device}")
            
            # Log current GPU status
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"Pre-inference GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error ensuring model on GPU: {e}")

# --- Helper Functions ---
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
            if not cap.isOpened(): 
                return 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e_cv2:
            logger.error(f"OpenCV duration error for {file_path}: {e_cv2}")
            return 0

def format_whisper_timestamp(seconds):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def _extract_audio(video_path, audio_path, job_id_log_prefix=""):
    """Extract audio from video"""
    logger.info(f"{job_id_log_prefix} Extracting audio from {video_path}")
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("No audio track found in video")
        
        video.audio.write_audiofile(
            audio_path, 
            codec='pcm_s16le',
            verbose=False,
            logger=None  # Suppress MoviePy logging
        )
        video.close()
        logger.info(f"{job_id_log_prefix} Audio extracted successfully")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Audio extraction failed: {e}")
        raise

def _audio_to_text(wav_path, srt_path, job_id_log_prefix=""):
    """Transcribe audio to text with GPU optimization"""
    logger.info(f"{job_id_log_prefix} Starting transcription using {device}")
    
    try:
        # Load and ensure model is ready
        model = load_whisper_model()
        ensure_model_on_gpu()
        
        # Pre-transcription GPU status
        if device == "cuda":
            pre_allocated = torch.cuda.memory_allocated(0) / 1024**3
            pre_cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"{job_id_log_prefix} Pre-transcription GPU Memory: {pre_allocated:.2f}GB allocated, {pre_cached:.2f}GB cached")
        
        # Transcription settings
        transcription_options = {
            'fp16': device == "cuda",  # Use FP16 only on GPU
            'verbose': False,  # Reduce logging
            'temperature': 0.0,  # Deterministic output
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
        }
        
        # Add GPU-specific optimizations
        if device == "cuda":
            transcription_options.update({
                'beam_size': 5,  # Smaller beam size for speed
                'best_of': 5,
            })
        
        # Perform transcription
        start_time = time.time()
        logger.info(f"{job_id_log_prefix} Transcribing audio file: {wav_path}")
        
        result = model.transcribe(wav_path, **transcription_options)
        
        processing_time = time.time() - start_time
        logger.info(f"{job_id_log_prefix} Transcription completed in {processing_time:.2f}s using {device}")
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                
                if text:  # Only write non-empty segments
                    f.write(f"{i+1}\n")
                    f.write(f"{format_whisper_timestamp(start)} --> {format_whisper_timestamp(end)}\n")
                    f.write(f"{text}\n\n")
        
        logger.info(f"{job_id_log_prefix} SRT file saved: {srt_path}")
        
        # Post-transcription cleanup
        if device == "cuda":
            post_allocated = torch.cuda.memory_allocated(0) / 1024**3
            post_cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"{job_id_log_prefix} Post-transcription GPU Memory: {post_allocated:.2f}GB allocated, {post_cached:.2f}GB cached")
            
            # Clear cache to free memory
            torch.cuda.empty_cache()
            
            final_allocated = torch.cuda.memory_allocated(0) / 1024**3
            final_cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"{job_id_log_prefix} After cleanup GPU Memory: {final_allocated:.2f}GB allocated, {final_cached:.2f}GB cached")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Transcription failed: {str(e)}", exc_info=True)
        raise

def _translate_srt(original_srt_path, target_lang, translated_srt_path, job_id_log_prefix=""):
    """Translate SRT file"""
    if target_lang.lower() in ['en', 'english']:
        logger.info(f"{job_id_log_prefix} Target is English, copying original SRT")
        shutil.copyfile(original_srt_path, translated_srt_path)
        return
    
    logger.info(f"{job_id_log_prefix} Translating to {target_lang}")
    
    try:
        subs = SubRipFile.open(original_srt_path, encoding='utf-8')
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        for sub in subs:
            try:
                if sub.text.strip():
                    translated = translator.translate(sub.text)
                    sub.text = translated if translated else sub.text
            except Exception as e_trans:
                logger.warning(f"{job_id_log_prefix} Translation failed for segment: {e_trans}")
        
        subs.save(translated_srt_path, encoding='utf-8')
        logger.info(f"{job_id_log_prefix} Translation completed")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Translation process failed: {e}")
        raise

def _add_captions_to_video(video_path, srt_path, output_path, font_opts, job_id_log_prefix=""):
    """Add captions to video with enhanced error handling"""
    try:
        logger.info(f"{job_id_log_prefix} Adding captions to video")
        
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        # Load video and subtitles
        video = VideoFileClip(video_path)
        subs = SubRipFile.open(srt_path, encoding='utf-8')
        
        # Font configuration
        font_name = font_opts.get('family', 'Arial.ttf')
        font_size = font_opts.get('size', 24)
        font_color = font_opts.get('color', '#FFFFFF')
        
        # Load font with fallbacks
        font = None
        for font_path in font_paths_to_try:
            try:
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"{job_id_log_prefix} Using font: {font_path}")
                break
            except (IOError, OSError):
                continue
        
        if font is None:
            logger.warning(f"{job_id_log_prefix} No TrueType font found, using default")
            font = ImageFont.load_default()

        def text_overlay_func(get_frame, t):
            """Add text overlay to frame"""
            try:
                frame_array = get_frame(t)
                img = Image.fromarray(frame_array)
                draw = ImageDraw.Draw(img)
                
                # Find active subtitles
                active_texts = []
                for sub in subs:
                    start = sub.start.ordinal / 1000.0
                    end = sub.end.ordinal / 1000.0
                    if start <= t <= end:
                        text = sub.text
                        # Handle RTL languages
                        if any('\u0600' <= char <= '\u06FF' for char in text):
                            text = get_display(arabic_reshaper.reshape(text))
                        active_texts.append(text)
                
                if not active_texts:
                    return np.array(img)
                
                full_caption = " ".join(active_texts)
                
                # Text wrapping
                max_width = img.width * 0.9
                lines = []
                words = full_caption.split()
                current_line = ""
                
                for word in words:
                    test_line = f"{current_line} {word}" if current_line else word
                    try:
                        text_width = draw.textlength(test_line, font=font)
                    except AttributeError:
                        # Fallback for older PIL versions
                        text_width = draw.textsize(test_line, font=font)[0]
                    
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Calculate position
                try:
                    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 5
                except AttributeError:
                    line_height = font.getsize("A")[1] + 5
                
                total_height = len(lines) * line_height
                margin = img.height * 0.05
                base_y = img.height - total_height - margin
                
                # Draw text with outline
                stroke_width = 2
                stroke_color = (0, 0, 0)
                
                # Parse color
                try:
                    if font_color.startswith('#'):
                        text_color = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    else:
                        text_color = (255, 255, 255)  # Default white
                except:
                    text_color = (255, 255, 255)
                
                for i, line in enumerate(lines):
                    if not line:
                        continue
                    
                    try:
                        text_width = draw.textlength(line, font=font)
                    except AttributeError:
                        text_width = draw.textsize(line, font=font)[0]
                    
                    x = (img.width - text_width) / 2
                    y = base_y + (i * line_height)
                    
                    # Draw stroke
                    for dx in [-stroke_width, stroke_width]:
                        for dy in [-stroke_width, stroke_width]:
                            draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
                    
                    # Draw main text
                    draw.text((x, y), line, font=font, fill=text_color)
                
                return np.array(img)
            
            except Exception as e:
                logger.error(f"{job_id_log_prefix} Frame processing error at {t}s: {e}")
                return get_frame(t)
        
        # Create captioned video
        logger.info(f"{job_id_log_prefix} Processing video frames...")
        captioned_clip = video.fl(text_overlay_func, apply_to=['video'])
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write output
        captioned_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=min(4, mp.cpu_count()),
            preset='medium',
            ffmpeg_params=["-crf", "23"],
            verbose=False,
            logger=None
        )
        
        logger.info(f"{job_id_log_prefix} Video processing completed: {output_path}")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Video captioning failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'video' in locals():
            video.close()
        if 'captioned_clip' in locals():
            captioned_clip.close()

# --- Flask Middleware ---
@app.before_request
def verify_api_key_middleware():
    if request.path in ['/health', '/gpu-status']:
        return
    api_key = request.headers.get('X-Api-Key')
    if api_key != SERVICE_API_KEY:
        logger.warning(f"Unauthorized access from {request.remote_addr}")
        return jsonify({'error': 'Unauthorized: Invalid API Key'}), 401

# --- Flask Routes ---
@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """Enhanced GPU status endpoint"""
    status = {
        'gpu_available': torch.cuda.is_available(),
        'device': device,
        'whisper_model_loaded': whisper_model is not None,
        'whisper_model_size': whisper_model_size,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
    }

    if device == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            status.update({
                'gpu_name': props.name,
                'gpu_memory_total_gb': round(props.total_memory / 1024**3, 2),
                'gpu_memory_allocated_gb': round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                'gpu_memory_cached_gb': round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                'current_device': torch.cuda.current_device(),
                'device_count': torch.cuda.device_count(),
                'gpu_utilization_percent': None  # Would need nvidia-ml-py for this
            })
            
            # Check model device
            if whisper_model is not None:
                model_device = next(whisper_model.parameters()).device
                status['model_device'] = str(model_device)
                status['model_on_gpu'] = model_device.type == 'cuda'
            
        except Exception as e:
            status['gpu_error'] = str(e)
            logger.error(f"GPU status check failed: {e}")

    return jsonify(status), 200

@app.route('/process_direct', methods=['POST'])
def process_video_directly():
    """Enhanced direct video processing endpoint"""
    user_id = request.headers.get('X-User-ID', 'unknown_user')
    job_id = uuid.uuid4().hex[:8]
    job_id_log_prefix = f"[DirectProcess-{user_id}-{job_id}]"
    
    logger.info(f"{job_id_log_prefix} Processing request received")
    
    # Validate request
    if 'video_file' not in request.files:
        return jsonify({'success': False, 'error': 'No video_file in request'}), 400
    
    video_file = request.files['video_file']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'success': False, 'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    # Parse parameters
    try:
        language = request.form.get('language', 'en')
        font_family = request.form.get('font_family', 'Arial.ttf')
        font_size = int(request.form.get('font_size', 24))
        font_color = request.form.get('font_color', '#FFFFFF')
        font_options = {
            'family': font_family,
            'size': font_size,
            'color': font_color
        }
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid parameters: {e}'}), 400
    
    # Setup file paths
    original_filename = secure_filename(video_file.filename)
    temp_filename = f"{user_id}_{job_id}_{original_filename}"
    temp_input_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    
    base_name = temp_filename.rsplit('.', 1)[0]
    audio_path = os.path.join(PROCESSED_FOLDER, f"{base_name}_audio.wav")
    srt_path = os.path.join(PROCESSED_FOLDER, f"{base_name}_original.srt")
    translated_srt_path = os.path.join(PROCESSED_FOLDER, f"{base_name}_translated.srt")
    output_path = os.path.join(PROCESSED_FOLDER, f"captioned_{temp_filename}")
    
    cleanup_files = [temp_input_path, audio_path, srt_path, translated_srt_path, output_path]
    
    try:
        # Save uploaded file
        video_file.save(temp_input_path)
        logger.info(f"{job_id_log_prefix} File saved: {temp_input_path}")
        
        # Validate file
        file_size = os.path.getsize(temp_input_path)
        if file_size > MAX_FILE_SIZE_MB:
            raise ValueError(f'File too large (max {MAX_FILE_SIZE_MB // (1024*1024)}MB)')
        
        duration = get_video_duration(temp_input_path)
        if duration == 0:
            raise ValueError('Invalid video file')
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f'Video too long (max {MAX_VIDEO_DURATION_SECONDS // 60} minutes)')
        
        logger.info(f"{job_id_log_prefix} Video validated. Duration: {duration:.2f}s. Processing on {device}")
        
        # Prepare GPU for processing
        if device == "cuda":
            torch.cuda.empty_cache()
            ensure_model_on_gpu()
        
        # Process video
        _extract_audio(temp_input_path, audio_path, job_id_log_prefix)
        _audio_to_text(audio_path, srt_path, job_id_log_prefix)
        _translate_srt(srt_path, language, translated_srt_path, job_id_log_prefix)
        
        # Use translated SRT if available and valid
        srt_to_use = translated_srt_path if (
            os.path.exists(translated_srt_path) and 
            os.path.getsize(translated_srt_path) > 0
        ) else srt_path
        
        _add_captions_to_video(temp_input_path, srt_to_use, output_path, font_options, job_id_log_prefix)
        
        logger.info(f"{job_id_log_prefix} Processing completed successfully")
        
        # Create response
        response = make_response(send_file(
            output_path,
            as_attachment=True,
            download_name=f"captioned_{original_filename}"
        ))
        
        # Add headers
        response.headers['X-Video-Duration-Seconds'] = str(int(duration))
        response.headers['X-GPU-Used'] = device
        response.headers['X-Processing-Device'] = device
        response.headers['Content-Type'] = 'video/mp4'
        
        return response
        
    except ValueError as ve:
        logger.error(f"{job_id_log_prefix} Validation error: {ve}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Processing error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Cleanup
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"{job_id_log_prefix} Cleaned up: {file_path}")
                except OSError as e:
                    logger.warning(f"{job_id_log_prefix} Cleanup failed: {e}")
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
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
        'whisper_model_size': whisper_model_size,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }

    if device == "cuda" and torch.cuda.is_available():
        try:
            status.update({
                'gpu_name': torch.cuda.get_device_properties(0).name,
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(0),
                'gpu_memory_cached': torch.cuda.memory_reserved(0),
                'current_device': torch.cuda.current_device(),
                'device_count': torch.cuda.device_count()
            })
            
            # Check if model is actually on GPU
            if whisper_model is not None:
                model_device = next(whisper_model.parameters()).device
                status['model_device'] = str(model_device)
                status['model_on_gpu'] = model_device.type == 'cuda'
            
        except Exception as e:
            status['gpu_error'] = str(e)

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
        
        # Ensure GPU is ready before processing
        if device == "cuda":
            torch.cuda.empty_cache()  # Clear cache before processing
            ensure_model_on_gpu()

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
        
        # Clear GPU cache after processing
        if device == "cuda":
            torch.cuda.empty_cache()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'message': 'Video Captioning Service is running.',
        'gpu_available': torch.cuda.is_available(),
        'device': device,
        'model_loaded': whisper_model is not None,
        'model_device': str(next(whisper_model.parameters()).device) if whisper_model else None
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
    # Test GPU availability first
    logger.info("=== GPU Setup Test ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_properties(0).name}")
    
    # Preload Whisper model on startup
    try:
        load_whisper_model()
        logger.info("Whisper model preloaded successfully")
        if whisper_model and device == "cuda":
            model_device = next(whisper_model.parameters()).device
            logger.info(f"Model confirmed on device: {model_device}")
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}")
    
    port = 5003
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true', threaded=True)