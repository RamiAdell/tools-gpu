import multiprocessing 
# MUST be before any CUDA imports
 
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
import os
import uuid
import json
import logging
import tempfile 
import time
import threading
import shutil
import psutil
from utils.utils import allowed_file, get_video_duration, extract_audio, audio_to_text, translate_srt, add_captions_to_video
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


# GPU Support - import after multiprocessing setup
import torch
import torch.backends.cudnn as cudnn

# Optimize environment variables for GPU performance
os.environ['OMP_NUM_THREADS'] = '4'  # Increased for better CPU utilization
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async GPU operations
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GPU Configuration with improved error handling ---
device = "cpu"  # Default fallback
gpu_info = None
cuda_available = False
gpu_memory_fraction = 0.85  # Use more GPU memory for better performance

def initialize_cuda():
    """Initialize CUDA with proper error handling and optimization"""
    global device, gpu_info, cuda_available
    
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Test CUDA initialization
            torch.cuda.init()
            
            # Enable optimizations
            cudnn.benchmark = True  # Optimize for consistent input sizes
            cudnn.deterministic = False  # Allow non-deterministic for speed
            
            device = "cuda"
            gpu_info = torch.cuda.get_device_properties(0)
            
            logger.info(f"GPU detected: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.1f} GB memory")
            logger.info(f"GPU compute capability: {gpu_info.major}.{gpu_info.minor}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"GPU multiprocessors: {gpu_info.multi_processor_count}")
            
            # Set memory allocation strategy for GPU
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            # Warm up GPU with a test operation
            logger.info("Warming up GPU...")
            test_tensor = torch.randn(1000, 1000, device='cuda')
            _ = torch.mm(test_tensor, test_tensor.t())
            del test_tensor
            torch.cuda.empty_cache()
            
            logger.info("CUDA initialization and warmup successful")
            
        else:
            logger.warning("No GPU detected, using CPU for processing")
            device = "cpu"
            
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        logger.warning("Falling back to CPU processing")
        device = "cpu"
        cuda_available = False

# Initialize CUDA outside of request context
initialize_cuda()

# --- Configuration (optimized for GPU deployment) ---
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/caption_uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', '/tmp/caption_processed')
FONT_FOLDER = os.getenv('FONT_FOLDER', '/app/fonts')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'mkv', 'avi'}
SERVICE_API_KEY = os.getenv('CAPTION_SERVICE_API_KEY', "YourSecretApiKeyForCaptionService123")
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '500')) * 1024 * 1024  # Increased for GPU processing
MAX_VIDEO_DURATION_SECONDS = int(os.getenv('MAX_VIDEO_DURATION_SECONDS', '600'))  # Increased for GPU

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
    "/System/Library/Fonts/Arial.ttf",  # macOS fallback
]

logger.debug(f"Looking for fonts in: {FONT_FOLDER}")
if os.path.exists(FONT_FOLDER):
    logger.debug(f"Font files present: {os.listdir(FONT_FOLDER)}")

# Global Whisper model - load once and reuse
whisper_model = None
whisper_model_size = os.getenv('WHISPER_MODEL_SIZE', 'medium')  # Use medium for better accuracy on GPU
whisper_model_lock = threading.Lock()

def load_whisper_model():
    """Load Whisper model with GPU support if available"""
    global whisper_model
    
    with whisper_model_lock:
        if whisper_model is None:
            try:
                logger.info(f"Loading Whisper model '{whisper_model_size}' on device: {device}")
                
                if device == "cuda" and cuda_available:
                    try:
                        # Ensure CUDA is properly initialized
                        torch.cuda.empty_cache()
                        torch.cuda.init()
                        
                        # Set the device explicitly
                        torch.cuda.set_device(0)
                        logger.info(f"Set CUDA device to: {torch.cuda.current_device()}")
                        
                        # Load model with explicit device parameter
                        whisper_model = whisper.load_model(whisper_model_size, device="cuda")
                        
                        # Verify the model is on GPU
                        model_device = next(whisper_model.parameters()).device
                        logger.info(f"Whisper model loaded on device: {model_device}")
                        
                        if model_device.type != 'cuda':
                            logger.warning(f"Model loaded on {model_device} instead of CUDA, moving to GPU")
                            whisper_model = whisper_model.cuda()
                            logger.info(f"Moved model to GPU: {next(whisper_model.parameters()).device}")
                        
                        # Optimize model for inference
                        whisper_model.eval()
                        
                        # Test GPU memory after loading
                        logger.info(f"GPU memory after model loading:")
                        logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                        logger.info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                        
                        # Warm up the model with a dummy input
                        logger.info("Warming up Whisper model...")
                        dummy_audio = torch.zeros(16000, device="cuda")  # 1 second of audio
                        with torch.no_grad():
                            _ = whisper_model.transcribe(dummy_audio.cpu().numpy(), fp16=True, verbose=False)
                        torch.cuda.empty_cache()
                        logger.info("Model warmup complete")
                        
                    except Exception as cuda_error:
                        logger.error(f"Failed to load Whisper model on GPU: {cuda_error}")
                        logger.info("Falling back to CPU")
                        whisper_model = whisper.load_model(whisper_model_size, device="cpu")
                        
                else:
                    whisper_model = whisper.load_model(whisper_model_size, device="cpu")
                    logger.info("Whisper model loaded on CPU")

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise e
                
        return whisper_model

def ensure_model_on_gpu():
    """Ensure the Whisper model is on GPU before inference"""
    global whisper_model
    if whisper_model is not None and device == "cuda" and cuda_available:
        try:
            model_device = next(whisper_model.parameters()).device
            if model_device.type != 'cuda':
                logger.info(f"Moving model from {model_device} to GPU")
                whisper_model = whisper_model.cuda()
                logger.info(f"Model now on: {next(whisper_model.parameters()).device}")
        except Exception as e:
            logger.error(f"Error ensuring model on GPU: {e}")

# --- Helper Functions ---
def audio_to_text(wav_path, srt_path, job_id_log_prefix=""):
    """Transcribe audio with GPU optimization"""
    logger.info(f"{job_id_log_prefix} Transcribing {wav_path} to {srt_path} using {device}")
    
    try:
        # Load model if not already loaded
        model = load_whisper_model()
        ensure_model_on_gpu()
        
        # Log GPU status before transcription
        if device == "cuda" and cuda_available:
            logger.info(f"{job_id_log_prefix} GPU memory before transcription:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            logger.info(f"  Model device: {next(model.parameters()).device}")
        
        # Transcribe with GPU acceleration and optimization
        start_time = time.time()
        
        # Optimized transcription parameters for GPU
        transcribe_options = {
            'fp16': device == "cuda" and cuda_available,  # Use FP16 on GPU for speed
            'verbose': False,
            'language': None,  # Auto-detect language
            'beam_size': 5 if device == "cuda" else 1,  # Larger beam size on GPU
            'best_of': 5 if device == "cuda" else 1,  # Multiple candidates on GPU
            'temperature': 0.0,  # Deterministic output
        }
        
        # Use context manager for memory management
        with torch.no_grad():
            result = model.transcribe(wav_path, **transcribe_options)
        
        processing_time = time.time() - start_time
        logger.info(f"{job_id_log_prefix} Transcription completed in {processing_time:.2f}s using {device}")
        
        # Write SRT file with better formatting
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start, end, text = segment['start'], segment['end'], segment['text'].strip()
                if text:  # Only write non-empty segments
                    f.write(f"{i+1}\n{format_whisper_timestamp(start)} --> {format_whisper_timestamp(end)}\n{text}\n\n")
        
        logger.info(f"{job_id_log_prefix} Transcription complete. SRT saved to {srt_path}")
        
        # Clean up GPU memory
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()
            logger.info(f"{job_id_log_prefix} GPU memory after cleanup:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error during transcription: {str(e)}", exc_info=True)
        # Clean up on error
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()
        raise

def format_whisper_timestamp(seconds):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000; milliseconds %= 3_600_000
    minutes = milliseconds // 60_000; milliseconds %= 60_000
    seconds = milliseconds // 1_000; milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def get_system_stats():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        'cpu_percent': cpu_percent,
        'memory_total_gb': memory.total / 1024**3,
        'memory_used_gb': memory.used / 1024**3,
        'memory_available_gb': memory.available / 1024**3,
        'memory_percent': memory.percent,
        'disk_total_gb': disk.total / 1024**3,
        'disk_used_gb': disk.used / 1024**3,
        'disk_free_gb': disk.free / 1024**3
    }
    
    if device == "cuda" and cuda_available:
        try:
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_reserved(0) / 1024**3,
                'gpu_memory_total_gb': gpu_info.total_memory / 1024**3 if gpu_info else 0
            })
        except:
            pass
    
    return stats


@app.before_request
def verify_api_key_middleware():
    if request.path in ['/health', '/gpu-status', '/system-stats']:
        return
    api_key = request.headers.get('X-Api-Key')
    if api_key != SERVICE_API_KEY:
        logger.warning(f"Unauthorized API key from {request.remote_addr} to {request.path}")
        return jsonify({'error': 'Unauthorized: Invalid API Key'}), 401

# --- Flask Routes ---
@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """Get comprehensive GPU status and information"""
    status = {
        'gpu_available': cuda_available,
        'device': device,
        'whisper_model_loaded': whisper_model is not None,
        'whisper_model_size': whisper_model_size,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': cudnn.version() if cudnn.is_available() else None,
        'gpu_memory_fraction': gpu_memory_fraction
    }

    if device == "cuda" and cuda_available:
        try:
            status.update({
                'gpu_name': gpu_info.name if gpu_info else "Unknown",
                'gpu_memory_total_gb': gpu_info.total_memory / 1024**3 if gpu_info else 0,
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_reserved(0) / 1024**3,
                'gpu_memory_free_gb': (gpu_info.total_memory - torch.cuda.memory_reserved(0)) / 1024**3 if gpu_info else 0,
                'current_device': torch.cuda.current_device(),
                'device_count': torch.cuda.device_count(),
                'compute_capability': f"{gpu_info.major}.{gpu_info.minor}" if gpu_info else None,
                'multiprocessor_count': gpu_info.multi_processor_count if gpu_info else None
            })
            
            # Check if model is actually on GPU
            if whisper_model is not None:
                model_device = next(whisper_model.parameters()).device
                status['model_device'] = str(model_device)
                status['model_on_gpu'] = model_device.type == 'cuda'
            
        except Exception as e:
            status['gpu_error'] = str(e)
            logger.error(f"Error getting GPU status: {e}")

    return jsonify(status), 200

@app.route('/system-stats', methods=['GET'])
def system_stats():
    """Get comprehensive system statistics"""
    try:
        stats = get_system_stats()
        stats['whisper_model_loaded'] = whisper_model is not None
        stats['device'] = device
        stats['timestamp'] = time.time()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({'error': 'Failed to get system stats'}), 500

 
@app.route('/process_direct', methods=['POST'])
def process_video_directly():
    """
    Single endpoint to receive video and parameters, process it, and stream back the result.
    """
    user_id = request.headers.get('X-User-ID', 'unknown_user')
    job_id_log_prefix = f"[DirectProcess-{user_id}-{uuid.uuid4().hex[:8]}]"
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
        if duration_seconds == 0 and file_size > 1000:
             raise ValueError('Invalid video file or could not determine duration.')
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f'Video too long (max {MAX_VIDEO_DURATION_SECONDS // 60} minutes).')

        logger.info(f"{job_id_log_prefix} Video validated. Duration: {duration_seconds:.2f}s. Starting pipeline on {device}.")
        
        # Ensure GPU is ready before processing
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()
            ensure_model_on_gpu()

        # --- Execute processing pipeline synchronously ---
        extract_audio(temp_input_filepath, audio_file_path, job_id_log_prefix)
        audio_to_text(audio_file_path, srt_file_path, job_id_log_prefix)
        translate_srt(srt_file_path, language, translated_srt_file_path, job_id_log_prefix)
        
        srt_to_use = translated_srt_file_path if os.path.exists(translated_srt_file_path) and os.path.getsize(translated_srt_file_path) > 0 else srt_file_path
        add_captions_to_video(temp_input_filepath, srt_to_use, final_output_video_path, font_options, job_id_log_prefix)
        # --- End of pipeline ---

        logger.info(f"{job_id_log_prefix} Processing complete. Output: {final_output_video_path}")
        
        response = make_response(send_file(
            final_output_video_path,
            as_attachment=True,
            download_name=f"captioned_{original_secure_filename}"
        ))
        # Send video duration back to Django for usage tracking
        response.headers['X-Video-Duration-Seconds'] = str(int(duration_seconds))
        response.headers['X-GPU-Used'] = str(device)
        response.headers['Content-Type'] = 'video/mp4'

        return response

    except ValueError as ve:
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
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'message': 'Video Captioning Service is running.',
        'gpu_available': cuda_available,
        'device': device,
        'model_loaded': whisper_model is not None,
        'model_device': str(next(whisper_model.parameters()).device) if whisper_model else None
    }), 200

@app.route('/progress', methods=['GET'])
def stream_progress_sse():
    """Server-Sent Events endpoint for real-time progress updates (if used)."""
    target_filename = request.args.get('filename')

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
    logger.info(f"Using device: {device}")
    if cuda_available:
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)