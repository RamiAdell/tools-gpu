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
from concurrent.futures import ThreadPoolExecutor
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

# PERFORMANCE OPTIMIZATIONS FOR HIGH-SPEC VM
# Optimize environment variables for GPU performance
os.environ['OMP_NUM_THREADS'] = '14'  # Half of your 28 CPUs for balanced performance
os.environ['MKL_NUM_THREADS'] = '14'
os.environ['NUMEXPR_NUM_THREADS'] = '14'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async GPU operations
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,garbage_collection_threshold:0.8'  # Larger split for your 58GB RAM
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 optimizations
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Memory optimization for large RAM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] += ',expandable_segments:True'

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# Configure Flask for high-performance
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # Cache control

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced GPU Configuration ---
device = "cpu"  # Default fallback
gpu_info = None
cuda_available = False
gpu_memory_fraction = 0.90  # Use 90% of GPU memory for better performance on dedicated VM

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

def initialize_cuda():
    """Initialize CUDA with optimizations for high-performance VM"""
    global device, gpu_info, cuda_available
    
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Test CUDA initialization
            torch.cuda.init()
            
            # Enable advanced optimizations
            cudnn.benchmark = True  # Optimize for consistent input sizes
            cudnn.deterministic = False  # Allow non-deterministic for speed
            cudnn.allow_tf32 = True  # Enable TF32 for faster training/inference
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            device = "cuda"
            gpu_info = torch.cuda.get_device_properties(0)
            
            logger.info(f"GPU detected: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.1f} GB memory")
            logger.info(f"GPU compute capability: {gpu_info.major}.{gpu_info.minor}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"GPU multiprocessors: {gpu_info.multi_processor_count}")
            
            # Aggressive memory allocation for dedicated VM
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            # Pre-allocate memory pool for faster allocations
            torch.cuda.empty_cache()
            
            # Warm up GPU with larger test operation for high-spec VM
            logger.info("Warming up GPU with intensive operation...")
            test_tensor = torch.randn(2000, 2000, device='cuda')
            for _ in range(3):  # Multiple warmup iterations
                _ = torch.mm(test_tensor, test_tensor.t())
            del test_tensor
            torch.cuda.empty_cache()
            
            # Set GPU frequency scaling for maximum performance
            try:
                os.system("nvidia-smi -pl 400")  # Set power limit to maximum
                os.system("nvidia-smi -ac 877,1215")  # Set memory and graphics clocks to maximum
            except:
                logger.info("Could not set GPU performance mode (non-critical)")
            
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

# --- Configuration (optimized for high-performance VM) ---
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/caption_uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', '/tmp/caption_processed')
FONT_FOLDER = os.getenv('FONT_FOLDER', '/app/fonts')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'mkv', 'avi', 'flv', 'm4v'}
SERVICE_API_KEY = os.getenv('CAPTION_SERVICE_API_KEY', "YourSecretApiKeyForCaptionService123")
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '2048')) * 1024 * 1024  # 2GB for high-spec VM
MAX_VIDEO_DURATION_SECONDS = int(os.getenv('MAX_VIDEO_DURATION_SECONDS', '1800'))  # 30 minutes for high-spec VM

# Use ramdisk for temporary files on high-RAM system
RAMDISK_PATH = '/dev/shm'
if os.path.exists(RAMDISK_PATH) and os.access(RAMDISK_PATH, os.W_OK):
    UPLOAD_FOLDER = os.path.join(RAMDISK_PATH, 'caption_uploads')
    PROCESSED_FOLDER = os.path.join(RAMDISK_PATH, 'caption_processed')
    logger.info("Using ramdisk for temporary files - significant speed boost!")

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

# Multiple Whisper models for different use cases
whisper_models = {}
whisper_model_sizes = ['large-v2', 'medium', 'small']  # Load multiple models
whisper_model_lock = threading.Lock()

def load_whisper_models():
    """Load multiple Whisper models with GPU support"""
    global whisper_models
    
    with whisper_model_lock:
        for model_size in whisper_model_sizes:
            if model_size not in whisper_models:
                try:
                    logger.info(f"Loading Whisper model '{model_size}' on device: {device}")
                    
                    if device == "cuda" and cuda_available:
                        try:
                            # Ensure CUDA is properly initialized
                            torch.cuda.empty_cache()
                            torch.cuda.init()
                            
                            # Set the device explicitly
                            torch.cuda.set_device(0)
                            
                            # Load model with explicit device parameter
                            model = whisper.load_model(model_size, device="cuda")
                            
                            # Verify the model is on GPU
                            model_device = next(model.parameters()).device
                            logger.info(f"Whisper model {model_size} loaded on device: {model_device}")
                            
                            if model_device.type != 'cuda':
                                logger.warning(f"Model {model_size} loaded on {model_device} instead of CUDA, moving to GPU")
                                model = model.cuda()
                                logger.info(f"Moved model {model_size} to GPU: {next(model.parameters()).device}")
                            
                            # Optimize model for inference
                            model.eval()
                            model.half()  # Use FP16 for faster inference
                            
                            whisper_models[model_size] = model
                            
                            # Warm up each model
                            logger.info(f"Warming up Whisper model {model_size}...")
                            dummy_audio = torch.zeros(16000, device="cuda")  # 1 second of audio
                            with torch.no_grad():
                                _ = model.transcribe(dummy_audio.cpu().numpy(), fp16=True, verbose=False)
                            torch.cuda.empty_cache()
                            logger.info(f"Model {model_size} warmup complete")
                            
                        except Exception as cuda_error:
                            logger.error(f"Failed to load Whisper model {model_size} on GPU: {cuda_error}")
                            logger.info("Falling back to CPU")
                            whisper_models[model_size] = whisper.load_model(model_size, device="cpu")
                            
                    else:
                        whisper_models[model_size] = whisper.load_model(model_size, device="cpu")
                        logger.info(f"Whisper model {model_size} loaded on CPU")

                except Exception as e:
                    logger.error(f"Failed to load Whisper model {model_size}: {e}")
                    continue

def get_optimal_model_size(duration_seconds):
    """Select optimal Whisper model based on video duration and available resources"""
    if duration_seconds <= 60:  # Short videos
        return 'small'
    elif duration_seconds <= 300:  # Medium videos (5 minutes)
        return 'medium'
    else:  # Long videos
        return 'large-v2'

def ensure_model_on_gpu(model):
    """Ensure the Whisper model is on GPU before inference"""
    if model is not None and device == "cuda" and cuda_available:
        try:
            model_device = next(model.parameters()).device
            if model_device.type != 'cuda':
                logger.info(f"Moving model from {model_device} to GPU")
                model = model.cuda().half()
                logger.info(f"Model now on: {next(model.parameters()).device}")
        except Exception as e:
            logger.error(f"Error ensuring model on GPU: {e}")
    return model

# --- Enhanced Helper Functions ---
def audio_to_text_optimized(wav_path, srt_path, duration_seconds, job_id_log_prefix=""):
    """Enhanced transcription with model selection and GPU optimization"""
    logger.info(f"{job_id_log_prefix} Transcribing {wav_path} to {srt_path} using {device}")
    
    try:
        # Select optimal model based on duration
        model_size = get_optimal_model_size(duration_seconds)
        
        # Load models if not already loaded
        if not whisper_models:
            load_whisper_models()
        
        model = whisper_models.get(model_size)
        if model is None:
            # Fallback to any available model
            model = next(iter(whisper_models.values()))
            logger.warning(f"Fallback to available model instead of {model_size}")
        
        model = ensure_model_on_gpu(model)
        
        # Log GPU status before transcription
        if device == "cuda" and cuda_available:
            logger.info(f"{job_id_log_prefix} GPU memory before transcription:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            logger.info(f"  Model device: {next(model.parameters()).device}")
        
        # Enhanced transcription parameters for high-performance VM
        start_time = time.time()
        
        transcribe_options = {
            'fp16': device == "cuda" and cuda_available,
            'verbose': False,
            'language': None,  # Auto-detect language
            'beam_size': 8 if device == "cuda" else 1,  # Larger beam size for accuracy
            'best_of': 8 if device == "cuda" else 1,  # Multiple candidates
            'temperature': 0.0,  # Deterministic output
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'condition_on_previous_text': True,
        }
        
        # Use mixed precision and optimized memory usage
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cuda_available):
                result = model.transcribe(wav_path, **transcribe_options)
        
        processing_time = time.time() - start_time
        logger.info(f"{job_id_log_prefix} Transcription completed in {processing_time:.2f}s using {device} with model {model_size}")
        
        # Write SRT file with enhanced formatting
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start, end, text = segment['start'], segment['end'], segment['text'].strip()
                if text and len(text) > 1:  # Filter out very short segments
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
    """Enhanced system resource monitoring"""
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)  # Per-CPU stats
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        'cpu_percent_avg': sum(cpu_percent) / len(cpu_percent),
        'cpu_percent_per_core': cpu_percent,
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': memory.total / 1024**3,
        'memory_used_gb': memory.used / 1024**3,
        'memory_available_gb': memory.available / 1024**3,
        'memory_percent': memory.percent,
        'disk_total_gb': disk.total / 1024**3,
        'disk_used_gb': disk.used / 1024**3,
        'disk_free_gb': disk.free / 1024**3,
        'active_models': list(whisper_models.keys()),
        'using_ramdisk': UPLOAD_FOLDER.startswith('/dev/shm')
    }
    
    if device == "cuda" and cuda_available:
        try:
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_reserved(0) / 1024**3,
                'gpu_memory_total_gb': gpu_info.total_memory / 1024**3 if gpu_info else 0,
                'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
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
    """Enhanced GPU status with performance metrics"""
    status = {
        'gpu_available': cuda_available,
        'device': device,
        'whisper_models_loaded': list(whisper_models.keys()),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': cudnn.version() if cudnn.is_available() else None,
        'gpu_memory_fraction': gpu_memory_fraction,
        'tf32_enabled': torch.backends.cuda.matmul.allow_tf32 if cuda_available else False,
        'cudnn_benchmark': cudnn.benchmark if cuda_available else False
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
            
            # Check if models are actually on GPU
            models_status = {}
            for model_name, model in whisper_models.items():
                if model is not None:
                    model_device = next(model.parameters()).device
                    models_status[model_name] = {
                        'device': str(model_device),
                        'on_gpu': model_device.type == 'cuda',
                        'dtype': str(next(model.parameters()).dtype)
                    }
            status['models_status'] = models_status
            
        except Exception as e:
            status['gpu_error'] = str(e)
            logger.error(f"Error getting GPU status: {e}")

    return jsonify(status), 200

@app.route('/system-stats', methods=['GET'])
def system_stats():
    """Enhanced system statistics"""
    try:
        stats = get_system_stats()
        stats['whisper_models_loaded'] = list(whisper_models.keys())
        stats['device'] = device
        stats['timestamp'] = time.time()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({'error': 'Failed to get system stats'}), 500

@app.route('/process_direct', methods=['POST'])
def process_video_directly():
    """
    Enhanced single endpoint with performance optimizations for high-spec VM
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
        
        # Performance options
        quality_mode = request.form.get('quality_mode', 'balanced')  # fast, balanced, high
        
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
        # Save file with optimized I/O
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

        logger.info(f"{job_id_log_prefix} Video validated. Duration: {duration_seconds:.2f}s. Starting enhanced pipeline on {device}.")
        
        # Pre-load optimal model
        if not whisper_models:
            load_whisper_models()
        
        # Ensure GPU is ready before processing
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()
            # Pre-warm GPU
            dummy_tensor = torch.randn(100, 100, device='cuda')
            _ = torch.mm(dummy_tensor, dummy_tensor.t())
            del dummy_tensor
            torch.cuda.empty_cache()

        # --- Execute enhanced processing pipeline ---
        pipeline_start = time.time()
        
        # Step 1: Audio extraction (parallel-ready)
        extract_audio(temp_input_filepath, audio_file_path, job_id_log_prefix)
        
        # Step 2: Enhanced transcription with model selection
        audio_to_text_optimized(audio_file_path, srt_file_path, duration_seconds, job_id_log_prefix)
        
        # Step 3: Translation (if needed)
        translate_srt(srt_file_path, language, translated_srt_file_path, job_id_log_prefix)
        
        # Step 4: Video processing with captions
        srt_to_use = translated_srt_file_path if os.path.exists(translated_srt_file_path) and os.path.getsize(translated_srt_file_path) > 0 else srt_file_path
        add_captions_to_video(temp_input_filepath, srt_to_use, final_output_video_path, font_options, job_id_log_prefix)
        
        pipeline_time = time.time() - pipeline_start
        logger.info(f"{job_id_log_prefix} Enhanced pipeline completed in {pipeline_time:.2f}s")
        # --- End of pipeline ---

        logger.info(f"{job_id_log_prefix} Processing complete. Output: {final_output_video_path}")
        
        response = make_response(send_file(
            final_output_video_path,
            as_attachment=True,
            download_name=f"captioned_{original_secure_filename}"
        ))
        # Enhanced response headers
        response.headers['X-Video-Duration-Seconds'] = str(int(duration_seconds))
        response.headers['X-GPU-Used'] = str(device)
        response.headers['X-Processing-Time'] = str(int(pipeline_time))
        response.headers['X-Model-Used'] = get_optimal_model_size(duration_seconds)
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'

        return response

    except ValueError as ve:
        logger.error(f"{job_id_log_prefix} Validation error: {str(ve)}", exc_info=True)
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error during direct processing: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'An internal error occurred: {str(e)}'}), 500
    finally:
        # Enhanced cleanup with better error handling
        cleanup_start = time.time()
        for f_path in files_to_cleanup:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logger.debug(f"{job_id_log_prefix} Cleaned up: {f_path}")
                except OSError as e_clean:
                    logger.warning(f"{job_id_log_prefix} Could not clean up file {f_path}: {e_clean}")
        
        # Aggressive GPU memory cleanup
        if device == "cuda" and cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete
        
        cleanup_time = time.time() - cleanup_start
        logger.debug(f"{job_id_log_prefix} Cleanup completed in {cleanup_time:.2f}s")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'message': 'Enhanced Video Captioning Service is running.',
        'gpu_available': cuda_available,
        'device': device,
        'models_loaded': list(whisper_models.keys()),
        'performance_mode': 'high-performance',
        'ramdisk_enabled': UPLOAD_FOLDER.startswith('/dev/shm')
    }), 200


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