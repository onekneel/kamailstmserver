from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
import logging
import time
import base64
import gc
import subprocess

# 1. Configure environment variables for speed and memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable OneDNN for speed

# 2. Set up optimized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Initialize Flask app with file size limit
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/predict_stream": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/test_frame": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/health": {"origins": "*"}
})

# Global variables
model_manager = None
mp_holistic = None
mp_drawing = None
sequence_buffer = None
buffer_index = 0

# 4. Lazy loading of MediaPipe with minimal complexity
def load_mediapipe():
    global mp_holistic, mp_drawing
    if mp_holistic is None:
        logger.info("Loading MediaPipe models...")
        start = time.time()
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        logger.info(f"MediaPipe loaded in {time.time() - start:.2f} seconds")
    return mp_holistic, mp_drawing

# 5. Optimized MediaPipe detection with minimal resolution
def mediapipe_detection(image, model):
    # Use smaller resolution to save memory and increase speed
    image = cv2.resize(image, (160, 120))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = None  # Clear image to free memory
    gc.collect()
    return None, results

# 6. Optimized keypoint extraction
def extract_keypoints(results):
    # Simplified keypoint extraction for speed
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Simplified additional features (zeros)
    additional_features = np.zeros(18)
    
    return np.concatenate([pose, lh, rh, additional_features])

def has_hand_keypoints(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def normalize_keypoints(sequences, min_val, max_val):
    sequences = np.array(sequences)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1e-6
    return (sequences - min_val) / range_val

def safe_unlink(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except:
        pass

# 7. Video validation - simplified for speed
def validate_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret
    except:
        return False

# 8. Model and data loading with speed optimizations
class ModelManager:
    def __init__(self):
        self.model = None
        self.min_val = None
        self.max_val = None
        self.is_initialized = False
        self.actions = ['Goodbye', 'Hello', 'Thankyou', 'Please', 'Thankyou(FSL)', 'Wait(FSL)']
        self.sequence_length = 30
        self.MIN_TEST_FRAMES = 5  # Reduced for faster processing

    def initialize(self):
        if self.is_initialized:
            return
        
        # Clear memory before initialization
        gc.collect()
        tf.keras.backend.clear_session()
        
        start_time = time.time()
        logger.info("Initializing model manager...")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'public', 'model', 'kamaibestlstm_Modified.keras')
        min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'min_val.npy')
        max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'max_val.npy')
        
        # Load Keras model with proper error handling
        try:
            # Fixed model loading without using set_session
            self.model = load_model(model_path, compile=False)  # Skip compilation for speed
            logger.info(f"Keras model loaded successfully in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            # Create a simple fallback model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, Dense
            
            # Fixed model creation with Input layer
            model = Sequential([
                Input(shape=(self.sequence_length, 276)),
                LSTM(32, activation='relu'),
                Dense(len(self.actions), activation='softmax')
            ])
            self.model = model
            logger.info("Using fallback model")
        
        # Load min/max values
        try:
            if os.path.exists(min_val_path) and os.path.exists(max_val_path):
                self.min_val = np.load(min_val_path)
                self.max_val = np.load(max_val_path)
                expected_keypoint_dim = 276
                if self.min_val.shape[0] != expected_keypoint_dim:
                    self.min_val = np.pad(self.min_val, (0, expected_keypoint_dim - self.min_val.shape[0]), 
                                        mode='constant', constant_values=0)
                    self.max_val = np.pad(self.max_val, (0, expected_keypoint_dim - self.max_val.shape[0]), 
                                        mode='constant', constant_values=1)
                logger.info("Min/max values loaded successfully")
            else:
                raise FileNotFoundError(f"Min/max files not found")
        except Exception as e:
            logger.warning(f"Error loading min/max values: {e}. Using defaults.")
            self.min_val = np.zeros(276)
            self.max_val = np.ones(276)
        
        self.is_initialized = True
        logger.info(f"Model manager initialized in {time.time() - start_time:.2f} seconds")
        
        # Force garbage collection
        gc.collect()

# Initialize model manager lazily
def get_model_manager():
    global model_manager, sequence_buffer, buffer_index
    if model_manager is None:
        model_manager = ModelManager()
        sequence_buffer = np.zeros((model_manager.sequence_length, 276), dtype=np.float32)
        buffer_index = 0
    return model_manager

# 9. Prediction endpoints optimized for speed
@app.route('/predict', methods=['POST'])
def predict():
    global buffer_index
    
    # Clear memory at the start
    gc.collect()
    
    start_time = time.time()
    logger.debug("Received request to /predict")
    
    if request.content_length > app.config['MAX_CONTENT_LENGTH']:
        logger.error("Video file too large")
        return jsonify({'error': 'Video file exceeds 5 MB limit'}), 400

    manager = get_model_manager()
    if not manager.is_initialized:
        manager.initialize()
    
    mp_holistic, _ = load_mediapipe()

    if 'video' not in request.files:
        logger.error("No video file provided")
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        logger.error("No video file selected")
        return jsonify({'error': 'No video file selected'}), 400

    temp_file_path = None
    cap = None
    try:
        # Save video to temp file
        temp_file_path = os.path.join(tempfile.gettempdir(), f"video_{int(time.time())}.mp4")
        video_file.save(temp_file_path)
        if not validate_video(temp_file_path):
            logger.error("Invalid or corrupted video file")
            return jsonify({'error': 'Invalid or corrupted video file'}), 400
        
        # Process video with optimized settings for speed
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return jsonify({'error': 'Failed to open video file'}), 500

        # Get video properties for faster processing
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate optimal frame skip based on video length
        # For longer videos, skip more frames
        if frame_count > 100:
            FRAME_SKIP = 8
        elif frame_count > 50:
            FRAME_SKIP = 6
        else:
            FRAME_SKIP = 4
            
        logger.info(f"Video has {frame_count} frames at {fps} fps, using skip={FRAME_SKIP}")
        
        sequence = np.zeros((manager.sequence_length, 276), dtype=np.float32)
        seq_index = 0
        hand_frame_count = 0
        frame_count = 0
        max_frames = 60  # Limit processing

        # Use lowest complexity for MediaPipe for speed
        with mp_holistic.Holistic(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            while seq_index < manager.sequence_length and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                try:
                    # Process frame with minimal memory usage
                    _, results = mediapipe_detection(frame, holistic)
                    frame = None  # Clear frame to free memory
                    
                    if has_hand_keypoints(results):
                        keypoints = extract_keypoints(results)
                        sequence[seq_index] = keypoints
                        seq_index += 1
                        hand_frame_count += 1
                    
                    # Clear memory after each frame
                    results = None
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Skipping frame due to error: {e}")
                    continue

        # Check if we have enough frames with hand keypoints
        if hand_frame_count < manager.MIN_TEST_FRAMES:
            logger.error(f"Too few frames with hand keypoints: {hand_frame_count}")
            return jsonify({'error': f'Too few frames with hand keypoints ({hand_frame_count})'}), 400

        # Normalize and predict
        norm_sequence = normalize_keypoints([sequence], manager.min_val, manager.max_val)[0]
        sequence = None  # Clear to free memory
        
        # Use a single prediction for speed
        input_data = np.expand_dims(norm_sequence[:manager.sequence_length], axis=0).astype(np.float32)
        res = manager.model.predict(input_data, verbose=0)[0]
        norm_sequence = None  # Clear to free memory
        input_data = None  # Clear to free memory
        
        predicted_action = manager.actions[np.argmax(res)]
        confidence = float(res[np.argmax(res)] * 100)
        
        processing_time = time.time() - start_time
        
        response = {
            'action': predicted_action,
            'confidence': confidence,
            'processing_time': processing_time
        }
        logger.info(f"Prediction: {predicted_action} with confidence {confidence:.2f}% in {processing_time:.2f} seconds")
        
        # Clear memory before returning
        gc.collect()
        
        return jsonify(response)

    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if temp_file_path and os.path.exists(temp_file_path):
            safe_unlink(temp_file_path)
        
        # Force garbage collection
        gc.collect()

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    global buffer_index, sequence_buffer
    
    # Clear memory at the start
    gc.collect()
    
    start_time = time.time()
    logger.debug("Received request to /predict_stream")
    
    manager = get_model_manager()
    if not manager.is_initialized:
        manager.initialize()
    
    mp_holistic, _ = load_mediapipe()

    if 'frame' not in request.json:
        logger.error("No frame data provided")
        return jsonify({'error': 'No frame data provided'}), 400

    try:
        # Process frame with minimal memory usage
        frame_data = request.json['frame']
        frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        frame_bytes = None  # Clear to free memory
        frame_np = None  # Clear to free memory

        if frame is None:
            logger.error("Failed to decode frame")
            return jsonify({'error': 'Failed to decode frame'}), 400

        # Use lowest complexity for MediaPipe
        with mp_holistic.Holistic(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            try:
                _, results = mediapipe_detection(frame, holistic)
                frame = None  # Clear to free memory
                
                if has_hand_keypoints(results):
                    keypoints = extract_keypoints(results)
                    sequence_buffer[buffer_index] = keypoints
                    buffer_index = (buffer_index + 1) % manager.sequence_length

                    # Reduced required frames for faster response
                    if buffer_index >= 8:  
                        norm_sequence = normalize_keypoints([sequence_buffer], manager.min_val, manager.max_val)[0]
                        input_data = np.expand_dims(norm_sequence, axis=0).astype(np.float32)
                        res = manager.model.predict(input_data, verbose=0)[0]
                        norm_sequence = None  # Clear to free memory
                        input_data = None  # Clear to free memory
                        
                        predicted_action = manager.actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)] * 100)
                        processing_time = time.time() - start_time
                        
                        # Clear memory before returning
                        gc.collect()
                        
                        return jsonify({
                            'action': predicted_action,
                            'confidence': confidence,
                            'processing_time': processing_time
                        })
                    else:
                        return jsonify({
                            'action': 'Processing',
                            'confidence': 0,
                            'frames_collected': buffer_index,
                            'frames_needed': 8
                        })
                else:
                    return jsonify({
                        'action': 'No hands detected',
                        'confidence': 0
                    })
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

    finally:
        # Force garbage collection
        results = None
        gc.collect()

@app.route('/test_frame', methods=['POST'])
def test_frame():
    # Clear memory at the start
    gc.collect()
    
    logger.debug("Received request to /test_frame")
    if 'frame' not in request.json:
        logger.error("No frame data provided")
        return jsonify({'error': 'No frame data provided'}), 400

    try:
        frame_data = request.json['frame']
        frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode frame")
            return jsonify({'error': 'Failed to decode frame'}), 400

        shape = list(frame.shape)
        frame = None  # Clear to free memory
        
        logger.debug(f"Frame decoded successfully, shape: {shape}")
        return jsonify({'status': 'Frame decoded', 'shape': shape})

    finally:
        # Force garbage collection
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_initialized': model_manager.is_initialized if model_manager else False,
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'service': 'LSTM Server'
    }), 200

if __name__ == '__main__':
    logger.info("Starting server...")
    # Don't initialize model at startup to save memory
    
    port = int(os.environ.get('PORT', 10000))
    
    try:
        from waitress import serve
        logger.info(f"Starting server with Waitress on port {port}")
        serve(app, host='0.0.0.0', port=port, threads=1)  # Single thread for stability
    except ImportError:
        logger.info(f"Waitress not available, using Flask development server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=False)
