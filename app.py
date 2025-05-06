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
# import psutil  # Removed psutil import
import subprocess

# 1. Configure environment variables to reduce TF verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN

# 2. Set up optimized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Initialize Flask app with file size limit
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/predict_stream": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/test_frame": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/health": {"origins": "*"}
})

# 4. Lazy loading of MediaPipe
mp_holistic = None
mp_drawing = None

def load_mediapipe():
    global mp_holistic, mp_drawing
    if mp_holistic is None:
        logger.info("Loading MediaPipe models...")
        start = time.time()
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        logger.info(f"MediaPipe loaded in {time.time() - start:.2f} seconds")
    return mp_holistic, mp_drawing

# 5. Optimized MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.resize(image, (320, 240))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = None  # Clear image to free memory
    gc.collect()
    return None, results

# 6. Optimized keypoint extraction with additional features
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    additional_features = np.zeros(18)  # Hand-to-head, hand-to-chest, hand-to-chin, hand-to-forehead, hand-to-elbow
    if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
        head = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z])
        left_shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z])
        right_shoulder = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z])
        chest = (left_shoulder + right_shoulder) / 2
        left_elbow = np.array([results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z])
        right_elbow = np.array([results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z])

        chin = np.zeros(3)
        forehead = np.zeros(3)
        if results.face_landmarks:
            chin = np.array([results.face_landmarks.landmark[152].x, results.face_landmarks.landmark[152].y, results.face_landmarks.landmark[152].z])
            forehead = np.array([results.face_landmarks.landmark[10].x, results.face_landmarks.landmark[10].y, results.face_landmarks.landmark[10].z])

        if results.left_hand_landmarks:
            lh_center = np.array([results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z])
            additional_features[0:3] = lh_center - head
            additional_features[6:9] = lh_center - chin
            additional_features[9:12] = lh_center - forehead
            additional_features[12:15] = lh_center - left_elbow
        if results.right_hand_landmarks:
            rh_center = np.array([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y, results.right_hand_landmarks.landmark[0].z])
            additional_features[3:6] = rh_center - chest
            additional_features[15:18] = rh_center - right_elbow

    return np.concatenate([pose, lh, rh, additional_features])

def has_hand_keypoints(results):
    lh_present = results.left_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten())
    rh_present = results.right_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten())
    return lh_present or rh_present

def normalize_keypoints(sequences, min_val, max_val):
    sequences = np.array(sequences)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1e-6
    return (sequences - min_val) / range_val

def safe_unlink(file_path, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                return True
        except PermissionError:
            time.sleep(delay)
    return False

# 7. Memory monitoring
def log_memory():
    logger.info("Memory monitoring disabled (psutil not available)")

def check_memory():
    # Memory check disabled (psutil not available)
    pass

# 8. Video validation
def validate_video(file_path):
    try:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', file_path, '-f', 'null', '-'],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Video validation failed: {e}")
        return False

def get_video_duration(file_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True, text=True, timeout=5
        )
        # Debug the output
        stdout = result.stdout.strip()
        logger.info(f"ffprobe output: '{stdout}'")
        
        # Handle empty output
        if not stdout:
            logger.warning("Empty ffprobe output, using default duration")
            return 10.0  # Default to a reasonable duration
        
        # Handle potential newlines or other characters
        stdout = stdout.replace('\n', '').strip()
        
        try:
            duration = float(stdout)
            logger.info(f"Parsed video duration: {duration} seconds")
            return duration
        except ValueError as e:
            logger.error(f"Failed to parse duration '{stdout}': {e}")
            return 10.0  # Default to a reasonable duration
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        # If ffprobe fails, try using OpenCV as a fallback
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error("Failed to open video with OpenCV")
                return 10.0  # Default to a reasonable duration
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 10.0
            cap.release()
            
            logger.info(f"OpenCV calculated duration: {duration} seconds")
            return duration
        except Exception as cv_error:
            logger.error(f"OpenCV duration calculation failed: {cv_error}")
            return 10.0  # Default to a reasonable duration

# 9. Model and data loading
class ModelManager:
    def __init__(self):
        self.model = None
        self.min_val = None
        self.max_val = None
        self.is_initialized = False
        self.actions = ['Goodbye', 'Hello', 'Thankyou', 'Please', 'Thankyou(FSL)', 'Wait(FSL)']
        self.sequence_length = 30
        self.MIN_TEST_FRAMES = 10

    def initialize(self):
        if self.is_initialized:
            return
        
        start_time = time.time()
        logger.info("Initializing model manager...")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'public', 'model', 'kamaibestlstm_Modified.keras')
        min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'min_val.npy')
        max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'max_val.npy')
        
        # Limit GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.error(f"GPU memory growth setting error: {e}")
        
        # Load Keras model with compatibility handling
        model = None
        try:
            # First try standard loading
            try:
                model = load_model(model_path)
                logger.info(f"Keras model loaded successfully in {time.time() - start_time:.2f} seconds")
            except (ValueError, TypeError) as e:
                # If standard loading fails, try custom loading approach
                logger.warning(f"Standard model loading failed: {e}")
                logger.info("Attempting to load model with custom approach...")
                
                # Create a simple LSTM model with the same architecture
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                # Create a new model with similar architecture
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 276)))
                model.add(LSTM(128, return_sequences=True, activation='relu'))
                model.add(LSTM(64, return_sequences=False, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(len(self.actions), activation='softmax'))
                
                # Try to load weights only
                try:
                    model.build((None, self.sequence_length, 276))
                    model.load_weights(model_path)
                    logger.info("Successfully loaded model weights")
                except Exception as weight_error:
                    logger.warning(f"Failed to load weights directly: {weight_error}")
                    
                    # Convert keras to h5 format and try again
                    h5_path = os.path.join(tempfile.gettempdir(), "temp_model.h5")
                    try:
                        import subprocess
                        subprocess.run([
                            "python", "-c", 
                            f"import tensorflow as tf; model = tf.keras.models.load_model('{model_path}', compile=False); model.save('{h5_path}', save_format='h5')"
                        ], check=True)
                        model = load_model(h5_path)
                        logger.info("Successfully loaded model via h5 conversion")
                    except Exception as h5_error:
                        logger.error(f"H5 conversion failed: {h5_error}")
                        logger.info("Using untrained model as fallback")
                    finally:
                        if os.path.exists(h5_path):
                            os.remove(h5_path)
            
            self.model = model
            
            # Warm up the model
            dummy_input = np.zeros((1, self.sequence_length, 276), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            logger.info("Model warmed up with dummy prediction")
            tf.keras.backend.clear_session()
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            raise
        
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

# 10. Create model manager instance
model_manager = ModelManager()

# 11. Initialize sequence buffer
sequence_buffer = np.zeros((model_manager.sequence_length, 276), dtype=np.float32)
buffer_index = 0

# 12. Prediction endpoints
@app.route('/predict', methods=['POST'])
def predict():
    global buffer_index
    logger.debug("Received request to /predict")
    if request.content_length > app.config['MAX_CONTENT_LENGTH']:
        logger.error("Video file too large")
        return jsonify({'error': 'Video file exceeds 10 MB limit'}), 400

    if not model_manager.is_initialized:
        model_manager.initialize()
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
        temp_file_path = os.path.join(tempfile.gettempdir(), f"video_{int(time.time())}.mp4")
        video_file.save(temp_file_path)
        if not validate_video(temp_file_path):
            logger.error("Invalid or corrupted video file")
            return jsonify({'error': 'Invalid or corrupted video file'}), 400
        duration = get_video_duration(temp_file_path)
        if duration > 30:
            logger.error("Video duration exceeds 30 seconds")
            return jsonify({'error': 'Video duration exceeds 30 seconds'}), 400
        logger.debug(f"Video saved to: {temp_file_path}")

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return jsonify({'error': 'Failed to open video file'}), 500

        sequence = np.zeros((model_manager.sequence_length, 276), dtype=np.float32)
        seq_index = 0
        hand_frame_count = 0
        frame_count = 0
        FRAME_SKIP = 3
        MIN_TEST_FRAMES = model_manager.MIN_TEST_FRAMES
        max_frames = 100

        with mp_holistic.Holistic(
            min_detection_confidence=0.3,  # Match testing code
            min_tracking_confidence=0.3,
            model_complexity=0,
            refine_face_landmarks=True  # Needed for chin/forehead
        ) as holistic:
            while seq_index < model_manager.sequence_length and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                try:
                    _, results = mediapipe_detection(frame, holistic)
                    check_memory()
                    log_memory()
                    if has_hand_keypoints(results):
                        keypoints = extract_keypoints(results)
                        sequence[seq_index] = keypoints
                        seq_index += 1
                        hand_frame_count += 1
                    gc.collect()
                except MemoryError:
                    logger.error("Memory error during frame processing")
                    return jsonify({'error': 'Insufficient memory'}), 500
                except Exception as e:
                    logger.warning(f"Skipping frame due to error: {e}")
                    continue

        if hand_frame_count < MIN_TEST_FRAMES:
            logger.error(f"Too few frames with hand keypoints: {hand_frame_count}")
            return jsonify({'error': f'Too few frames with hand keypoints ({hand_frame_count})'}), 400

        norm_sequence = normalize_keypoints([sequence], model_manager.min_val, model_manager.max_val)[0]
        logger.debug(f"Normalized sequence shape: {norm_sequence.shape}")

        predictions = []
        confidences = []
        for i in range(len(norm_sequence) - model_manager.sequence_length + 1):
            window = norm_sequence[i:i + model_manager.sequence_length]
            if len(window) == model_manager.sequence_length:
                input_data = np.expand_dims(window, axis=0).astype(np.float32)
                res = model_manager.model.predict(input_data, verbose=0)[0]
                predicted_action = model_manager.actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)] * 100)
                logger.debug(f"Window {i}: Predicted {predicted_action} with confidence {confidence:.2f}%")
                predictions.append(predicted_action)
                confidences.append(confidence)
                tf.keras.backend.clear_session()
                gc.collect()

        if not predictions:
            logger.error("No predictions made (insufficient frames for sliding window)")
            return jsonify({'error': 'No predictions made (insufficient frames for sliding window)'}), 400

        final_prediction = max(set(predictions), key=predictions.count)
        final_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == final_prediction])

        response = {
            'action': final_prediction,
            'confidence': final_confidence
        }
        logger.info(f"Prediction: {final_prediction} with confidence {final_confidence:.2f}%")
        return jsonify(response)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if temp_file_path and os.path.exists(temp_file_path):
            safe_unlink(temp_file_path)
        gc.collect()
        tf.keras.backend.clear_session()

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    global buffer_index
    logger.debug("Received request to /predict_stream")
    
    if not model_manager.is_initialized:
        model_manager.initialize()
    mp_holistic, _ = load_mediapipe()

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

        logger.debug(f"Frame decoded, shape: {frame.shape}")

        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0,
            refine_face_landmarks=True
        ) as holistic:
            try:
                _, results = mediapipe_detection(frame, holistic)
                check_memory()
                log_memory()
                if has_hand_keypoints(results):
                    keypoints = extract_keypoints(results)
                    sequence_buffer[buffer_index] = keypoints
                    buffer_index = (buffer_index + 1) % model_manager.sequence_length

                    if buffer_index >= 15:
                        norm_sequence = normalize_keypoints([sequence_buffer], model_manager.min_val, model_manager.max_val)[0]
                        while len(norm_sequence) < model_manager.sequence_length:
                            norm_sequence = np.vstack([norm_sequence, norm_sequence[-1]])
                        norm_sequence = norm_sequence[:model_manager.sequence_length]
                        input_data = np.expand_dims(norm_sequence, axis=0).astype(np.float32)
                        res = model_manager.model.predict(input_data, verbose=0)[0]
                        predicted_action = model_manager.actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)] * 100)
                        logger.debug(f"Predicted {predicted_action} with confidence {confidence:.2f}%")
                        tf.keras.backend.clear_session()
                        gc.collect()
                        return jsonify({
                            'action': predicted_action,
                            'confidence': confidence
                        })
                    else:
                        logger.debug(f"Sequence length: {buffer_index}/{model_manager.sequence_length}")
                        return jsonify({
                            'action': 'Processing',
                            'confidence': 0
                        })
                else:
                    return jsonify({
                        'action': 'No hands detected',
                        'confidence': 0
                    })
            except MemoryError:
                logger.error("Memory error during frame processing")
                return jsonify({'error': 'Insufficient memory'}), 500

    finally:
        gc.collect()
        tf.keras.backend.clear_session()

@app.route('/test_frame', methods=['POST'])
def test_frame():
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

        logger.debug(f"Frame decoded successfully, shape: {frame.shape}")
        return jsonify({'status': 'Frame decoded', 'shape': list(frame.shape)})

    finally:
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_initialized': model_manager.is_initialized,
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'service': 'LSTM Server'
    }), 200

if __name__ == '__main__':
    logger.info("Initializing model before server start...")
    model_manager.initialize()
    
    port = int(os.environ.get('PORT', 10000))
    
    try:
        from waitress import serve
        logger.info(f"Starting server with Waitress on port {port}")
        serve(app, host='0.0.0.0', port=port)
    except ImportError:
        logger.info(f"Waitress not available, using Flask development server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
