from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import tempfile
import logging
import time
import base64
from PIL import Image

# 1. Configure environment variables to reduce TF verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN custom operations

# 2. Set up optimized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/predict_stream": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/test_frame": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/health": {"origins": "*"}  # Add this line
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

# 5. Optimized mediapipe detection
def mediapipe_detection(image, model):
    # Resize to smaller dimensions for faster processing
    image = cv2.resize(image, (320, 240))  # Reduced from 640x480 for faster processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# 6. Optimized keypoint extraction
def extract_keypoints(results):
    # Use numpy vectorized operations for better performance
    pose = np.zeros(33*4)
    if results.pose_landmarks:
        pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        pose = pose_landmarks.flatten()
    
    lh = np.zeros(21*3)
    if results.left_hand_landmarks:
        lh_landmarks = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        lh = lh_landmarks.flatten()
    
    rh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh_landmarks = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        rh = rh_landmarks.flatten()
    
    # Simplified additional features calculation
    additional_features = np.zeros(18)
    
    # Concatenate and pad if needed
    keypoints = np.concatenate([pose, lh, rh, additional_features])
    if keypoints.shape[0] < 276:
        keypoints = np.pad(keypoints, (0, 276 - keypoints.shape[0]), mode='constant', constant_values=0)
    
    return keypoints

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

# 7. Model and data loading functions
class ModelManager:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.min_val = None
        self.max_val = None
        self.is_initialized = False
        self.actions = ['Goodbye', 'Hello', 'Thankyou', 'Please', 'Thankyou(FSL)', 'Wait(FSL)']
        self.sequence_length = 30
    
    def initialize(self):
        if self.is_initialized:
            return
        
        start_time = time.time()
        logger.info("Initializing model manager...")
        
        # Load model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'public', 'model', 'klstm.tflite')
        min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints','min_val.npy')
        max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'max_val.npy')
        
        # Control GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.error(f"GPU memory growth setting error: {e}")
        
        # Load TFLite model
        try:
            # Create interpreter with optimizations
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=4  # Adjust based on your CPU cores
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
            # Warm up the model with a dummy prediction
            dummy_input = np.zeros((1, self.sequence_length, 276), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            _ = self.interpreter.get_tensor(self.output_details[0]['index'])
            logger.info("Model warmed up with dummy prediction")
        except Exception as e:
            logger.error(f"Error loading TFLite model: {e}")
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
                raise FileNotFoundError(f"Min/max files not found at {min_val_path} and {max_val_path}")
        except Exception as e:
            logger.warning(f"Error loading min/max values: {e}. Using defaults.")
            keypoint_dim = 276
            self.min_val = np.zeros(keypoint_dim)
            self.max_val = np.ones(keypoint_dim)
        
        self.is_initialized = True
        logger.info(f"Model manager initialized in {time.time() - start_time:.2f} seconds")

# 8. Create model manager instance
model_manager = ModelManager()

# 9. Initialize sequence buffer for streaming
sequence_buffer = []

# 10. Prediction endpoints
@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received request to /predict")
    
    # Ensure model is initialized
    if not model_manager.is_initialized:
        model_manager.initialize()
    
    # Ensure MediaPipe is loaded
    mp_holistic, _ = load_mediapipe()
    
    if 'video' not in request.files:
        logger.error("No video file provided")
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        logger.error("No video file selected")
        return jsonify({'error': 'No video file selected'}), 400

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"video_{int(time.time())}.mp4")
    try:
        video_file.save(temp_file_path)
        logger.debug(f"Video saved to temporary file: {temp_file_path}")
    except Exception as e:
        logger.error(f"Failed to save video file: {e}")
        safe_unlink(temp_file_path)
        return jsonify({'error': f'Failed to save video file: {e}'}), 500

    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            safe_unlink(temp_file_path)
            logger.error("Failed to open video file")
            return jsonify({'error': 'Failed to open video file'}), 500

        sequence = []
        hand_frame_count = 0
        frame_count = 0
        FRAME_SKIP = 3  # Increased from 2 to 3 for faster processing
        MIN_TEST_FRAMES = 10
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.2, 
            min_tracking_confidence=0.2,
            model_complexity=0  # Use simpler model for faster processing
        ) as holistic:
            while len(sequence) < model_manager.sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue
                
                start_time = time.time()
                image, results = mediapipe_detection(frame, holistic)
                process_time = time.time() - start_time
                logger.debug(f"Frame {frame_count}: Mediapipe processing took {process_time:.3f} seconds")
                
                if has_hand_keypoints(results):
                    try:
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        hand_frame_count += 1
                    except AttributeError as e:
                        logger.warning(f"Skipping frame due to missing landmarks: {e}")
                        continue
            
            if hand_frame_count >= MIN_TEST_FRAMES:
                while len(sequence) < model_manager.sequence_length:
                    sequence.append(sequence[-1] if sequence else np.zeros(276))
            else:
                cap.release()
                cv2.destroyAllWindows()
                safe_unlink(temp_file_path)
                logger.error(f"Too few frames with hand keypoints: {hand_frame_count}")
                return jsonify({'error': f'Too few frames with hand keypoints ({hand_frame_count})'}), 400
        
        cap.release()
        cv2.destroyAllWindows()
        logger.debug(f"Extracted {hand_frame_count} frames with hand keypoints")

        norm_sequence = normalize_keypoints([sequence], model_manager.min_val, model_manager.max_val)[0]
        logger.debug(f"Normalized sequence shape: {norm_sequence.shape}")

        predictions = []
        confidences = []
        for i in range(len(norm_sequence) - model_manager.sequence_length + 1):
            window = norm_sequence[i:i + model_manager.sequence_length]
            if len(window) == model_manager.sequence_length:
                input_data = np.expand_dims(window, axis=0).astype(np.float32)
                model_manager.interpreter.set_tensor(model_manager.input_details[0]['index'], input_data)
                model_manager.interpreter.invoke()
                res = model_manager.interpreter.get_tensor(model_manager.output_details[0]['index'])[0]
                predicted_action = model_manager.actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)] * 100)
                logger.debug(f"Window {i}: Predicted {predicted_action} with confidence {confidence:.2f}%")
                predictions.append(predicted_action)
                confidences.append(confidence)

        if not predictions:
            safe_unlink(temp_file_path)
            logger.error("No predictions made (insufficient frames for sliding window)")
            return jsonify({'error': 'No predictions made (insufficient frames for sliding window)'}), 400

        final_prediction = max(set(predictions), key=predictions.count)
        final_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == final_prediction])

        response = {
            'action': final_prediction,
            'confidence': final_confidence
        }
        logger.info(f"Prediction: {final_prediction} with confidence {final_confidence:.2f}%")

        safe_unlink(temp_file_path)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if os.path.exists(temp_file_path):
            safe_unlink(temp_file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    logger.debug("Received request to /predict_stream")
    
    # Ensure model is initialized
    if not model_manager.is_initialized:
        model_manager.initialize()
    
    # Ensure MediaPipe is loaded
    mp_holistic, _ = load_mediapipe()
    
    if 'frame' not in request.json:
        logger.error("No frame data provided")
        return jsonify({'error': 'No frame data provided'}), 400

    try:
        # Decode base64 frame
        frame_data = request.json['frame']
        frame_data = frame_data.split(',')[1]  # Remove data:image/jpeg;base64,
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode frame")
            return jsonify({'error': 'Failed to decode frame'}), 400

        logger.debug(f"Frame decoded, shape: {frame.shape}")

        # Process frame
        with mp_holistic.Holistic(
            min_detection_confidence=0.2, 
            min_tracking_confidence=0.2,
            model_complexity=0  # Use simpler model for faster processing
        ) as holistic:
            start_time = time.time()
            image, results = mediapipe_detection(frame, holistic)
            process_time = time.time() - start_time
            logger.debug(f"Frame processing took {process_time:.3f} seconds")

            if has_hand_keypoints(results):
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)

                # Maintain sequence length
                if len(sequence_buffer) > model_manager.sequence_length:
                    sequence_buffer.pop(0)

                # Predict if we have enough frames
                if len(sequence_buffer) >= 15:  # Allow prediction with 15+ frames
                    norm_sequence = normalize_keypoints([sequence_buffer], model_manager.min_val, model_manager.max_val)[0]
                    # Pad to 30 frames if needed
                    while len(norm_sequence) < model_manager.sequence_length:
                        norm_sequence = np.vstack([norm_sequence, norm_sequence[-1]])
                    norm_sequence = norm_sequence[:model_manager.sequence_length]
                    input_data = np.expand_dims(norm_sequence, axis=0).astype(np.float32)
                    logger.debug(f"Input data shape: {input_data.shape}")
                    model_manager.interpreter.set_tensor(model_manager.input_details[0]['index'], input_data)
                    model_manager.interpreter.invoke()
                    res = model_manager.interpreter.get_tensor(model_manager.output_details[0]['index'])[0]
                    predicted_action = model_manager.actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)] * 100)

                    logger.debug(f"Predicted {predicted_action} with confidence {confidence:.2f}%")
                    return jsonify({
                        'action': predicted_action,
                        'confidence': confidence
                    })
                else:
                    logger.debug(f"Sequence length: {len(sequence_buffer)}/{model_manager.sequence_length}")
                    return jsonify({
                        'action': 'Processing',
                        'confidence': 0
                    })
            else:
                return jsonify({
                    'action': 'No hands detected',
                    'confidence': 0
                })

    except Exception as e:
        logger.error(f"Error processing stream frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_frame', methods=['POST'])
def test_frame():
    logger.debug("Received request to /test_frame")
    if 'frame' not in request.json:
        logger.error("No frame data provided")
        return jsonify({'error': 'No frame data provided'}), 400

    try:
        # Decode base64 frame
        frame_data = request.json['frame']
        frame_data = frame_data.split(',')[1]  # Remove data:image/jpeg;base64,
        frame_bytes = base64.b64decode(frame_data)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode frame")
            return jsonify({'error': 'Failed to decode frame'}), 400

        logger.debug(f"Frame decoded successfully, shape: {frame.shape}")
        return jsonify({'status': 'Frame decoded', 'shape': list(frame.shape)})

    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return jsonify({'error': str(e)}), 400

# Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_initialized': model_manager.is_initialized,
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'service': 'LSTM Server'
    }), 200

if __name__ == '__main__':
    # Initialize model before starting server
    logger.info("Initializing model before server start...")
    model_manager.initialize()
    
    port = int(os.environ.get('PORT', 10000))
    
    # Use production-ready WSGI server if available
    try:
        from waitress import serve
        logger.info(f"Starting server with Waitress on port {port}")
        serve(app, host='0.0.0.0', port=port)
    except ImportError:
        logger.info(f"Waitress not available, using Flask development server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)