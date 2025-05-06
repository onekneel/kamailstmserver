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

# Configure environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
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

# Lazy loading of MediaPipe
def load_mediapipe():
    global mp_holistic, mp_drawing
    if mp_holistic is None:
        logger.info("Loading MediaPipe models...")
        start = time.time()
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        logger.info(f"MediaPipe loaded in {time.time() - start:.2f} seconds")
    return mp_holistic, mp_drawing

# MediaPipe detection - MATCHED TO COLAB
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extract keypoints - MATCHED TO COLAB
def extract_keypoints(results):
    # Standard keypoints
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Add relative positions (hand-to-head, hand-to-chest, hand-to-chin, hand-to-forehead, hand-to-arm) to match training
    additional_features = np.zeros(18)  # 3 for left hand to head, 3 for right hand to chest, 3 for hand to chin, 3 for hand to forehead, 3 for left hand to left elbow, 3 for right hand to right elbow
    if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
        # Head (landmark 0), chest (midpoint of shoulders: landmarks 11 and 12)
        head = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z])
        left_shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z])
        right_shoulder = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z])
        chest = (left_shoulder + right_shoulder) / 2

        # Chin (face landmark 152) and forehead (face landmark 10)
        chin = np.zeros(3)
        forehead = np.zeros(3)
        if results.face_landmarks:
            chin = np.array([results.face_landmarks.landmark[152].x, results.face_landmarks.landmark[152].y, results.face_landmarks.landmark[152].z])
            forehead = np.array([results.face_landmarks.landmark[10].x, results.face_landmarks.landmark[10].y, results.face_landmarks.landmark[10].z])

        # Elbows (pose landmarks 13 and 14 for left and right elbows)
        left_elbow = np.array([results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z])
        right_elbow = np.array([results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z])

        if results.left_hand_landmarks:
            lh_center = np.array([results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z])
            additional_features[0:3] = lh_center - head  # Left hand to head
            additional_features[6:9] = lh_center - chin  # Left hand to chin
            additional_features[9:12] = lh_center - forehead  # Left hand to forehead
            additional_features[12:15] = lh_center - left_elbow  # Left hand to left elbow
        if results.right_hand_landmarks:
            rh_center = np.array([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y, results.right_hand_landmarks.landmark[0].z])
            additional_features[3:6] = rh_center - chest  # Right hand to chest
            additional_features[15:18] = rh_center - right_elbow  # Right hand to right elbow

    return np.concatenate([pose, lh, rh, additional_features])

def has_hand_keypoints(results):
    lh_present = results.left_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten())
    rh_present = results.right_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten())
    return lh_present or rh_present

# Normalize keypoints - MATCHED TO COLAB
def normalize_keypoints(sequences, min_val=None, max_val=None):
    sequences = np.array(sequences)
    
    if min_val is None or max_val is None:
        min_val = np.min(sequences, axis=(0, 1)) if len(sequences.shape) > 2 else np.min(sequences, axis=0)
        max_val = np.max(sequences, axis=(0, 1)) if len(sequences.shape) > 2 else np.max(sequences, axis=0)
    
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1e-6
    return (sequences - min_val) / range_val, min_val, max_val

def safe_unlink(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except:
        pass

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

# Model manager with improved feature extraction
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
        
        # Clear memory before initialization
        gc.collect()
        tf.keras.backend.clear_session()
        
        start_time = time.time()
        logger.info("Initializing model manager...")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'public', 'model', 'kamaibestlstm_Modified.keras')
        min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'min_val.npy')
        max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'max_val.npy')
        
        # Load Keras model
        try:
            self.model = load_model(model_path, compile=False)
            logger.info(f"Keras model loaded successfully in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            # Create a simple fallback model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, Dense
            
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

# Prediction endpoint with sliding window approach
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
        
        # Process video with sliding window approach
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return jsonify({'error': 'Failed to open video file'}), 500

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate optimal frame skip based on video length
        if total_frames > 100:
            FRAME_SKIP = 3
        else:
            FRAME_SKIP = 2
            
        logger.info(f"Video has {total_frames} frames at {fps} fps, using skip={FRAME_SKIP}")
        
        # Collect all frames with hand keypoints
        all_keypoints = []
        frame_count = 0
        hand_frame_count = 0

        # Use holistic model with face landmarks enabled
        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,  # Use medium complexity for better accuracy
            refine_face_landmarks=True  # Enable face landmarks for additional features
        ) as holistic:
            while frame_count < 300:  # Limit to 300 frames max
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                try:
                    # Process frame with full feature extraction
                    image, results = mediapipe_detection(frame, holistic)
                    
                    if has_hand_keypoints(results):
                        keypoints = extract_keypoints(results)
                        all_keypoints.append(keypoints)
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

        # Normalize all keypoints
        all_keypoints = np.array(all_keypoints)
        norm_keypoints, _, _ = normalize_keypoints(all_keypoints)
        
        # Use sliding window approach for prediction
        predictions = []
        confidences = []
        
        # Ensure we have enough frames for at least one window
        if len(norm_keypoints) >= manager.sequence_length:
            # Create sliding windows
            for i in range(len(norm_keypoints) - manager.sequence_length + 1):
                window = norm_keypoints[i:i + manager.sequence_length]
                
                # Make prediction on this window
                input_data = np.expand_dims(window, axis=0).astype(np.float32)
                res = manager.model.predict(input_data, verbose=0)[0]
                
                predicted_action = manager.actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)] * 100)
                
                predictions.append(predicted_action)
                confidences.append(confidence)
                
                # Clear memory
                input_data = None
                gc.collect()
            
            # Get the most common prediction
            if predictions:
                # Count occurrences of each prediction
                prediction_counts = {}
                for pred in predictions:
                    if pred in prediction_counts:
                        prediction_counts[pred] += 1
                    else:
                        prediction_counts[pred] = 1
                
                # Find the most common prediction
                final_prediction = max(prediction_counts, key=prediction_counts.get)
                
                # Calculate average confidence for the final prediction
                final_confidence = 0
                count = 0
                for i, pred in enumerate(predictions):
                    if pred == final_prediction:
                        final_confidence += confidences[i]
                        count += 1
                
                if count > 0:
                    final_confidence /= count
                
                # Create detailed response
                response = {
                    'action': final_prediction,
                    'confidence': final_confidence,
                    'processing_time': time.time() - start_time,
                    'prediction_distribution': {action: predictions.count(action) for action in set(predictions)}
                }
                
                logger.info(f"Prediction: {final_prediction} with confidence {final_confidence:.2f}% in {time.time() - start_time:.2f} seconds")
                logger.info(f"Distribution: {response['prediction_distribution']}")
                
                # Clear memory before returning
                gc.collect()
                
                return jsonify(response)
            else:
                return jsonify({'error': 'No predictions could be made'}), 500
        else:
            return jsonify({'error': f'Not enough frames for prediction. Need {manager.sequence_length}, got {len(norm_keypoints)}'}), 400

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

        # Use holistic model with face landmarks enabled
        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,
            refine_face_landmarks=True
        ) as holistic:
            try:
                image, results = mediapipe_detection(frame, holistic)
                frame = None  # Clear to free memory
                
                if has_hand_keypoints(results):
                    keypoints = extract_keypoints(results)
                    sequence_buffer[buffer_index] = keypoints
                    buffer_index = (buffer_index + 1) % manager.sequence_length

                    if buffer_index >= 10:  # Need at least 10 frames
                        # Normalize the sequence
                        norm_sequence, _, _ = normalize_keypoints([sequence_buffer])
                        norm_sequence = norm_sequence[0]  # Get the first (and only) sequence
                        
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
                            'frames_needed': 10
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
