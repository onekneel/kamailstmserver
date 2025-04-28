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
import gc

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorFlow to be memory-efficient
logger.info("Configuring TensorFlow...")
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if (physical_devices):
        logger.info(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Enabled memory growth for GPU: {device}")
            except Exception as dev_e:
                logger.warning(f"Could not enable memory growth for {device}: {dev_e}")
    else:
        logger.warning("No GPU devices found, using CPU only")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception as e:
    logger.error(f"Error configuring TensorFlow devices: {e}")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["https://kamaibisubilar.vercel.app", "http://localhost:3000", "http://192.168.1.3:3000"]}})

# Initialize MediaPipe Holistic
try:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe: {e}")
    mp_holistic = None
    mp_drawing = None

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    # Initialize zero arrays for all keypoints
    pose = np.zeros(33*4)
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    additional_features = np.zeros(18)

    # Extract pose landmarks if available
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()

    # Extract hand landmarks if available
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    # Compute additional features if pose and at least one hand are present
    if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
        head = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z])
        left_shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z])
        right_shoulder = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z])
        chest = (left_shoulder + right_shoulder) / 2
        chin = np.zeros(3)
        forehead = np.zeros(3)
        if results.face_landmarks:
            chin = np.array([results.face_landmarks.landmark[152].x, results.face_landmarks.landmark[152].y, results.face_landmarks.landmark[152].z])
            forehead = np.array([results.face_landmarks.landmark[10].x, results.face_landmarks.landmark[10].y, results.face_landmarks.landmark[10].z])
        left_elbow = np.array([results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z])
        right_elbow = np.array([results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z])
        
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
    lh_present = results.left_hand_landmarks is not None
    rh_present = results.right_hand_landmarks is not None
    return lh_present or rh_present

def normalize_keypoints(sequences, min_val, max_val):
    sequences = np.array(sequences)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1e-6
    return (sequences - min_val) / range_val

def safe_unlink(file_path, retries=5, delay=1.0):
    """Attempt to delete a file with retries to handle Windows file locking issues."""
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Successfully deleted {file_path}")
            return True
        except PermissionError as e:
            logger.warning(f"Attempt {attempt + 1} failed to delete {file_path}: {e}")
            time.sleep(delay)
    logger.error(f"Failed to delete {file_path} after {retries} attempts")
    return False

# Configuration
actions = ['Goodbye', 'Hello', 'Thankyou', 'Please', 'Thankyou(FSL)', 'Wait(FSL)']
sequence_length = 30
MIN_TEST_FRAMES = 10

# Load precomputed min/max values
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'min_val.npy')
max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'max_val.npy')
try:
    if os.path.exists(min_val_path) and os.path.exists(max_val_path):
        min_val = np.load(min_val_path)
        max_val = np.load(max_val_path)
        logger.info("Loaded min/max values successfully")
    else:
        raise FileNotFoundError(f"Min/max files not found at {min_val_path} and {max_val_path}")
except Exception as e:
    logger.warning(f"Min/Max files not found or error loading: {e}. Using default values (may reduce accuracy).")
    keypoint_dim = 33*4 + 21*3 + 21*3 + 18
    min_val = np.zeros(keypoint_dim)
    max_val = np.ones(keypoint_dim)
    min_val[:33*4:4] = 0
    max_val[:33*4:4] = 1
    min_val[1:33*4:4] = 0
    max_val[1:33*4:4] = 1
    min_val[2:33*4:4] = -1
    max_val[2:33*4:4] = 1
    min_val[3:33*4:4] = 0
    max_val[3:33*4:4] = 1
    min_val[33*4:33*4+21*3:3] = 0
    max_val[33*4:33*4+21*3:3] = 1
    min_val[33*4+1:33*4+21*3:3] = 0
    max_val[33*4+1:33*4+21*3:3] = 1
    min_val[33*4+2:33*4+21*3:3] = -1
    max_val[33*4+2:33*4+21*3:3] = 1
    min_val[33*4+21*3:33*4+21*3+21*3:3] = 0
    max_val[33*4+21*3:33*4+21*3+21*3:3] = 1
    min_val[33*4+21*3+1:33*4+21*3+21*3:3] = 0
    max_val[33*4+21*3+1:33*4+21*3+21*3:3] = 1
    min_val[33*4+21*3+2:33*4+21*3+21*3:3] = -1
    max_val[33*4+21*3+2:33*4+21*3+21*3:3] = 1
    min_val[33*4+21*3+21*3:] = -1
    max_val[33*4+21*3+21*3:] = 1

# Load model on demand
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(BASE_DIR, 'public', 'model', 'kamaibestlstm_Modified.keras')
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
    return model

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'details': str(e) if app.debug else 'An unexpected error occurred'
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Basic health check
        if mp_holistic is None:
            return jsonify({'status': 'error', 'message': 'MediaPipe not initialized'}), 503
        if get_model() is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received request to /predict")

    # Clear memory from previous requests
    gc.collect()
    
    # Load model on demand
    current_model = get_model()
    if current_model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500

    if mp_holistic is None:
        logger.error("MediaPipe not initialized")
        return jsonify({'error': 'MediaPipe not initialized'}), 500

    if 'video' not in request.files:
        logger.error("No video file provided")
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        logger.error("No video file selected")
        return jsonify({'error': 'No video file selected'}), 400

    # Save video to temporary file
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
        # Process video
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            safe_unlink(temp_file_path)
            logger.error("Failed to open video file")
            return jsonify({'error': 'Failed to open video file'}), 500

        # Get video properties to check size
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Video dimensions: {frame_width}x{frame_height}, total frames: {total_frames}")

        # Memory optimization: Process in chunks if the video is long
        MAX_CHUNK_SIZE = 100  # Max frames to process at once
        CHUNK_OVERLAP = 10    # Overlap between chunks to ensure continuity

        sequence = []
        hand_frame_count = 0
        frames_to_process = min(total_frames, MAX_CHUNK_SIZE)
        
        with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, 
                                  model_complexity=1) as holistic:  # model_complexity can be 0, 1, or 2
            frame_idx = 0
            while frame_idx < total_frames and len(sequence) < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process this frame
                image, results = mediapipe_detection(frame, holistic)
                
                # Free memory
                del frame
                
                if has_hand_keypoints(results):
                    try:
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        hand_frame_count += 1
                        
                        # Free memory
                        del keypoints
                    except AttributeError as e:
                        logger.warning(f"Skipping frame due to missing landmarks: {e}")
                
                # Free memory
                del results
                del image
                
                # Force garbage collection periodically
                frame_idx += 1
                if frame_idx % 10 == 0:
                    gc.collect()
                    
            # Check if we have enough frames with hand keypoints
            if hand_frame_count >= MIN_TEST_FRAMES:
                # Pad the sequence if needed (keeping exact same logic for accuracy)
                while len(sequence) < sequence_length:
                    sequence.append(sequence[-1] if sequence else np.zeros(33*4 + 21*3 + 21*3 + 18))
            else:
                cap.release()
                cv2.destroyAllWindows()
                safe_unlink(temp_file_path)
                logger.error(f"Too few frames with hand keypoints: {hand_frame_count}")
                return jsonify({'error': f'Too few frames with hand keypoints ({hand_frame_count})'}), 400
        
        cap.release()
        cv2.destroyAllWindows()
        logger.debug(f"Extracted {hand_frame_count} frames with hand keypoints")

        # Free more memory before normalization and prediction
        gc.collect()

        # Apply global normalization
        norm_sequence = normalize_keypoints(np.array([sequence]), min_val, max_val)[0]
        logger.debug("Applied global normalization")

        # Free memory
        del sequence
        gc.collect()

        # Predict
        predictions = []
        confidences = []
        
        # Process predictions in smaller batches to reduce memory usage
        batch_size = 5
        for i in range(0, len(norm_sequence) - sequence_length + 1, batch_size):
            batch_end = min(i + batch_size, len(norm_sequence) - sequence_length + 1)
            batch_windows = []
            
            for j in range(i, batch_end):
                window = norm_sequence[j:j + sequence_length]
                if len(window) == sequence_length:
                    batch_windows.append(window)
            
            if batch_windows:
                # Predict on batch
                batch_results = current_model.predict(np.array(batch_windows), verbose=0)
                
                # Process results
                for k, res in enumerate(batch_results):
                    predicted_action = actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)] * 100)
                    logger.debug(f"Window {i+k}: Predicted {predicted_action} with confidence {confidence}%")
                    predictions.append(predicted_action)
                    confidences.append(confidence)
                
                # Free memory
                del batch_windows
                del batch_results
                gc.collect()

        if not predictions:
            safe_unlink(temp_file_path)
            logger.error("No predictions made (insufficient frames for sliding window)")
            return jsonify({'error': 'No predictions made (insufficient frames for sliding window)'}), 400

        # Handle Wait(FSL) misclassification (keeping exact same logic for accuracy)
        final_prediction = max(set(predictions), key=predictions.count)
        final_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == final_prediction])
        
        # Confidence-based adjustment for Wait(FSL)
        if final_prediction == 'Goodbye':
            wait_fsl_conf = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == 'Wait(FSL)']) if 'Wait(FSL)' in predictions else 0
            if wait_fsl_conf > 0 and (final_confidence - wait_fsl_conf) < 15:  # Increased threshold
                final_prediction = 'Wait(FSL)'
                final_confidence = wait_fsl_conf
                logger.debug("Adjusted prediction to Wait(FSL) due to close confidence")

        # Format response for frontend
        response = {
            'action': final_prediction,
            'confidence': final_confidence
        }
        logger.info(f"Prediction: {final_prediction} with confidence {final_confidence}%")

        # Clean up
        del norm_sequence
        del predictions
        del confidences
        gc.collect()
        
        safe_unlink(temp_file_path)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if os.path.exists(temp_file_path):
            safe_unlink(temp_file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    # Add startup logging
    logger.info(f"Starting server on port {port}")
    logger.info(f"Debug mode: {app.debug}")
    logger.info(f"CUDA available: {tf.test.is_built_with_cuda()}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Debug=False for production