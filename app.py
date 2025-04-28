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

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000"]}})

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Load model
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

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received request to /predict")
    if model is None:
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

        sequence = []
        hand_frame_count = 0
        with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            while len(sequence) < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                if has_hand_keypoints(results):
                    try:
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        hand_frame_count += 1
                    except AttributeError as e:
                        logger.warning(f"Skipping frame due to missing landmarks: {e}")
                        continue
            if hand_frame_count >= MIN_TEST_FRAMES:
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

        # Apply global normalization
        norm_sequence = normalize_keypoints(np.array([sequence]), min_val, max_val)[0]
        logger.debug("Applied global normalization")

        # Predict
        predictions = []
        confidences = []
        for i in range(len(norm_sequence) - sequence_length + 1):
            window = norm_sequence[i:i + sequence_length]
            if len(window) == sequence_length:
                res = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
                predicted_action = actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)] * 100)
                logger.debug(f"Window {i}: Predicted {predicted_action} with confidence {confidence}%")
                predictions.append(predicted_action)
                confidences.append(confidence)

        if not predictions:
            safe_unlink(temp_file_path)
            logger.error("No predictions made (insufficient frames for sliding window)")
            return jsonify({'error': 'No predictions made (insufficient frames for sliding window)'}), 400

        # Handle Wait(FSL) misclassification
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
        safe_unlink(temp_file_path)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if os.path.exists(temp_file_path):
            safe_unlink(temp_file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT or default to 10000
    app.run(host='0.0.0.0', port=port, debug=False)  # Debug=False for production