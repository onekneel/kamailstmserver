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
import io
from PIL import Image

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/predict_stream": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]},
    r"/test_frame": {"origins": ["http://localhost:3000", "http://192.168.1.3:3000", "https://kamaibisubilar.vercel.app"]}
})

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
    image = cv2.resize(image, (640, 480))  # Downsample to 640x480
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    additional_features = np.zeros(18)
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
    
    keypoints = np.concatenate([pose, lh, rh, additional_features])
    # Pad to 276 features to match model expectation (temporary fix)
    if keypoints.shape[0] < 276:
        keypoints = np.pad(keypoints, (0, 276 - keypoints.shape[0]), mode='constant', constant_values=0)
    logger.debug(f"Extracted keypoints shape: {keypoints.shape}")
    return keypoints

def has_hand_keypoints(results):
    lh_present = results.left_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten())
    rh_present = results.right_hand_landmarks is not None and np.any(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten())
    logger.debug(f"Hand detection: Left hand present={lh_present}, Right hand present={rh_present}")
    return lh_present or rh_present

def normalize_keypoints(sequences, min_val, max_val):
    sequences = np.array(sequences)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1e-6
    return (sequences - min_val) / range_val

def safe_unlink(file_path, retries=5, delay=1.0):
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
sequence_length = 30  # Restored to 30 to match model
MIN_TEST_FRAMES = 10
FRAME_SKIP = 2

# Load precomputed min/max values
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
min_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'latest', 'min_val.npy')
max_val_path = os.path.join(BASE_DIR, 'public', 'keypoints', 'latest', 'max_val.npy')
try:
    if os.path.exists(min_val_path) and os.path.exists(max_val_path):
        min_val = np.load(min_val_path)
        max_val = np.load(max_val_path)
        logger.info("Loaded min/max values successfully")
        expected_keypoint_dim = 276  # Updated to match model
        if min_val.shape[0] != expected_keypoint_dim:
            logger.warning(f"Min/max shape mismatch: {min_val.shape[0]} (expected {expected_keypoint_dim}). Padding...")
            min_val = np.pad(min_val, (0, expected_keypoint_dim - min_val.shape[0]), mode='constant', constant_values=0)
            max_val = np.pad(max_val, (0, expected_keypoint_dim - min_val.shape[0]), mode='constant', constant_values=1)
    else:
        raise FileNotFoundError(f"Min/max files not found at {min_val_path} and {max_val_path}")
except Exception as e:
    logger.warning(f"Min/Max files not found or error loading: {e}. Using default values.")
    keypoint_dim = 276
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

# Load TFLite model
model_path = os.path.join(BASE_DIR, 'public', 'model', 'klstm.tflite')
try:
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        expected_input_shape = (1, sequence_length, 276)
        if tuple(input_details[0]['shape']) != expected_input_shape:
            logger.error(f"Model input shape mismatch: Expected {expected_input_shape}, got {input_details[0]['shape']}")
            raise ValueError(f"Model input shape mismatch: Expected {expected_input_shape}, got {input_details[0]['shape']}")
        logger.info("TFLite model loaded successfully")
    else:
        raise FileNotFoundError(f"TFLite model file not found at {model_path}")
except Exception as e:
    logger.error(f"Error loading TFLite model: {e}")
    interpreter = None
    input_details = None
    output_details = None

# In-memory sequence buffer for real-time prediction
sequence_buffer = []

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received request to /predict")
    if interpreter is None:
        logger.error("TFLite model not loaded")
        return jsonify({'error': 'TFLite model not loaded'}), 500

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
        with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
            while len(sequence) < sequence_length:
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
                while len(sequence) < sequence_length:
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

        norm_sequence = normalize_keypoints([sequence], min_val, max_val)[0]
        logger.debug(f"Normalized sequence shape: {norm_sequence.shape}")

        predictions = []
        confidences = []
        for i in range(len(norm_sequence) - sequence_length + 1):
            window = norm_sequence[i:i + sequence_length]
            if len(window) == sequence_length:
                input_data = np.expand_dims(window, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                res = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted_action = actions[np.argmax(res)]
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
    if interpreter is None:
        logger.error("TFLite model not loaded")
        return jsonify({'error': 'TFLite model not loaded'}), 500

    if mp_holistic is None:
        logger.error("MediaPipe not initialized")
        return jsonify({'error': 'MediaPipe not initialized'}), 500

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
        with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
            start_time = time.time()
            image, results = mediapipe_detection(frame, holistic)
            process_time = time.time() - start_time
            logger.debug(f"Frame processing took {process_time:.3f} seconds")

            if has_hand_keypoints(results):
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)

                # Maintain sequence length
                if len(sequence_buffer) > sequence_length:
                    sequence_buffer.pop(0)

                # Predict if we have enough frames
                if len(sequence_buffer) >= 15:  # Allow prediction with 15+ frames
                    norm_sequence = normalize_keypoints([sequence_buffer], min_val, max_val)[0]
                    # Pad to 30 frames if needed
                    while len(norm_sequence) < sequence_length:
                        norm_sequence = np.vstack([norm_sequence, norm_sequence[-1]])
                    norm_sequence = norm_sequence[:sequence_length]
                    input_data = np.expand_dims(norm_sequence, axis=0).astype(np.float32)
                    logger.debug(f"Input data shape: {input_data.shape}")
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    res = interpreter.get_tensor(output_details[0]['index'])[0]
                    predicted_action = actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)] * 100)

                    logger.debug(f"Predicted {predicted_action} with confidence {confidence:.2f}%")
                    return jsonify({
                        'action': predicted_action,
                        'confidence': confidence
                    })
                else:
                    logger.debug(f"Sequence length: {len(sequence_buffer)}/{sequence_length}")
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)