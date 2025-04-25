import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import warnings
import logging
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the TensorFlow model
model_path = os.path.join(os.path.dirname(__file__), "public", "model", "signKamaiLSTM.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Configuration
actions = ['Hello', 'Thankyou', 'Help', 'Please']
sequence_length = 30
MIN_TEST_FRAMES = 5
FRAME_SKIP = 1
CONFIDENCE_THRESHOLD = 0.7
RESIZE_FACTOR = 1.0
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Your existing functions (mediapipe_detection, extract_keypoints, etc.)
def mediapipe_detection(image, model):
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=10)
    results = model.process(image_rgb)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    additional_features = np.zeros(6)
    if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
        head = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z])
        left_shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z])
        right_shoulder = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z])
        chest = (left_shoulder + right_shoulder) / 2
        if results.left_hand_landmarks:
            lh_center = np.array([results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z])
            additional_features[0:3] = lh_center - head
        if results.right_hand_landmarks:
            rh_center = np.array([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y, results.right_hand_landmarks.landmark[0].z])
            additional_features[3:6] = rh_center - chest
    return np.concatenate([pose, lh, rh, additional_features])

def has_hand_keypoints(results):
    lh_present = results.left_hand_landmarks is not None
    rh_present = results.right_hand_landmarks is not None
    return lh_present or rh_present

def normalize_keypoints(sequence):
    min_val = np.min(sequence, axis=0)
    max_val = np.max(sequence, axis=0)
    range_val = np.maximum(max_val - min_val, 1e-6)
    return (sequence - min_val) / range_val

def predict_sign_language(video_path, debug=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"action": "None", "confidence": 0.0, "error": "Could not open video", "debug_info": {}}
    sequence = []
    predictions = []
    hand_frame_count = 0
    frame_count = 0
    prediction_counts = {}
    debug_info = {"total_frames": 0, "frames_with_hands": 0, "video_path": video_path}
    holistic_config = {
        'static_image_mode': False,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'model_complexity': 1
    }
    with mp_holistic.Holistic(**holistic_config) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            debug_info["total_frames"] = frame_count
            if frame_count % FRAME_SKIP != 0:
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)))
            frame, results = mediapipe_detection(frame, holistic)
            if has_hand_keypoints(results):
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                hand_frame_count += 1
                if len(sequence) >= sequence_length:
                    window = sequence[-sequence_length:]
                    norm_window = normalize_keypoints(np.array(window))
                    res = model.predict(np.expand_dims(norm_window, axis=0), verbose=0)[0]
                    predicted_action = actions[np.argmax(res)]
                    predictions.append(predicted_action)
                    prediction_counts[predicted_action] = prediction_counts.get(predicted_action, 0) + 1
                    most_common = max(prediction_counts.items(), key=lambda x: x[1]) if prediction_counts else (None, 0)
                    if most_common[1] >= 5 and most_common[1] / sum(prediction_counts.values()) > CONFIDENCE_THRESHOLD:
                        break
    cap.release()
    if hand_frame_count > 0:
        if predictions:
            final_prediction = max(set(predictions), key=predictions.count)
            confidence = float(max([predictions.count(p) / len(predictions) for p in set(predictions)]) * 100)
            return {
                "action": final_prediction,
                "confidence": confidence,
                "frame_count": hand_frame_count,
                "prediction_distribution": dict((x, predictions.count(x)) for x in set(predictions)),
                "debug_info": debug_info
            }
        else:
            return {
                "action": "None",
                "confidence": 0.0,
                "error": "No predictions made (insufficient frames for sliding window)",
                "debug_info": debug_info
            }
    else:
        return {
            "action": "None",
            "confidence": 0.0,
            "error": f"Too few frames with hand keypoints ({hand_frame_count})",
            "debug_info": debug_info
        }

# Flask Routes
@app.route('/')
def index():
    return jsonify({"message": "Sign Language Prediction API", "status": "running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        try:
            result = predict_sign_language(video_path, debug=False)
            os.remove(video_path)  # Clean up uploaded file
            return jsonify(result)
        except Exception as e:
            os.remove(video_path)  # Clean up on error
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Allowed: mp4, avi, mov"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))