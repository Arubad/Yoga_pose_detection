from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
import threading
import time

app = Flask(__name__)

# Global video frame and pose data
global_frame = None
pose_results = []
stop_thread = False

# Load model and labels
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/model.h5')
LABELS_PATH = os.environ.get('LABELS_PATH', 'model/labels.npy')

try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)
    print(f"Model loaded. Labels: {labels}")
except Exception as e:
    print(f"Model/label load error: {e}")
    model = None
    labels = []

# Load YOLO for person detection
yolo_model = YOLO("yolov8n.pt")

# BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose():
    global global_frame, stop_thread, pose_results

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    while not stop_thread:
        ret, frm = cap.read()
        if not ret:
            break

        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        results = yolo_model.predict(source=rgb, classes=[0], verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy() if results else []

        current_pose_results = []
        person_id = 1

        for det in detections:
            x1, y1, x2, y2 = map(int, det)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frm.shape[1]), min(y2, frm.shape[0])
            person_crop = rgb[y1:y2, x1:x2]

            if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                continue

            person_crop_resized = cv2.resize(person_crop, (256, 256))
            person_result = pose.process(person_crop_resized)

            label = "No pose"
            if person_result.pose_landmarks:
                keypoints = [val for lm in person_result.pose_landmarks.landmark
                             for val in (lm.x, lm.y, lm.z, lm.visibility)]
                keypoints = np.array(keypoints).reshape(1, -1)

                if keypoints.shape[1] == 132 and model is not None:
                    pred = model.predict(keypoints)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]
                    label = f"{labels[idx]} ({conf*100:.1f}%)" if conf > 0.5 else "Low Confidence"

                # Draw pose landmarks back on original frame
                scale_x = (x2 - x1) / 256
                scale_y = (y2 - y1) / 256

                for lm in person_result.pose_landmarks.landmark:
                    cx, cy = int(lm.x * 256 * scale_x + x1), int(lm.y * 256 * scale_y + y1)
                    cv2.circle(frm, (cx, cy), 3, (0, 255, 0), -1)

            cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frm, f"Person {person_id}: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            current_pose_results.append({
                "id": f"Person {person_id}",
                "pose": label
            })
            person_id += 1

        pose_results = current_pose_results

        _, buffer = cv2.imencode('.jpg', frm)
        global_frame = buffer.tobytes()
        time.sleep(0.01)

    cap.release()

def generate_frames():
    global global_frame
    while True:
        if global_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
        else:
            blank = np.ones((480, 640, 3), np.uint8) * 255
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_data')
def pose_data():
    return jsonify(pose_results)

@app.route('/start', methods=['POST'])
def start_detection():
    global stop_thread
    stop_thread = False
    threading.Thread(target=detect_pose).start()
    return "Detection started"

@app.route('/stop', methods=['POST'])
def stop_detection():
    global stop_thread
    stop_thread = True
    return "Detection stopped"

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
