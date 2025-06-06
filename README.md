# 🧘‍♂️ Yoga Pose Detection App

A real-time web-based application that detects and classifies yoga poses from a webcam feed using deep learning and computer vision.

## 🔍 Features

* Real-time yoga pose detection via webcam
* Person detection using YOLOv8
* Pose estimation using MediaPipe BlazePose
* Custom pose classification using a trained Keras model
* Web interface for live video feed and pose updates
* Multi-person support with bounding boxes and classification

## 🛠️ Tech Stack

| Layer               | Technology                            |
| ------------------- | ------------------------------------- |
| Frontend            | HTML + MJPEG Streaming (`index.html`) |
| Backend             | Flask (Python)                        |
| Pose Estimation     | MediaPipe BlazePose                   |
| Person Detection    | YOLOv8 (via `ultralytics` library)    |
| Pose Classification | Keras-trained `.h5` model             |
| Webcam Access       | OpenCV (`cv2.VideoCapture`)           |

## 🖼️ App Architecture

```
Webcam Feed --> YOLOv8 --> Person Cropping --> BlazePose -->
Landmark Extraction --> Classification Model -->
Pose Labeling --> Annotated Video Frame --> Flask Web Interface
```

## 📁 Project Structure

```
yoga-pose-detector/
├── model/
│   ├── model.h5         # Trained Keras model
│   └── labels.npy       # Corresponding pose labels
├── templates/
│   └── index.html       # Main UI
├── app.py               # Main Flask application
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yoga-pose-detector.git
cd yoga-pose-detector
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Create `requirements.txt` like so:

```txt
flask
opencv-python
mediapipe
tensorflow
ultralytics
```

> Note: YOLOv8 is used via the `ultralytics` Python package.

### 3. Add Your Model

Place your trained Keras model and label file inside the `model/` folder:

```
model/model.h5
model/labels.npy
```

### 4. Run the App

```bash
python app.py
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

## 🚀 Example Usage

* Visit `/` for the live webcam feed.
* Pose predictions are streamed over the frame.
* JSON endpoint `/pose_data` shows current detected poses.
* Use `/start` and `/stop` (POST) to control detection.

## 📦 Deployment

For production deployment, use Docker or run with a WSGI server like Gunicorn.

```bash
gunicorn app:app
```

## 📌 Notes

* Uses the lightweight YOLOv8n model.
* Requires a webcam.
* Performance may vary depending on hardware (GPU preferred).

## 🧠 Model Training

This app expects:

* Input: 132-dimensional keypoint vectors (33 landmarks × 4 values: x, y, z, visibility)
* Output: A pose label (e.g., "Tree Pose", "Warrior", etc.)

You can train this using custom datasets captured with BlazePose and labeled accordingly.


