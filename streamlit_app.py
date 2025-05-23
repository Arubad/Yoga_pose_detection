import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Yoga Pose Detection",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.block-container {
    background: white;
    border-radius: 10px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    drawing = mp.solutions.drawing_utils
    return mp_pose, pose, drawing

# Load model with caching
@st.cache_resource
def load_yoga_model():
    try:
        # For Streamlit Cloud, you'll need to include these files in your repo
        model_path = "model/model.h5"
        labels_path = "model/labels.npy"
        
        if os.path.exists(model_path) and os.path.exists(labels_path):
            model = load_model(model_path)
            labels = np.load(labels_path)
            return model, labels
        else:
            st.error("Model files not found. Please upload model.h5 and labels.npy to the model/ directory.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_frame(frame, pose, drawing, mp_pose, model, labels):
    """Process a single frame for pose detection"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(rgb_frame)
    
    # Draw landmarks on the frame
    if results.pose_landmarks:
        drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract keypoints for prediction
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        keypoints = np.array(keypoints).reshape(1, -1)
        
        # Make prediction if model is loaded
        if model is not None and keypoints.shape[1] == 132:
            pred = model.predict(keypoints, verbose=0)[0]
            idx = np.argmax(pred)
            conf = pred[idx]
            
            if conf > 0.7:
                pose_name = labels[idx]
                confidence = conf * 100
                
                # Add text to frame
                cv2.putText(frame, f"{pose_name}: {confidence:.1f}%", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return frame, pose_name, confidence
            else:
                cv2.putText(frame, "Low confidence", (10, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame, "Low confidence", conf * 100
    else:
        cv2.putText(frame, "No pose detected", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
        return frame, "No pose detected", 0
    
    return frame, None, 0

def main():
    # Header
    st.markdown('<h1 class="main-header">🧘‍♀️ Yoga Pose Detection</h1>', unsafe_allow_html=True)
    
    # Load components
    mp_pose, pose, drawing = load_mediapipe()
    model, labels = load_yoga_model()
    
    # Sidebar
    st.sidebar.title("📋 Control Panel")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Input Mode",
        ["📷 Upload Image", "🎥 Upload Video", "📱 Use Webcam (Local Only)"]
    )
    
    # Model status
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"🏷️ Available poses: {len(labels)}")
    else:
        st.sidebar.error("❌ Model not loaded")
        st.sidebar.info("Upload model files to enable pose detection")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "📷 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process image
                if st.button("🔍 Analyze Pose"):
                    with st.spinner("Analyzing pose..."):
                        # Convert PIL to OpenCV format
                        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Process the image
                        processed_frame, pose_name, confidence = process_frame(
                            opencv_image, pose, drawing, mp_pose, model, labels
                        )
                        
                        # Display results
                        st.subheader("Analysis Results")
                        processed_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st.image(processed_image, caption="Pose Detection Results", use_column_width=True)
                        
                        # Show detection results in sidebar
                        with col2:
                            if pose_name and confidence > confidence_threshold * 100:
                                st.success(f"🎯 **Detected Pose:** {pose_name}")
                                st.metric("Confidence", f"{confidence:.1f}%")
                            else:
                                st.warning("⚠️ No confident pose detected")
        
        elif mode == "🎥 Upload Video":
            uploaded_video = st.file_uploader(
                "Choose a video...", 
                type=['mp4', 'avi', 'mov']
            )
            
            if uploaded_video is not None:
                # Save uploaded video to temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                # Display original video
                st.subheader("Original Video")
                st.video(uploaded_video)
                
                if st.button("🎬 Process Video"):
                    with st.spinner("Processing video... This may take a while."):
                        # Process video
                        cap = cv2.VideoCapture(tfile.name)
                        
                        # Get video properties
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process frames
                        processed_frames = []
                        detections = []
                        
                        for i in range(0, frame_count, fps // 2):  # Process every 2 seconds
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, frame = cap.read()
                            
                            if ret:
                                processed_frame, pose_name, confidence = process_frame(
                                    frame, pose, drawing, mp_pose, model, labels
                                )
                                
                                if pose_name and confidence > confidence_threshold * 100:
                                    detections.append({
                                        'time': i / fps,
                                        'pose': pose_name,
                                        'confidence': confidence
                                    })
                                
                                # Update progress
                                progress = i / frame_count
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {i}/{frame_count}")
                        
                        cap.release()
                        
                        # Display results
                        st.subheader("Video Analysis Results")
                        if detections:
                            st.success(f"Found {len(detections)} pose detections!")
                            
                            # Create a DataFrame for better display
                            import pandas as pd
                            df = pd.DataFrame(detections)
                            st.dataframe(df)
                            
                            # Show most common pose
                            most_common = df['pose'].mode()[0]
                            st.info(f"Most common pose: **{most_common}**")
                        else:
                            st.warning("No confident poses detected in the video.")
                
                # Clean up
                if 'tfile' in locals():
                    os.unlink(tfile.name)
        
        else:  # Webcam mode
            st.subheader("📱 Webcam Mode")
            st.info("""
            **Note:** Webcam access works best when running locally. 
            For cloud deployment, use image or video upload modes.
            """)
            
            # Instructions for local use
            st.markdown("""
            ### To use webcam mode locally:
            1. Install requirements: `pip install streamlit opencv-python mediapipe tensorflow`
            2. Run: `streamlit run streamlit_app.py`
            3. The webcam will activate automatically
            """)
    
    with col2:
        st.subheader("📊 Detection Info")
        
        if model is not None and labels is not None:
            st.write("**Available Poses:**")
            for i, label in enumerate(labels):
                st.write(f"{i+1}. {label}")
        
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. **Upload** an image or video
        2. **Click** the analyze button
        3. **View** pose detection results
        4. **Adjust** confidence threshold if needed
        """)
        
        st.subheader("🔧 Model Info")
        st.markdown("""
        - **Framework:** TensorFlow/Keras
        - **Pose Detection:** MediaPipe
        - **Input:** 132 keypoint features
        - **Output:** Yoga pose classification
        """)

if __name__ == "__main__":
    main()