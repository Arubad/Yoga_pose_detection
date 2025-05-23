# Yoga Pose Detection - Streamlit Web App

A beautiful, interactive web application for yoga pose detection using Streamlit, perfect for deployment on Streamlit Cloud.

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a new GitHub repository** (or use existing one)

2. **Upload these files to your repository:**
   ```
   yoga-pose-detection-streamlit/
   ├── streamlit_app.py          # Main Streamlit application
   ├── requirements.txt          # Python dependencies
   ├── .streamlit/
   │   └── config.toml          # Streamlit configuration
   ├── model/                   # Your model files
   │   ├── model.h5            # Your trained model
   │   └── labels.npy          # Your labels
   └── README.md               # This file
   ```

3. **Add your model files:**
   - Upload your `model.h5` and `labels.npy` files to the `model/` directory
   - Make sure these files are committed to your repository

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with your GitHub account**

3. **Click "New app"**

4. **Fill in the deployment form:**
   - **Repository:** Select your repository (e.g., `username/yoga-pose-detection-streamlit`)
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Choose a custom URL (optional)

5. **Click "Deploy!"**

6. **Wait for deployment** (usually takes 2-5 minutes)

7. **Your app will be live** at `https://your-app-name.streamlit.app`

## 🎯 Features

- **📷 Image Upload:** Upload and analyze yoga pose images
- **🎥 Video Processing:** Process entire videos for pose detection
- **📱 Responsive Design:** Works on desktop and mobile
- **🎛️ Interactive Controls:** Adjust confidence thresholds
- **📊 Detailed Results:** View pose names and confidence scores
- **🎨 Beautiful UI:** Professional styling with custom CSS

## 🛠️ Local Development

### Run Locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yoga-pose-detection-streamlit.git
   cd yoga-pose-detection-streamlit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model files:**
   ```bash
   mkdir -p model
   cp /path/to/your/model.h5 model/
   cp /path/to/your/labels.npy model/
   ```

4. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser:** Go to `http://localhost:8501`

## 📁 Project Structure Explained

### `streamlit_app.py`
- Main application file
- Contains all UI components and logic
- Handles image/video processing
- Integrates with your ML model

### `requirements.txt`
- Lists all Python dependencies
- Optimized for Streamlit Cloud
- Uses `opencv-python-headless` for cloud compatibility

### `.streamlit/config.toml`
- Streamlit configuration
- Custom theme colors
- Server settings for cloud deployment

### `model/` Directory
- **Important:** Must contain your trained model files
- `model.h5` - your trained TensorFlow model
- `labels.npy` - array of yoga pose labels

## 🔧 Customization

### Change Colors:
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#your-color"
backgroundColor = "#your-bg-color"
```

### Add More Poses:
1. Retrain your model with additional poses
2. Update the `labels.npy` file
3. Replace the model files in the `model/` directory

### Modify UI:
Edit the `streamlit_app.py` file:
- Change layout with `st.columns()`
- Add new components with Streamlit widgets
- Customize styling with CSS in `st.markdown()`

## 🚨 Important Notes for Streamlit Cloud

### Model Files:
- **Must be included in your Git repository**
- GitHub has a 100MB file size limit
- If your model is larger, consider:
  - Using Git LFS (Large File Storage)
  - Hosting model files externally and downloading them
  - Using a smaller model architecture

### Performance:
- Streamlit Cloud provides limited resources
- Processing large videos may timeout
- Consider limiting video processing length

### Security:
- Don't commit sensitive data
- Model files will be public if your repo is public
- Consider using private repositories for proprietary models

## 🐛 Troubleshooting

### Common Issues:

1. **"Model files not found"**
   - Ensure `model.h5` and `labels.npy` are in the `model/` directory
   - Check file paths are correct
   - Verify files are committed to Git

2. **Import errors**
   - Check `requirements.txt` has all dependencies
   - Ensure TensorFlow version compatibility

3. **Deployment fails**
   - Check Streamlit Cloud logs
   - Verify all files are in the repository
   - Ensure requirements.txt is correct

4. **App runs but no predictions**
   - Check model loading in the logs
   - Verify model input dimensions (should be 132 features)
   - Check confidence threshold settings

## 📞 Support

If you encounter issues:
1. Check the [Streamlit Community Forum](https://discuss.streamlit.io/)
2. Review [Streamlit Documentation](https://docs.streamlit.io/)
3. Check your app logs in Streamlit Cloud dashboard

## 🎉 Success!

Once deployed, your yoga pose detection app will be:
- ✅ Publicly accessible
- ✅ Automatically updated when you push to GitHub
- ✅ Free to host (with Streamlit Cloud limits)
- ✅ Professional looking and responsive

Share your app URL with others and enjoy your deployed ML application!