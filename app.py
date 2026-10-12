import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 1. Page Configuration & Styling
st.set_page_config(page_title="Dog vs Cat AI Detector", page_icon="🐶🐱", layout="centered")

# Custom CSS for Modern UI, Card Layouts & Soft Colors
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Header Styling */
    .main-title {
        text-align: center;
        color: #1E1E2F;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-text {
        text-align: center;
        color: #6C757D;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Custom Card Design */
    .custom-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Metric Card Customization */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# App Header Section
st.markdown('<p class="main-title">🐾 Pet Classifier AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Experience High-Accuracy Deep Learning (MobileNetV2)</p>', unsafe_allow_html=True)

# 2. Load the trained model (Cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dog_cat_model.keras')

with st.spinner("🔄 Initializing AI Engine... Please wait..."):
    model = load_model()

# 3. Sidebar Navigation with Clean Styling
st.sidebar.markdown("### ⚙️ Navigation Panel")
option = st.sidebar.radio("Select Input Mode:", ("📤 Upload an Image", "📸 Take a Live Photo"))

image_to_analyze = None

if option == "📤 Upload an Image":
    st.markdown("#### 📁 Upload Pet Picture")
    uploaded_file = st.file_uploader("Choose a high-resolution image...", type=["jpg", "jpeg", "png", "webp", "heic"])
    
    if uploaded_file is not None:
        try:
            img_raw = Image.open(uploaded_file)
            img_raw = img_raw.convert('RGB')
            img_raw.thumbnail((400, 400))
            image_to_analyze = img_raw
        except Exception as e:
            st.error(f"⚠️ Could not read image. Error: {e}")

elif option == "📸 Take a Live Photo":
    st.markdown("#### 📷 Live Camera Capture")
    camera_photo = st.camera_input("Capture pet photo directly")
    if camera_photo is not None:
        try:
            img_raw = Image.open(camera_photo)
            img_raw = img_raw.convert('RGB')
            img_raw.thumbnail((400, 400))
            image_to_analyze = img_raw
        except Exception as e:
            st.error(f"⚠️ Error reading camera photo.")

# 4. Display & Modern Results Dashboard
if image_to_analyze is not None:
    st.markdown("---")
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("**🖼️ Preview Image**")
        st.image(image_to_analyze, use_column_width=True)
        
    with col2:
        st.markdown("**📊 Analysis Result**")
        with st.spinner("🧠 Analyzing features..."):
            temp_path = "temp_image.jpg"
            image_to_analyze.save(temp_path, "JPEG")

            img = tf.keras.utils.load_img(temp_path, target_size=(150, 150))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Prediction
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            class_names = ['Cat 🐱', 'Dog 🐶']
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

        # UI Result Box Styling
        if "Cat" in predicted_class:
            st.success(f"### Result: **{predicted_class}**")
        else:
            st.warning(f"### Result: **{predicted_class}**")
            
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")

        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.markdown("---")
    st.info("💡 **Tip:** Use the sidebar menu to upload an image from your device or take a live photo to start the AI analysis!")