import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 1. Page Configuration & Styling
st.set_page_config(page_title="Dog vs Cat AI Detector", page_icon="🐶🐱", layout="centered")

# Custom CSS for gorgeous UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .sub-text {
        text-align: center;
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown('<p class="main-title">🐶🐱 Dog vs Cat AI Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Powered by Deep Learning & Transfer Learning (MobileNetV2)</p>', unsafe_allow_html=True)

# 2. Load the trained model (Cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dog_cat_model.keras')

with st.spinner("🔄 Loading AI Model... Please wait..."):
    model = load_model()

# 3. Sidebar or Radio for Options
st.sidebar.header("⚙️ Settings & Options")
option = st.sidebar.radio("Choose Input Mode:", ("📤 Upload an Image", "📸 Take a Live Photo"))

image_to_analyze = None

# Option A: File Uploader
if option == "📤 Upload an Image":
    st.subheader("Upload a Picture")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG or PNG file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file)

# Option B: Camera Input
elif option == "📸 Take a Live Photo":
    st.subheader("Live Camera Capture")
    camera_photo = st.camera_input("Smile! Take a picture of a dog or cat")
    if camera_photo is not None:
        image_to_analyze = Image.open(camera_photo)

# 4. Display & Analyze the Image
if image_to_analyze is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_to_analyze, caption='Selected Image', use_container_width=True)
        
    with col2:
        st.info("🧠 AI Analysis in progress...")
        
        # Preprocessing
        temp_path = "temp_image.jpg"
        image_to_analyze = image_to_analyze.convert('RGB')
        image_to_analyze.save(temp_path)

        img = tf.keras.utils.load_img(temp_path, target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Make Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['Cat 🐱', 'Dog 🐶']
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # Show Results with styling
        st.markdown("---")
        if "Cat" in predicted_class:
            st.success(f"### Prediction: **{predicted_class}**")
        else:
            st.warning(f"### Prediction: **{predicted_class}**")
            
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.markdown("---")
    st.warning("👉 Please upload an image or take a photo from the sidebar to get started!")