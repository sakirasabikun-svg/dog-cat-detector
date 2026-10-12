import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 1. Page Configuration & Styling
st.set_page_config(page_title="Dog vs Cat AI Detector", page_icon="🐶🐱", layout="centered")

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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🐶🐱 Dog vs Cat AI Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Powered by Deep Learning & Transfer Learning (MobileNetV2)</p>', unsafe_allow_html=True)

# 2. Load the trained model (Cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dog_cat_model.keras')

with st.spinner("🔄 Loading AI Model... Please wait..."):
    model = load_model()

# 3. Sidebar Navigation
st.sidebar.header("⚙️ Settings & Options")
option = st.sidebar.radio("Choose Input Mode:", ("📤 Upload an Image", "📸 Take a Live Photo"))

image_to_analyze = None

if option == "📤 Upload an Image":
    st.subheader("Upload a Picture")
    uploaded_file = st.file_uploader("Choose a picture...", type=["jpg", "jpeg", "png", "webp", "heic"])
    
    if uploaded_file is not None:
        try:
            img_raw = Image.open(uploaded_file)
            img_raw = img_raw.convert('RGB')
            
            # Mobile memory safe thumbnail resizing
            img_raw.thumbnail((400, 400))
            image_to_analyze = img_raw
            
        except Exception as e:
            st.error(f"⚠️ Could not read image. Error: {e}")

elif option == "📸 Take a Live Photo":
    st.subheader("Live Camera Capture")
    camera_photo = st.camera_input("Take a live picture")
    if camera_photo is not None:
        try:
            img_raw = Image.open(camera_photo)
            img_raw = img_raw.convert('RGB')
            img_raw.thumbnail((400, 400))
            image_to_analyze = img_raw
        except Exception as e:
            st.error(f"⚠️ Error reading camera photo.")

# 4. Display & Analyze
if image_to_analyze is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        # width parameter remove kore dewa holo jate mobile browser crash na kore
        st.image(image_to_analyze, caption='Processed Image', use_column_width=True)
        
    with col2:
        st.info("🧠 AI Analysis in progress...")
        
        temp_path = "temp_image.jpg"
        image_to_analyze.save(temp_path, "JPEG")

        img = tf.keras.utils.load_img(temp_path, target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Make Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['Cat 🐱', 'Dog 🐶']
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.markdown("---")
        if "Cat" in predicted_class:
            st.success(f"### Prediction: **{predicted_class}**")
        else:
            st.warning(f"### Prediction: **{predicted_class}**")
            
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.markdown("---")
    st.warning("👉 Please upload an image or take a photo from the sidebar to get started!")