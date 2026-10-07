import tensorflow as tf
import numpy as np
import os

# 1. Define image parameters
img_height = 150
img_width = 150

# 2. Load the trained model
print("Loading the trained model...")
model = tf.keras.models.load_model('dog_cat_model.keras')

# 3. Load and preprocess the test image
image_path = 'test_image.jpg' 

if not os.path.exists(image_path):
    print(f"Error: Could not find '{image_path}'. Please make sure the image is in the folder.")
else:
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 4. Make a prediction
    print("Analyzing the image...")
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Convert to percentages

    class_names = ['cat', 'dog']
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 5. Print the final result
    print("\n" + "="*40)
    print(f"🐶🐱 PREDICTION RESULT 🐶🐱")
    print(f"The model thinks this is a: {predicted_class.upper()}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("="*40 + "\n")