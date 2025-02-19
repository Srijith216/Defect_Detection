import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model("defect_classification_model.h5")

# Define class labels
class_labels = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Streamlit UI
st.title("Coil Surface Defect Detector")
st.write("Upload a BMP image to classify the defect.")

uploaded_file = st.file_uploader("Choose an image...", type=["bmp", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert BMP to RGB if necessary
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((150, 150))  # Resize to match model input
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"### Predicted Defect: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
