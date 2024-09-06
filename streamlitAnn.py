import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model (Assuming it's a TensorFlow/Keras model)
model = tf.keras.models.load_model('ann_model.h5') 

def preprocess_image(image):
    """Preprocess the image before passing it to the model."""
    image = image.resize((224, 224))  # Resizing the image
    image = np.array(image) / 255.0   # Normalizing
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Run the model prediction on the image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Run the model prediction
    prediction = predict(image)
    st.write(f"Prediction: {prediction}")