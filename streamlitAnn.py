import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('ann_model.h5')

def preprocess_image(image):
    """Preprocess the image to match the input format of the model."""
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((8, 8))  # Resize to 8x8 pixels (assuming model was trained on 8x8 images)
    image = np.array(image)
    image = np.interp(image, (0, 255), (0, 1))  # Normalize to 0-1
    image = image.flatten().reshape(1, -1)  # Flatten to 1D array and reshape for model input
    return image

def predict(image):
    """Run the model prediction on the preprocessed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Handwritten Digit Classification App")

uploaded_file = st.file_uploader("Upload a handwritten digit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Make a prediction on the uploaded image
    prediction = predict(image)
    
    # Find the predicted class (index of the highest probability)
    predicted_class = np.argmax(prediction[0])
    
    # Display the prediction result
    st.write(f"Prediction: Digit {predicted_class}")
    
    # Optionally, display the full prediction array
    st.write("Full prediction:", prediction)
