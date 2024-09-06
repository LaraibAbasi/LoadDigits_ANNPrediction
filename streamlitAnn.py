import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load your trained model (Assuming it's a model trained on 8x8 images from load_digits)
model = joblib.load('ANN_model.pkl')

def preprocess_image(image):
    """Preprocess the image to match the input format of the load_digits model."""
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Resize the image to 8x8 pixels (the size used in the load_digits dataset)
    image = image.resize((8, 8))
    
    # Convert image to a numpy array and normalize pixel values to match the dataset's scale
    image = np.array(image)
    image = np.interp(image, (0, 255), (0, 16))  # Scale pixels to 0-16 like the dataset

    # Flatten the image to a 1D array (64 features)
    image = image.flatten().reshape(1, -1)
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
    
    # Display the prediction result
    st.write(f"Prediction: Digit {prediction[0]}")
