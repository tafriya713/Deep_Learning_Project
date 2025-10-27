import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices = json.load(open(f"{working_dir}/trained_model/class_indices.json"))


# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


# Function to predict class and probabilities
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_indices[str(predicted_class_index)]
    # Prepare probabilities dataframe
    class_prob_df = pd.DataFrame({
        "Class": [class_indices[str(i)] for i in range(len(predictions))],
        "Probability": predictions
    }).sort_values(by="Probability", ascending=False)
    return predicted_class_name, class_prob_df


# Streamlit UI
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
st.title("🌱 Plant Disease Classifier")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a clear image of the plant leaf.
2. Click 'Classify' to predict the disease.
3. See probabilities for all classes below.
""")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("Classify"):
            with st.spinner("Predicting..."):
                prediction, prob_df = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"Prediction: **{prediction}**")

                # Display top 5 predictions
                st.subheader("Top 5 Predictions")
                st.table(prob_df.head(5))

                # Optional: bar chart for all probabilities
                st.subheader("All Class Probabilities")
                st.bar_chart(prob_df.set_index("Class"))
