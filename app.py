import keras.utils as image
from keras.applications.resnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

# Define the disease labels
labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict_disease(img_path):
    # Load and preprocess the input image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted disease label
    predicted_disease = labels[predicted_class_index]

    return predicted_disease




# Streamlit app
def main():
        
    st.title("DETECTION AND CLASSIFICATION OF DIABETIC RETINOPATHY, GLUACOMA AND CATARACT")
    

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", width=300)
        st.write("")
        st.markdown("<h3 style='text-align: center; '>Classifying...</h3>", unsafe_allow_html=True)

       
        # Make prediction
        prediction = predict_disease(uploaded_file)
        st.write(f"<h1 style='text-align: center;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
        
        

if __name__ == "__main__":
    main()
