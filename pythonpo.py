import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

model_directory = "C:\\Users\\vamsh\\OneDrive\\Desktop\\New folder\\models\\1"
loaded_model = load_model(model_directory)

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(img_array)

    # Map the prediction to class labels
    class_labels = ['Abhisheck',
 'Anandam',
 'Bharat Bukya',
 'Karthik',
 'Laxmi',
 'Pravalika',
 'Rajasheckar',
 'Sanjeeva',
 'Shiva mani(laddu)',
 'Vinay',
 'vamshi']
    predicted_class = class_labels[np.argmax(predictions[0])]

    return predicted_class

# Streamlit app
def main():
    # Display background image with transparency
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://example.com/path/to/agriculture_background.png');
            background-size: cover;
        }
        .content-container {
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.7); /* Adjust the alpha value for transparency */
            border-radius: 10px;
            margin: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a container with a transparent background
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    # Title
    st.title("Potato Disease Classifier")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with reduced width
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True, width=400)

        # Make predictions on button click
        if st.button("Predict Disease"):
            # Get the prediction
            prediction = predict_disease(uploaded_file)

            # Display the prediction
            st.success(f"The predicted disease class is: {prediction}")

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()















