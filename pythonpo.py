import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Define the path to your model file
model_path = r"C:\Users\vamsh\OneDrive\Desktop\Potato-disease Classifier\model.h5"

# Load the model directly
model = tf.keras.models.load_model(model_path)

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
    predictions = model.predict(img_array)

    # Debugging: Print the raw prediction values
    print(f"Raw predictions: {predictions}")

    # Assuming your model is for 3 classes, map the first 3 outputs to the class labels
    class_labels = ['Early_Blight', 'Healthy', 'Late_Blight']

    # Check if the predictions array length matches the class labels length
    if predictions.shape[1] != len(class_labels):
        raise ValueError("Number of predictions does not match number of class labels")

    # Map the prediction to class labels
    predicted_class = class_labels[np.argmax(predictions[0])]

    return predicted_class

# Streamlit app
def main():
    # Display background image with transparency

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
            try:
                # Get the prediction
                prediction = predict_disease(uploaded_file)

                # Display the prediction
                st.success(f"The predicted disease class is: {prediction}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
