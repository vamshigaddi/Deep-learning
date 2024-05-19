# Potato Disease Classification

This project aims to classify potato leaf diseases using a deep learning model. The model can identify three classes: Early Blight, Healthy, and Late Blight. The project utilizes a convolutional neural network (CNN) built with TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [License](#license)
## DEMO
https://www.loom.com/share/19de17de7b724816a64f70af6f6726e3?sid=c0d6b617-1a74-4160-9c6b-c927e40db8a6
## Introduction
Potato diseases can significantly impact crop yield and quality. Early detection and classification of these diseases can help in timely intervention and control measures. This project uses image classification techniques to identify common potato leaf diseases.

## Dataset
The dataset used for training the model consists of images of potato leaves categorized into three classes:
#### You can download the dataset from the Kaggle
- Early Blight
- Healthy
- Late Blight

The images should be organized in subdirectories corresponding to each class.

## Installation
To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/vamshigaddi/potato-disease-classification.git
cd potato-disease-classification
pip install -r requirements.txt
```
## Usage
```bash
import numpy as np
from PIL import Image
import tensorflow as tf

# Define the path to your model file
model_path ="/content/model.h5"

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
```
## Streamlit APP
  ```bash
  Streamlit run python.py
  ```
   
# Model Training
- Download the Dataset
- Load the data with tensorflow preprocessing API
- Split the data into train,test,valid 
- Build CNN model from scratch or Load a pretrained Model
- Train the model
- Inference the values

## Results
 ```bash
Set All the Parameters
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=20
Accuracy-99%
 ```


## License

This project is licensed under the MIT License. See the LICENSE file for details.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

