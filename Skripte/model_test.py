import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(model_path, image_path):
    # Load the pre-trained model
    model = load_model(model_path)

    # Process the image
    img = load_and_preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)
    print("Prediction score:", prediction)

    # Interpret the prediction
    if prediction < 0.5:
        print("The image is classified as NOT YOU.")
    else:
        print("The image is classified as YOU.")

# Specify the paths
model_path = 'face_verification_model.h5'
test_image_path = 'Me\\test_images_me\\frame_5.jpg'  # Update this path to your test image

# Run the prediction function
predict_image(model_path, test_image_path)
