import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Process the MQTT message for testing the model.')
parser.add_argument('mqtt_message', type=str, help='MQTT message payload')
args = parser.parse_args()

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(model_path, image_path, label):
    # Load the pre-trained model
    model = load_model(model_path)

    # Process the image
    img = load_and_preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)
    print("Prediction score for "+ label+" :", prediction)

    # Interpret the prediction
    #if prediction < 0.5:
     #   print("The image is classified as NOT YOU.")
   # else:
     #   print("The image is classified as YOU.")
# Specify the paths
model_path = f'/ai_models/{args.mqtt_message}.h5'
image_path = 'Me\\test_images_me\\frame_50.jpg' 

# Run the prediction function
#predict_image(model_path, test_image_path_patrick, "patrick")
#predict_image(model_path, test_image_path_me, "js")
#predict_image(model_path, test_image_path_random, "random")
