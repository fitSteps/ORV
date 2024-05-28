import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the VGG16 model with pre-trained ImageNet weights
model = VGG16(weights='imagenet')

# Summary of the model
model.summary()

# Function to preprocess the image and make a prediction
def preprocess_and_predict(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Decode the prediction
    decoded_prediction = decode_predictions(prediction, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_prediction):
        print(f"{i+1}: {label} ({score:.2f})")

# Test with a single image
test_image_path = 'Me\\frames\\frame_1.jpg'  # Replace with the path to your image
preprocess_and_predict(test_image_path)
