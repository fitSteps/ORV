import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
import paho.mqtt.client as mqtt


# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Process the MQTT message for testing the model.')
parser.add_argument('mqtt_message', type=str, help='MQTT message payload')
args = parser.parse_args()

# MQTT Settings
MQTT_BROKER = "172.201.117.179"
MQTT_PORT = 1883
RESPONSE_TOPIC = f"topic/{args.mqtt_message}"  # Topic to publish responses to

# Define event callback
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully.")
        
    else:
        print(f"Connection failed with code {rc}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker with result code " + str(rc))

# Set up client
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model_path, image_path):
    # Load the pre-trained model
    model = load_model(model_path)

    # Process the image
    img = load_and_preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)

    if prediction > 0.8:
        client.publish(RESPONSE_TOPIC, f"Authenticated  {prediction}")
        #print("succ"+prediction)
    else:
        #print("fail"+prediction)
        client.publish(RESPONSE_TOPIC, f"Not authenticated {prediction}")
# Specify the paths
model_path = f'/ai_models/{args.mqtt_message}.h5'
image_path = f'/app/photos/{args.mqtt_message}.jpg' 

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    predict_image(model_path, image_path)
except Exception as e:
    print(f"Could not connect to MQTT broker: {e}")
