import functions
import paho.mqtt.client as mqtt
import signal
import sys
import subprocess


# MQTT Settings
MQTT_BROKER = "172.201.117.179"
MQTT_PORT = 1883
TOPICS = [("image", 0), ("video", 0)]  # List of topics to subscribe to with QoS level 0
RESPONSE_TOPIC = "test"  # Topic to publish responses to

# Define event callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully.")
        client.subscribe(TOPICS)
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    message_content = msg.payload.decode('utf-8')
    print(f"Received message on topic: {msg.topic}")
    print("Message content:", message_content)

    # Responding on topic 'test' based on the message topic received
    if msg.topic == "image":
        client.publish(RESPONSE_TOPIC, "image")
        print("Published 'image' to topic 'test'")
        # Run ai_model.py with the MQTT message as an argument
        subprocess.run(['python', 'Skripte/model_test.py', message_content])
    elif msg.topic == "video":
        client.publish(RESPONSE_TOPIC, "video"+message_content)
        print("Published 'video' to topic 'test'")
        # Run model_test.py with the MQTT message as an argument
        subprocess.run(['python', 'Skripte/ai_model.py', message_content])

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker with result code " + str(rc))

def handle_sigterm(*args):
    print("SIGTERM received, shutting down...")
    client.loop_stop()  # Gracefully stop the loop
    client.disconnect()  # Disconnect the MQTT client
    sys.exit(0)  # Exit cleanly

# Set up client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)  # Also handle Ctrl-C gracefully

try:
    print("Scraping images")
    functions.scrape_images(500, "./scraped_images")
    print("Scraping finished")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()  # Use loop_forever instead of loop_start
except Exception as e:
    print(f"Could not connect to MQTT broker: {e}")

# The finally block is not needed because loop_forever will handle the loop