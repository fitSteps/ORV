import functions
import paho.mqtt.client as mqtt

functions.scrape_images(500, "./scraped_images")

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("image")
    client.subscribe("video")

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    # Here, you can add if-else blocks to handle different messages
    userID = msg.payload
    if msg.topic == "image":
        # Call the function to comapre image to AI model
        print("Comparing image to AI model")
    
    if msg.topic == "video":
        print("Converting video to images")

        functions.video_to_images("./{userID}.mp4", "./{userID}/video_frames", frame_rate=1, max_frames=1000)

        print("Creating AI model from video frames")
        
        # Call the function to create AI model


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting.
client.loop_forever()
