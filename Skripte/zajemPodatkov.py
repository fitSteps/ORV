import os
import requests
from PIL import Image
from io import BytesIO
import time

def scrape_images(num_images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory {folder_path}")

    for i in range(num_images):
        response = requests.get("https://thispersondoesnotexist.com")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(f"{folder_path}/image_{i+1}.jpg")
            print(f"Image {i+1} saved.")
            time.sleep(1)
        else:
            print(f"Failed to retrieve image {i+1}, Status code: {response.status_code}")

folder_path = "TPDNE_images"
scrape_images(100, folder_path)
