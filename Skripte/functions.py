import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import time
import numpy as np

def pretvori_v_sivo(image):
    return image.convert('L')

def linearizacija_sivin(image):
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(channel) for channel in channels]
    return  cv2.merge(eq_channels)

def gaussov_filter(image, sigma=1):
    return cv2.GaussianBlur(image, (0, 0), sigma)
    
def obdelava_slike(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image) 

    image = image.resize((244, 244))
    image = Image.fromarray(gaussov_filter(np.array(image), sigma=1))
    #image = Image.fromarray(linearizacija_sivin(np.array(image)))
    return image

def scrape_images(num_images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory {folder_path}")

    for i in range(num_images):
        response = requests.get("https://thispersondoesnotexist.com")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = obdelava_slike(image)


            image.save(f"{folder_path}/image_{i+1}.jpg")
            print(f"Image {i+1} saved.")
            time.sleep(0.5)
        else:
            print(f"Failed to retrieve image {i+1}, Status code: {response.status_code}")


def video_to_images(video_path, output_folder, frame_rate=1, max_frames=100):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 0
    image_id = 0

    while True:
        success, frame = video.read()
        if not success:
            break 
        if frame_id >= max_frames:
            break

        if frame_id % frame_rate == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = obdelava_slike(frame)

            if isinstance(frame, Image.Image):
                frame = np.array(frame)

            if frame.shape[-1] == 3: 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image_path = os.path.join(output_folder, f"frame_{image_id}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            image_id += 1

        frame_id += 1

    video.release()
    print("Finished extracting frames.")

def augm_horizontal_flip(image):
    if np.random.randint(0, 2) == 0:
        return image
    return cv2.flip(image, 1)

def augm_adjust_brightness(image, value=12):
    image = image.astype(np.int16)  
    image += np.random.randint(-value, value)
    image = np.clip(image, 0, 255) 
    return image.astype(np.uint8)

def augm_random_crop(image, crop_ratio=1.05):
    height, width = image.shape[:2]
    crop_height = np.random.randint(height // crop_ratio, height)
    crop_width = np.random.randint(width // crop_ratio, width)
    top = np.random.randint(0, height - crop_height)
    left = np.random.randint(0, width - crop_width)
    return cv2.resize(image[top:top + crop_height, left:left + crop_width], (width, height))

def augm_adjust_contrast(image,factor_ratio=0.05):
    image = image.astype(np.float32)
    factor = np.random.uniform(1-factor_ratio, 1+factor_ratio) 
    mean = 128
    image = (image - mean) * factor + mean
    image = np.clip(image, 0, 255) 
    return image.astype(np.uint8)



video_path = 'Me/kuplen.mp4'
video_testpath = 'Me/test.mp4'
video_frames_folder = 'Me/frames/'
video_testfolder = 'Me\\test_images_me'
scrape_folder = 'RandomPeople/'
scrape_testfolder = 'Me\Random_testfolder'
#video_to_images(video_path, video_frames_folder, frame_rate=1,max_frames=100)
#scrape_images(99, scrape_folder)
#video_to_images('Me/aljaz.mp4', 'Me\\cimri\\aljaz', frame_rate=1,max_frames=20)
#video_to_images('Me/tina.mp4', 'Me\\cimri\\tina', frame_rate=1,max_frames=20)
#video_to_images('Me/lara.mp4', 'Me\\cimri\\lara', frame_rate=1,max_frames=20)
#video_to_images('Me/patrick.mp4', 'Me\\cimri\\patrick', frame_rate=1,max_frames=20)
#video_to_images('Me/domen.mp4', 'Me\\cimri\\domen', frame_rate=1,max_frames=20)

#scrape_images(99, scrape_testfolder)




