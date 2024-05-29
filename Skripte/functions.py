import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import time
import numpy as np


def konvolucija(slika, jedro):
    visina_slike, sirina_slike, kanali = slika.shape
    visina_jedra, sirina_jedra = jedro.shape
    polovica_v_jedru = visina_jedra // 2
    polovica_s_jedru = sirina_jedra // 2
    
    filtrirana_slika = np.zeros_like(slika)
    slika_padded = np.pad(slika, ((polovica_v_jedru, polovica_v_jedru), (polovica_s_jedru, polovica_s_jedru), (0, 0)), mode='constant')

    for y in range(polovica_v_jedru, visina_slike + polovica_v_jedru):
        for x in range(polovica_s_jedru, sirina_slike + polovica_s_jedru):
            for c in range(kanali): 
                regija = slika_padded[y - polovica_v_jedru:y + polovica_v_jedru + 1, x - polovica_s_jedru:x + polovica_s_jedru + 1, c]
                filtrirana_slika[y - polovica_v_jedru, x - polovica_s_jedru, c] = np.sum(regija * jedro)
            
    return filtrirana_slika



def filtriraj_z_gaussovim_jedrom(slika, sigma):
    velikost_jedra = int(2 * sigma * 2 + 1)
    k = (velikost_jedra / 2) - 0.5

    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)

    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x = i - k 
            y = j - k
            jedro[i, j] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    jedro /= np.sum(jedro)
    
    return konvolucija(slika, jedro)

def pretvori_v_sivo(image):
    return image.convert('L')

def linearizacija_sivin(image):
    return Image.fromarray(cv2.equalizeHist(np.array(image)))
    

def obdelava_slike(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image) 

    image = image.resize((244+16, 244+16))
    image = Image.fromarray(filtriraj_z_gaussovim_jedrom(np.array(image), sigma=1))
    image = image.crop((8, 8, 244+8, 244+8))
    #image = Image.fromarray(linearizacija_sivin(np.array(image)))
    return image

def scrape_images(num_images, folder_path, gray=False, equalize_hist=False):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory {folder_path}")

    for i in range(num_images):
        response = requests.get("https://thispersondoesnotexist.com")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = obdelava_slike(image)
            if gray:
                image = pretvori_v_sivo(image)
                if equalize_hist:
                    image = linearizacija_sivin(image)


            image.save(f"{folder_path}/image_{i+1}.jpg")
            print(f"Image {i+1} saved.")
            time.sleep(0.5)
        else:
            print(f"Failed to retrieve image {i+1}, Status code: {response.status_code}")


def video_to_images(video_path, output_folder, frame_rate=1, max_frames=100, gray=False, equalize_hist=False):

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Capture video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 1
    image_id = 1

    while True:
        # Read next frame from video
        success, frame = video.read()
        if not success:
            break  # No more frames or error
        if frame_id % (max_frames*frame_rate) == 0:
            break

        # Save frame as JPEG file
        if frame_id % frame_rate == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = obdelava_slike(frame)
            if gray:
                frame = pretvori_v_sivo(frame)
                if equalize_hist:
                    frame = linearizacija_sivin(frame)

            # Convert back to array for saving
            if isinstance(frame, Image.Image):
                frame = np.array(frame)

            # Ensure the image is in BGR format if it is still in RGB
            if frame.shape[-1] == 3:  # Only if it's a color image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image_path = os.path.join(output_folder, f"frame_{image_id}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            image_id += 1

        frame_id += 1

    video.release()
    print("Finished extracting frames.")

def horizontal_flip(image):
    if np.random.randint(0, 2) == 0:
        return image
    return image[:, ::-1]

def adjust_brightness(image):
    image = image.astype(np.int16)  
    image += np.random.randint(-66, 66)
    image = np.clip(image, 0, 255) 
    return image.astype(np.uint8)

def random_crop(image):
    height, width = image.shape[:2]
    crop_height = np.random.randint(height // 1.5, height)
    crop_width = np.random.randint(width // 1.5, width)
    if height <= crop_height or width <= crop_width:
        raise ValueError("Crop size must be smaller than image dimensions.")
    top = np.random.randint(0, height - crop_height)
    left = np.random.randint(0, width - crop_width)
    return cv2.resize(image[top:top + crop_height, left:left + crop_width], (width, height))

def adjust_contrast(image):

    # Pretvori sliko v float za preprečevanje izgube podatkov
    image = image.astype(np.float32)
    factor = np.random.uniform(0.33, 1.66)  # Naključni faktor kontrasta
    
    # Središčenje pikslov okoli 128 in prilagajanje kontrasta
    mean = 128
    image = (image - mean) * factor + mean
    image = np.clip(image, 0, 255)  # Omeji vrednosti nazaj na [0, 255]

    return image.astype(np.uint8)


video_path = 'Me/kuplen.mp4'
video_testpath = 'Me/test.mp4'
video_frames_folder = 'Me/frames/'
video_testfolder = 'Me\\test_images_me'
scrape_folder = 'RandomPeople/'
scrape_testfolder = 'Me\Random_testfolder'
#video_to_images(video_path, video_frames_folder, frame_rate=2,max_frames=100, gray=False, equalize_hist=True)
#scrape_images(99, scrape_folder, gray=False, equalize_hist=True)




