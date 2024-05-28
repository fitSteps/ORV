import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import time
import numpy as np
import numba
from numba import jit, prange

'''
def linearizacija_sivin(image, min_val=32, max_val=255-32):
     # Calculate the minimum and maximum of the image
    imin, imax = image.min(), image.max()

    # Scale the image
    image_scaled = (image - imin) / (imax - imin) * (max_val - min_val) + min_val
    image_scaled = np.clip(image_scaled, min_val, max_val).astype('uint8')
    
    return image_scaled
'''

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
    image = image.resize((128+16, 128+16))
    image = Image.fromarray(filtriraj_z_gaussovim_jedrom(np.array(image), sigma=1))
    image = image.crop((8, 8, 128+8, 128+8))
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

folder_path = "RandomPeople"
num_images = 10
gray = False
equalize_hist = False
scrape_images(num_images, folder_path, gray, equalize_hist)
