import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import time
import numpy as np
import numba
from numba import jit, prange

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



def scrape_images(num_images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory {folder_path}")

    for i in range(num_images):
        response = requests.get("https://thispersondoesnotexist.com")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))

            resized_image = image.resize((288, 288))

            filtered_image = Image.fromarray(filtriraj_z_gaussovim_jedrom(np.array(resized_image), sigma=1))

            cropped_image = filtered_image.crop((16, 16, 272, 272))

            cropped_image.save(f"{folder_path}/image_{i+1}.jpg")
            print(f"Image {i+1} saved.")
            time.sleep(0.5)
        else:
            print(f"Failed to retrieve image {i+1}, Status code: {response.status_code}")

folder_path = "Images"
scrape_images(10, folder_path)
