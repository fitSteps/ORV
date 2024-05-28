import os
import cv2
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

def process_images(folder_path, output_folder, sigma):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            slika = cv2.imread(img_path)
            if slika is not None:
                filtrirana_slika = filtriraj_z_gaussovim_jedrom(slika, sigma)
                output_path = os.path.join(output_folder, f'filtered_{filename}')
                cv2.imwrite(output_path, filtrirana_slika)
                print(f'Processed and saved {output_path}')

# Example Usage
folder_path = '../TPDNE_images'
output_folder = '../Filtered_Images'
process_images(folder_path, output_folder, sigma=1)
