import os
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray
from skimage import morphology

# Chemin vers le dossier contenant les images
folder_path = 'Images'

# Créer un dossier pour stocker les images résultantes
result_folder = 'Images_Résultat'
os.makedirs(result_folder, exist_ok=True)

# Initialiser un DataFrame pour stocker les moyennes des canaux BGR de chaque grain
data = pd.DataFrame()

# Initialiser une liste pour stocker les moyennes BGR
mean_bgr_list = []

# Parcourir le dossier
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filtrer les fichiers image par extension
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            
            # Appliquer SLIC
            segments = slic(image, n_segments=30, compactness=10, sigma=1)

            # Appliquer Fz
            segments = fz

            #essayer fz + afficher resultats segments

            #label region
            #pour chaque region segmentee tu vas faire un mask
            #recup moyenne

            # Créer une image avec les contours des segments
            image_segmented = mark_boundaries(image, segments)

            # Enregistrer l'image avec le grain segmenté
            result_path = os.path.join(result_folder, filename)

            # Calculer la moyenne des canaux BGR
            mean_bgr = np.mean(image, axis=(0, 1))
            mean_bgr_list.append(mean_bgr)

            # Afficher les moyennes BGR
            print(f'{filename} : {mean_bgr}')

            # Afficher l'image
            cv2.imshow(filename, image)
            cv2.waitKey(0)

cv2.destroyAllWindows()
