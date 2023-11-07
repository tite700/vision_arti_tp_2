import os
import cv2
import pandas as pd
from tabulate import tabulate 

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# Chemin vers le dossier contenant les images
folder_path = 'Images2'

# Initialiser un DataFrame pour stocker les moyennes des canaux BGR de chaque grain
data = pd.DataFrame()

# Initialiser une liste pour stocker les moyennes BGR
mean_bgr_list = []

# Parcourir le dossier
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            segments_fz = felzenszwalb(image, scale=275, sigma=0.55, min_size=200)

            # Afficher les résultats de la segmentation de fz
            image_res = mark_boundaries(image, segments_fz)

            # Labeliser les segments pour pouvoir les identifier
            labeled_segments = np.unique(segments_fz)

            for segment_id in labeled_segments:
                # Pour chaque région segmentée, vous pouvez créer un masque pour extraire la région segmentée.
                mask = (segments_fz == segment_id)

                #Supprimer les régions segmentées trop petites
                if np.sum(mask) < 800:
                     continue

                # Créer une image avec seulement le grain segmenté
                segment_image = image * np.stack([mask] * 3, axis=-1)

                # Afficher l'image avec le grain segmenté de titre "Grain numéro X" dont la fenetre à pour nom "Image numéro X"
                plt.figure(str(filename))
                plt.imshow(segment_image)
                plt.title("Grain numéro " + str(segment_id))
                plt.show()

                # Calculer la moyenne des canaux BGR sans prendre en compte les pixels noirs

                # Créer un masque pour les pixels non noirs
                non_black_mask = np.any(segment_image > 0, axis=-1)

                # Appliquer le masque pour exclure les pixels noirs
                non_black_pixels = segment_image[non_black_mask]

                # Calculer la moyenne des canaux BGR uniquement pour les pixels non noirs
                bgr_mean = np.mean(non_black_pixels, axis=0)

                # bgr_mean = np.mean(segment_image, axis=(0, 1))
                mean_bgr_list.append(bgr_mean)

                # Remplir le DataFrame avec les moyennes BGR de chaque image de grain
                data = data.append(pd.Series(bgr_mean, name=filename))


# Afficher le DataFrame 
data.columns = ["Moyenne de Bleu", "Moyenne de Vert", "Moyenne de Rouge"]
data.index = [f'Grain {i+1}' for i in range(len(data))]
print(tabulate(data, headers='keys', tablefmt='fancy_grid'))
