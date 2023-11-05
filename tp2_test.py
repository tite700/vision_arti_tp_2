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
            # Augmenter le contraste de l'image
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

            # Convertir l'image en niveaux de gris pour simplifier la segmentation
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Appliquer un algorithme de segmentation SLIC K-means pour détecter les grains
            segments = slic(image, n_segments=100, sigma=5)

            # Appliquer une opération de binarisation pour segmenter les grains
            threshold_value = 70
            binary_image = gray_image > threshold_value

            # Appliquer une opération de fermeture pour éliminer les petits trous dans les grains
            kernel = np.ones((3, 3), np.uint8)
            closing = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            # Dessiner les contours des segments sur l'image originale
            segment_contours = mark_boundaries(image, closing)

            # Enregistrer l'image résultante dans le dossier "Images_Résultat"
            result_image_path = os.path.join(result_folder, filename)
            cv2.imwrite(result_image_path, segment_contours)

            cv2.imshow('Segmentation', segment_contours)
            cv2.waitKey(0)

cv2.destroyAllWindows()
