import os
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray
from skimage import morphology
from skimage.filters import threshold_otsu

# Chemin vers le dossier contenant les images
folder_path = 'Images'


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
            threshold_value = threshold_otsu(gray_image)
            #threshold_value = 80
            binary_image = gray_image > threshold_value

            # Dessiner les contours des segments sur l'image originale
            segment_contours = mark_boundaries(image, binary_image)

            cv2.imshow('Segmentation', segment_contours)
            cv2.waitKey(0)

cv2.destroyAllWindows()
