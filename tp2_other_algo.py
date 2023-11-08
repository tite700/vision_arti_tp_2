import os
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt

# Chemin vers le dossier contenant les images
folder_path = 'Images'

# Créer un dossier pour stocker les images résultantes
result_folder = 'Images_Resultat'
os.makedirs(result_folder, exist_ok=True)

# Initialiser un DataFrame pour stocker les moyennes des canaux BGR de chaque grain
data = pd.DataFrame()

# Initialiser une liste pour stocker les moyennes BGR
mean_bgr_list = []

def algorithm_threshold(image, show_img=False):
    # copy the original image
    original_image = image.copy()
    if show_img:
        cv2.imshow("original image", original_image)
        cv2.waitKey(0)

    # Augmenter le contraste de l'image
    image = cv2.convertScaleAbs(image, alpha=5, beta=-125)
    if show_img:
        cv2.imshow("convert scale abs", image)
        cv2.waitKey(0)

    # Convertir l'image en niveaux de gris pour simplifier la segmentation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_img:
        cv2.imshow("gray image", gray_image)
        cv2.waitKey(0)

    _, thresh_img = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if show_img:
        cv2.imshow("thresh_img", thresh_img)
        cv2.waitKey(0)

    # Appliquer une opération de fermeture pour éliminer les petits trous dans les grains
    kernel = np.ones((9, 9), np.uint8)
    closing = cv2.morphologyEx(thresh_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if show_img:
        cv2.imshow("closing", closing)
        cv2.waitKey(0)

    # Dessiner les contours des segments sur l'image originale
    segment_contours = mark_boundaries(original_image, closing)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if show_img:
        cv2.imshow('Segmentation', segment_contours)
        cv2.waitKey(0)

    calculate_dataframe(original_image, contours)
    cv2.destroyAllWindows()

    return segment_contours

def algorithm_canny(image, show_img=False):
    original_image = image.copy()

    image = cv2.convertScaleAbs(image, alpha=4, beta=0)
    if show_img:
        cv2.imshow("convert scale abs", image)
        cv2.waitKey(0)

    # Convertir l'image en niveaux de gris pour simplifier la segmentation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_img:
        cv2.imshow("gray image", gray_image)
        cv2.waitKey(0)

    edges = cv2.Canny(gray_image,200,50)
    if show_img:
        cv2.imshow("edges", edges)
        cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # draw white filled contour on black background
        cv2.drawContours(edges, [c], 0, 255, -1)

    if show_img:
        cv2.imshow('edges', edges)
        cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    calculate_dataframe(original_image, contours)

    cv2.drawContours(original_image, contours, -1, [255, 0, 0], 1)
    if show_img:
        cv2.imshow('original_image', original_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return original_image


def algorithm_watershed(image, show_img=False):
    original_image = image.copy()

    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])

    sharp_image = cv2.filter2D(image, -1, sharpen_filter)
    if show_img:
        cv2.imshow("sharp_image", sharp_image)
        cv2.waitKey(0)

    image = cv2.convertScaleAbs(image, alpha=2.9, beta=0)
    if show_img:
        cv2.imshow("convert scale abs", image)
        cv2.waitKey(0)

    # Convertir l'image en niveaux de gris pour simplifier la segmentation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_img:
        cv2.imshow("gray image", gray_image)
        cv2.waitKey(0)

    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if show_img:
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    if show_img:
        cv2.imshow('sure_bg', sure_bg)
        cv2.waitKey(0)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    if show_img:
        cv2.imshow('sure_fg', sure_fg)
        cv2.waitKey(0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    if show_img:
        cv2.imshow('unknown', unknown)
        cv2.waitKey(0)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_image, markers)

    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    calculate_dataframe(original_image, contours)

    original_image[markers == -1] = [255, 0, 0]
    if show_img:
        cv2.imshow('original_image', original_image)
        cv2.waitKey(0)



    cv2.destroyAllWindows()

    return original_image

def calculate_dataframe(image, contours):
    mean_bgr_list = []

    i = 0
    for c in contours:
        # Pour chaque région segmentée, vous pouvez créer un masque pour extraire la région segmentée.
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

        mean = cv2.mean(image, mask)
        mean_bgr_list.append(list(mean[:3]))
        i+=1

    data = pd.DataFrame(mean_bgr_list, columns=["Moyenne de B", "Moyenne de G", "Moyenne de R"])
    print(data)

    return data


# Parcourir le dossier
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filtrer les fichiers image par extension
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            segment_contours = algorithm_watershed(image, show_img=False)
            cv2.imshow('segment_contours', segment_contours)
            cv2.waitKey(0)

            # Enregistrer l'image résultante dans le dossier "Images_Résultat"
            result_image_path = os.path.join(result_folder, filename)
            cv2.imwrite(result_image_path, segment_contours)

cv2.destroyAllWindows()
