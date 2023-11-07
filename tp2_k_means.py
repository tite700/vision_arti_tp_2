import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

def process_image(file_path):
    # Lire l'image
    image = cv2.imread(file_path)
    # Convertir l'image en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Appliquer un filtre Gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)

    # Reshaper l'image pour qu'elle soit une liste de pixels pour k-means
    pixels = blurred.reshape((-1, 3))
    
    # Utiliser KMeans pour segmenter l'image en clusters de couleur
    kmeans = KMeans(n_clusters=40) # Le nombre de clusters (grains) est un paramètre
    kmeans.fit(pixels)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    
    # Labels pour chaque pixel
    labels = kmeans.labels_
    # Reshaper les labels en une image
    labels_image = labels.reshape(blurred.shape[:2])
    
    # Créer un dataframe pour stocker les résultats
    df = pd.DataFrame(columns=['R_mean', 'G_mean', 'B_mean'])

    print(f"Nombre de grains: {len(dominant_colors)}")
    print(f"Valeurs des couleurs dominantes: {dominant_colors}")

    # Pour chaque couleur dominante, trouver les pixels correspondants et calculer la moyenne RGB
    tolerance = 20
    for color in dominant_colors:
        # Créer un masque basé sur la tolérance de couleur
        diff = np.abs(image_rgb.astype(np.int) - color.astype(np.int))
        mask = np.all(diff <= tolerance, axis=-1)
        print(f"Nombre de pixels pour la couleur {color}: {np.sum(mask)}")
        # Trouver les pixels de l'image originale où le masque est True
        selected_pixels = image_rgb[mask]
        print(f"Nombre de pixels sélectionnés: {len(selected_pixels)}")
        if len(selected_pixels) > 0:
            # Calculer les moyennes de ces pixels
            mean_colors = np.mean(selected_pixels, axis=0)
            df = df.append({'R_mean': mean_colors[0], 'G_mean': mean_colors[1], 'B_mean': mean_colors[2]}, ignore_index=True)
    
    return df

# Parcourir le dossier "./Images" et appliquer la fonction sur chaque image
results = {}
for filename in os.listdir("./Images"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join("./Images", filename)
        results[filename] = process_image(filepath)

# Affichage des résultats
for filename, df in results.items():
    print(f"Résultats pour {filename}:")
    print(df)
