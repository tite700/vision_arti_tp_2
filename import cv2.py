import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread("./Images/Echantillion1Mod2_301.png")

# Afficher l'image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("image")
plt.show()

# Convertir l'image en espace couleur Lab
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
a = lab_image[:, :, 1]
b = lab_image[:, :, 2]

# Utiliser la détection de contours pour définir des régions d'intérêt
edges = cv2.Canny(a, b, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

nColors = len(contours)
sample_regions = np.zeros((image.shape[0], image.shape[1], nColors), dtype=bool)

for count, contour in enumerate(contours):
    sample_regions[:, :, count] = cv2.fillPoly(np.zeros_like(image), [contour], 1).astype(bool)

# Convertir l'image en espace couleur Lab
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
a = lab_image[:, :, 1]
b = lab_image[:, :, 2]
color_markers = np.zeros((nColors, 2))

for count in range(nColors):
    color_markers[count, 0] = np.mean(a[sample_regions[:, :, count]])
    color_markers[count, 1] = np.mean(b[sample_regions[:, :, count]])

print(color_markers[1, 0], color_markers[1, 1])

color_labels = np.arange(nColors)
a = a.astype(float)
b = b.astype(float)
distance = np.zeros((a.shape[0], a.shape[1], nColors))

for count in range(nColors):
    distance[:, :, count] = np.sqrt((a - color_markers[count, 0])**2 + (b - color_markers[count, 1])**2)

label = np.argmin(distance, axis=2)
label = color_labels[label]

rgb_label = np.repeat(label[:, :, np.newaxis], 3, axis=2)
segmented_images = np.zeros((image.shape[0], image.shape[1], 3, nColors), dtype=np.uint8)

for count in range(nColors):
    color = np.copy(image)
    color[rgb_label != color_labels[count]] = 0
    segmented_images[:, :, :, count] = color

segmented_images = segmented_images.transpose(3, 0, 1, 2)

plt.figure()
plt.title("Montage of Red, Green, Purple, Magenta, and Yellow Objects, and Background")
montage = np.hstack(segmented_images)
plt.imshow(montage.transpose(1, 2, 0))
plt.show()

purple = "#774998"
plot_labels = ["k", "r", "g", purple, "m", "y"]

plt.figure()
for count in range(nColors):
    plot_label = plot_labels[count]
    plt.plot(a[label == count], b[label == count], ".", markersize=2, markeredgecolor=plot_label, markerfacecolor=plot_label)

plt.title("Scatterplot of Segmented Pixels in a*b* Space")
plt.xlabel("a* Values")
plt.ylabel("b* Values")
plt.show()
