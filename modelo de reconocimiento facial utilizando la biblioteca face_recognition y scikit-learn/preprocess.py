import cv2
import os

# Ruta del dataset
dataset_path = 'dataset'

# Tamaño de imagen
img_size = (224, 224)

# Preprocesar imágenes
for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file))
    img = cv2.resize(img, img_size)
    cv2.imwrite(os.path.join(dataset_path, file), img)
