import cv2
import os
import face_recognition
from sklearn import svm
import numpy as np

# Ruta del dataset
dataset_path = 'dataset'

# Tamaño de imagen
img_size = (224, 224)

# Preprocesar imágenes
for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file))
    img = cv2.resize(img, img_size)
    cv2.imwrite(os.path.join(dataset_path, file), img)

# Entrenar modelo
X = []
y = []

for file in os.listdir(dataset_path):
    img = face_recognition.load_image_file(os.path.join(dataset_path, file))
    encodings = face_recognition.face_encodings(img)
    
    if len(encodings) > 0:
        encoding = encodings[0]
        X.append(encoding)
        y.append(file.split('.')[0])

# Verificar si hay etiquetas duplicadas
if len(set(y)) != len(y):
    raise ValueError("Hay etiquetas duplicadas en el conjunto de entrenamiento")

clf = svm.SVC()
clf.fit(np.array(X), np.array(y))