#Proyecto de Reconocimiento Facial con TensorFlow y Face Recognition
Autor: Gustavo Isidro Arretureta González
Correo: gustavo25880521@gmail.com
Descripción:
Este proyecto utiliza la biblioteca Face Recognition y TensorFlow para crear un sistema de reconocimiento facial. La aplicación permite entrenar un modelo de reconocimiento facial utilizando un conjunto de imágenes y luego evaluar su precisión.
Características:
Entrenamiento de modelo de reconocimiento facial utilizando Face Recognition y TensorFlow
Evaluación de precisión del modelo utilizando métricas de clasificación
Interfaz gráfica para seleccionar conjunto de imágenes y configurar parámetros del modelo
Visualización de proceso de entrenamiento y resultados
Requisitos:
Python 3.x
Face Recognition
TensorFlow
OpenCV
Pillow
Scikit-learn
Instalación:
Clonar el repositorio
Instalar dependencias utilizando pip install -r requirements.txt
Ejecutar la aplicación utilizando python main.py
Uso:
Seleccionar conjunto de imágenes para entrenamiento
Configurar parámetros del modelo (kernel, gamma, C)
Entrenar modelo
Evaluar precisión del modelo
Estructura del proyecto:
main.py: Archivo principal de la aplicación
entrenar_modelo.py: Función para entrenar modelo de reconocimiento facial
seleccionar_dataset.py: Función para seleccionar conjunto de imágenes
actualizar_gamma.py y actualizar_C.py: Funciones para actualizar parámetros del modelo
README.md: Documentación del proyecto
Licencia:
Este proyecto está bajo licencia Apache 2.0. Puedes utilizar, modificar y distribuir el código libremente, siempre y cuando se cumplan los términos de la licencia.
Copyright 2023 Gustavo Isidro Arretureta González
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
