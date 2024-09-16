# Proyecto de Clasificación con Regresión Logística

Este proyecto utiliza un modelo de regresión logística para clasificar datos de ejemplo. El código incluye la creación de un DataFrame, la división de los datos en conjuntos de entrenamiento y prueba, el entrenamiento del modelo, la evaluación del modelo y la visualización de los resultados mediante gráficos.

## Requisitos

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Puedes instalar las bibliotecas necesarias utilizando pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Uso

1. **Crear y Guardar el DataFrame**: El código crea un DataFrame de ejemplo y lo guarda en un archivo CSV llamado `tus_datos.csv`.

2. **Cargar y Dividir los Datos**: Los datos se cargan desde el archivo CSV y se dividen en conjuntos de entrenamiento y prueba.

3. **Entrenar el Modelo**: Se entrena un modelo de regresión logística utilizando los datos de entrenamiento.

4. **Hacer Predicciones y Evaluar el Modelo**: Se hacen predicciones con los datos de prueba y se evalúa el modelo utilizando una matriz de confusión y un informe de clasificación.

5. **Visualización de Resultados**: Se crean gráficos para visualizar la dispersión de las características y la matriz de confusión.

## Ejecución del Código

Para ejecutar el código, simplemente copia y pega el siguiente script en tu entorno de desarrollo:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Crear un DataFrame de ejemplo
data = pd.DataFrame({
    'caracteristica1': [1, 2, 3, 4, 5],
    'caracteristica2': [5, 4, 3, 2, 1],
    'etiqueta': [0, 1, 0, 1, 0]
})

# Guardar el DataFrame en un archivo CSV
data.to_csv('tus_datos.csv', index=False)

# Cargar datos
data = pd.read_csv('tus_datos.csv')

# Dividir los datos
X = data.drop('etiqueta', axis=1)
y = data['etiqueta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo con zero_division
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Crear un panel de control con gráficos
plt.figure(figsize=(10, 5))

# Gráfico de dispersión
plt.subplot(1, 2, 1)
sns.scatterplot(x='caracteristica1', y='caracteristica2', hue='etiqueta', data=data)
plt.title('Dispersión de Características')

# Matriz de confusión
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')

plt.tight_layout()
plt.show()
```
