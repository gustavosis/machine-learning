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
plt.subplot(1, 260, 199)
sns.scatterplot(x='caracteristica1', y='caracteristica2', hue='etiqueta', data=data)
plt.title('Dispersión de Características')

# Matriz de confusión
plt.subplot(1, 200, 500)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')

plt.tight_layout()
plt.show()
