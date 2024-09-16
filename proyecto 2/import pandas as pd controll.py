import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import ipywidgets as widgets
from IPython.display import display, clear_output

# Crear un DataFrame vacío para almacenar los datos
data = pd.DataFrame(columns=['precio', 'cantidad_vendida', 'categoria', 'etiqueta'])

# Función para agregar datos al DataFrame
def agregar_datos(precio, cantidad_vendida, categoria, etiqueta):
    global data
    nuevo_dato = pd.DataFrame({
        'precio': [precio],
        'cantidad_vendida': [cantidad_vendida],
        'categoria': [categoria],
        'etiqueta': [etiqueta]
    })
    data = pd.concat([data, nuevo_dato], ignore_index=True)
    clear_output(wait=True)
    display(data)
    generar_graficos()

# Función para generar gráficos
def generar_graficos():
    if not data.empty:
        plt.figure(figsize=(12, 6))
        
        # Gráfico de dispersión
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='precio', y='cantidad_vendida', hue='etiqueta', data=data)
        plt.title('Dispersión de Ventas de PC')
        
        # Matriz de confusión si hay suficientes datos
        if len(data) > 1:
            X = data[['precio', 'cantidad_vendida', 'categoria']]
            X = pd.get_dummies(X, drop_first=True)
            y = data['etiqueta']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.subplot(1, 2, 2)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusión')
        
        plt.tight_layout()
        plt.show()

# Widgets para la entrada de datos
precio_input = widgets.FloatText(description='Precio:')
cantidad_vendida_input = widgets.IntText(description='Cantidad Vendida:')
categoria_input = widgets.Dropdown(options=['gaming', 'oficina'], description='Categoría:')
etiqueta_input = widgets.Dropdown(options=[0, 1], description='Etiqueta:')
agregar_button = widgets.Button(description='Agregar Datos')

# Función para manejar el evento de clic del botón
def on_agregar_button_clicked(b):
    agregar_datos(precio_input.value, cantidad_vendida_input.value, categoria_input.value, etiqueta_input.value)

agregar_button.on_click(on_agregar_button_clicked)

# Mostrar los widgets
display(precio_input, cantidad_vendida_input, categoria_input, etiqueta_input, agregar_button)

# Función para guardar el DataFrame en un archivo CSV
def guardar_csv():
    data.to_csv('ventas_pc.csv', index=False)
    print("Datos guardados en ventas_pc.csv")

# Botón para guardar los datos
guardar_button = widgets.Button(description='Guardar Datos')
guardar_button.on_click(lambda b: guardar_csv())

# Mostrar el botón de guardar
display(guardar_button)
