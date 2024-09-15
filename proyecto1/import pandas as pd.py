import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cargar_datos(ruta):
    return pd.read_csv(ruta)

def mostrar_primeras_filas(df):
    print(df.head())

def descripcion_estadistica(df):
    print(df.describe())

def visualizar_distribucion(df, columna):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], kde=True)
    plt.title('Distribución de la ' + columna)
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()

def matriz_correlacion(df):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()

if __name__ == "__main__":
    # Imprimir el directorio de trabajo actual
    print("Directorio de trabajo actual:", os.getcwd())
    
    # Usar una ruta absoluta
    ruta = 'C:\\Users\\Isi25\\OneDrive\\Documentos\\machine learning\\dataset.csv'
    df = cargar_datos(ruta)
    mostrar_primeras_filas(df)
    descripcion_estadistica(df)
    visualizar_distribucion(df, 'variable_de_interes')
    matriz_correlacion(df)
