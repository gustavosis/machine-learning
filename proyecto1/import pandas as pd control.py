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
21
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

def menu():
    print("Menú de Análisis Exploratorio de Datos")
    print("1. Cargar datos")
    print("2. Mostrar primeras filas")
    print("3. Descripción estadística")
    print("4. Visualizar distribución")
    print("5. Matriz de correlación")
    print("6. Salir")

if __name__ == "__main__":
    df = None
    ruta = 'C:\\Users\\Isi25\\OneDrive\\Documentos\\machine learning\\dataset.csv'
    
    while True:
        menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == '1':
            df = cargar_datos(ruta)
            print("Datos cargados correctamente.")
        elif opcion == '2':
            if df is not None:
                mostrar_primeras_filas(df)
            else:
                print("Primero debes cargar los datos.")
        elif opcion == '3':
            if df is not None:
                descripcion_estadistica(df)
            else:
                print("Primero debes cargar los datos.")
        elif opcion == '4':
            if df is not None:
                columna = input("Introduce el nombre de la columna: ")
                visualizar_distribucion(df, columna)
            else:
                print("Primero debes cargar los datos.")
        elif opcion == '5':
            if df is not None:
                matriz_correlacion(df)
            else:
                print("Primero debes cargar los datos.")
        elif opcion == '6':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
