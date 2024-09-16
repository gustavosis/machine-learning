import numpy as np

def main():
    np.set_printoptions(precision=1)

    # Matriz de características
    x = np.array(
        [
            [25, 2, 50, 1, 500], 
            [39, 3, 10, 1, 1000], 
            [13, 2, 13, 1, 1000], 
            [82, 5, 20, 2, 120], 
            [130, 6, 10, 2, 600],
            [115, 6, 10, 1, 550]
        ]
    )   

    # Vector de valores objetivo
    y = np.array([127900, 222100, 143750, 268000, 460700, 407000])

    # Calcular los coeficientes de la regresión lineal
    c = np.linalg.lstsq(x, y, rcond=None)[0]
    print("Coeficientes:", c)

    # Predecir los valores de y usando los coeficientes
    y_pred = x @ c
    print("Predicciones:", y_pred)

main()
