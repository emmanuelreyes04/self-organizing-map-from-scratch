import math
import random
import matplotlib.pyplot as plt

DATOS_ENTRADA = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.5, 0.5, 0.5]
]
DIM_ENTRADA = len(DATOS_ENTRADA[0])

FILAS_SOM = 2
COLUMNAS_SOM = 2
EPOCAS = 1000
ALPHA_INICIAL = 0.6
RADIO_INICIAL = 0
LAMBDA_ALPHA = 0
LAMBDA_RADIO = 0

def distancia_euclidiana(v1, v2):
    suma_cuadrados = sum([(a - b)**2 for a, b in zip(v1, v2)])
    return math.sqrt(suma_cuadrados)

def encontrar_bmu(vector_entrada, som_pesos):
    bmu_distancia = float('inf')
    bmu_coordenadas = (0, 0)

    for r in range(FILAS_SOM):
        for c in range(COLUMNAS_SOM):
            pesos_neurona = som_pesos[r][c]
            distancia = distancia_euclidiana(vector_entrada, pesos_neurona)

            if distancia < bmu_distancia:
                bmu_distancia = distancia
                bmu_coordenadas = (r, c)

    return bmu_coordenadas

def funcion_vecindad(distancia_neurona, radio_t):
    if radio_t == 0:
        return 0.0 if distancia_neurona > 0 else 1.0
    return math.exp(-(distancia_neurona**2) / (2 * radio_t**2))

def actualizar_parametros(t):
    alpha_t = ALPHA_INICIAL * math.exp(-t / LAMBDA_ALPHA)
    radio_t = RADIO_INICIAL * math.exp(-t / LAMBDA_RADIO)
    return alpha_t, radio_t

def inicializar_pesos(filas, columnas, dim_entrada):
    som_pesos = []
    for _ in range(filas):
        fila_pesos = []
        for _ in range(columnas):
            pesos = [random.random() for _ in range(dim_entrada)]
            fila_pesos.append(pesos)
        som_pesos.append(fila_pesos)
    return som_pesos

def entrenar_som(som_pesos, datos):

    for t in range(EPOCAS):
        alpha_t, radio_t = actualizar_parametros(t)
        vector_entrada = random.choice(datos)
        bmu_r, bmu_c = encontrar_bmu(vector_entrada, som_pesos)

        for r in range(FILAS_SOM):
            for c in range(COLUMNAS_SOM):
                dist_neurona_bmu = distancia_euclidiana((r, c), (bmu_r, bmu_c))
                h_rc = funcion_vecindad(dist_neurona_bmu, radio_t)

                for j in range(DIM_ENTRADA):
                    diferencia = vector_entrada[j] - som_pesos[r][c][j]
                    ajuste = alpha_t * h_rc * diferencia
                    som_pesos[r][c][j] += ajuste

    return som_pesos

def graficar_som(som_pesos, titulo):
    plt.figure(figsize=(COLUMNAS_SOM, FILAS_SOM))
    plt.imshow(som_pesos, interpolation='nearest')
    plt.title(titulo)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show()

def ejecutar_som_y_probar_tamanos():
    global FILAS_SOM, COLUMNAS_SOM, RADIO_INICIAL, LAMBDA_ALPHA, LAMBDA_RADIO, EPOCAS

    FILAS_SOM, COLUMNAS_SOM = 2, 2
    EPOCAS = 1000
    RADIO_INICIAL = max(FILAS_SOM, COLUMNAS_SOM) / 2
    LAMBDA_ALPHA = EPOCAS / 2
    LAMBDA_RADIO = EPOCAS / 2

    print("\n" + "="*50)
    print(f"PRUEBA 1: SOM de {FILAS_SOM}x{COLUMNAS_SOM}")
    print("="*50)

    pesos_iniciales_2x2 = inicializar_pesos(FILAS_SOM, COLUMNAS_SOM, DIM_ENTRADA)
    graficar_som(pesos_iniciales_2x2, f"Pesos Iniciales SOM {FILAS_SOM}x{COLUMNAS_SOM}")

    print("Entrenamiento en curso...")
    pesos_finales_2x2 = entrenar_som(pesos_iniciales_2x2, DATOS_ENTRADA)

    print("Entrenamiento finalizado.")
    graficar_som(pesos_finales_2x2, f"Pesos Finales SOM {FILAS_SOM}x{COLUMNAS_SOM} (Mapa de Colores)")

    FILAS_SOM, COLUMNAS_SOM = 4, 4
    EPOCAS = 2000
    RADIO_INICIAL = max(FILAS_SOM, COLUMNAS_SOM) / 2
    LAMBDA_ALPHA = EPOCAS / 2
    LAMBDA_RADIO = EPOCAS / 2

    print("\n" + "="*50)
    print(f"PRUEBA 2: SOM de {FILAS_SOM}x{COLUMNAS_SOM}")
    print("="*50)

    pesos_iniciales_4x4 = inicializar_pesos(FILAS_SOM, COLUMNAS_SOM, DIM_ENTRADA)
    graficar_som(pesos_iniciales_4x4, f"Pesos Iniciales SOM {FILAS_SOM}x{COLUMNAS_SOM}")

    print("Entrenamiento en curso...")
    pesos_finales_4x4 = entrenar_som(pesos_iniciales_4x4, DATOS_ENTRADA)

    print("Entrenamiento finalizado.")
    graficar_som(pesos_finales_4x4, f"Pesos Finales SOM {FILAS_SOM}x{COLUMNAS_SOM} (Mapa de Colores)")

ejecutar_som_y_probar_tamanos()