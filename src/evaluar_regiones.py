
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ============================================================
# 1. RUTAS Y CONFIGURACIÓN
# ============================================================

BASE_PROYECTO = Path(__file__).resolve().parents[1]

BASE_DATASET = (
    BASE_PROYECTO
    / "WaTur-Water-Turbidity-Dataset-main"
    / "Formazine"
)

CARPETA_RESULTADOS = (
    BASE_PROYECTO
    / "src"
    / "resultados_evaluacion_regiones"
)

CARPETA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

IMAGENES = [
    ("1", "55.jpg"),
    ("5", "1266.jpg"),
    ("40", "834.jpg"),
    ("2.5", "99.jpg"),
    ("1", "1069.jpg"),
    ("1", "1013.jpg"),
    ("7.5", "389.jpg"),
    ("1", "453.jpg"),
    ("0", "1503.jpg"),
    ("4", "1383.jpg"),
]

# Región horizontal de análisis.
# Se excluyen los bordes laterales de la imagen.
X_INICIO = 20
X_FIN = 224

# Franjas horizontales superpuestas.
ALTURA_FRANJA = 20
PASO = 5

# Intervalo vertical que analizaremos.
Y_INICIO = 20
Y_FIN = 224


# ============================================================
# 2. CALCULAR LOS UMBRALES DE INTENSIDAD EXTREMA
# ============================================================

def calcular_umbrales_extremos(imagen_hsv, x1, x2):

    canal_v = imagen_hsv[:, :, 2]

    roi_v = canal_v[
        Y_INICIO:Y_FIN,
        x1:x2
    ]

    # Percentiles calculados para cada imagen.
    umbral_bajo = np.percentile(roi_v, 5)
    umbral_alto = np.percentile(roi_v, 95)

    return umbral_bajo, umbral_alto


# ============================================================
# 3. EVALUAR UNA FRANJA HORIZONTAL
# ============================================================

def evaluar_franja(
    imagen,
    imagen_hsv,
    inicio,
    fin,
    x1,
    x2,
    umbral_bajo,
    umbral_alto
):

    franja = imagen[
        inicio:fin,
        x1:x2
    ]

    franja_hsv = imagen_hsv[
        inicio:fin,
        x1:x2
    ]

    gris = cv2.cvtColor(
        franja,
        cv2.COLOR_BGR2GRAY
    )

    canal_v = franja_hsv[:, :, 2]

    # Intensidad media y desviación estándar.
    intensidad_media = float(
        np.mean(canal_v)
    )

    intensidad_std = float(
        np.std(canal_v)
    )

    # Varianza del Laplaciano.
    laplaciano = cv2.Laplacian(
        gris,
        cv2.CV_64F
    )

    varianza_laplaciano = float(
        np.var(laplaciano)
    )

    # Proporción de píxeles en los extremos de intensidad.
    fraccion_oscuros = float(
        np.mean(canal_v <= umbral_bajo) * 100
    )

    fraccion_brillantes = float(
        np.mean(canal_v >= umbral_alto) * 100
    )

    return {
        "fila_inicio": inicio,
        "fila_fin": fin,
        "fila_centro": (inicio + fin) / 2,
        "V_media": intensidad_media,
        "V_std": intensidad_std,
        "var_laplaciano": varianza_laplaciano,
        "pixeles_oscuros_pct": fraccion_oscuros,
        "pixeles_brillantes_pct": fraccion_brillantes
    }


# ============================================================
# 4. CREAR VISUALIZACIÓN DE LAS FRANJAS
# ============================================================

def crear_visualizacion(imagen):

    alto, ancho = imagen.shape[:2]

    visualizacion = imagen.copy()

    # Rectángulo de la región horizontal analizada.
    cv2.rectangle(
        visualizacion,
        (X_INICIO, Y_INICIO),
        (X_FIN, Y_FIN),
        (0, 255, 255),
        2
    )

    # Dibujar franjas no superpuestas para facilitar
    # la interpretación visual.
    for inicio in range(
        Y_INICIO,
        Y_FIN - ALTURA_FRANJA + 1,
        ALTURA_FRANJA
    ):

        fin = inicio + ALTURA_FRANJA

        cv2.rectangle(
            visualizacion,
            (X_INICIO, inicio),
            (X_FIN, fin),
            (255, 180, 0),
            1
        )

        cv2.putText(
            visualizacion,
            str(inicio),
            (X_FIN - 30, inicio + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    # Redimensionar para mostrarla en pantalla.
    visualizacion = cv2.resize(
        visualizacion,
        (488, 488)
    )

    return visualizacion


# ============================================================
# 5. PROCESAR UNA IMAGEN
# ============================================================

def procesar_imagen(nivel, nombre):

    ruta = BASE_DATASET / nivel / nombre

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"No se pudo abrir: {ruta}")
        return []

    alto, ancho = imagen.shape[:2]

    x1 = max(0, X_INICIO)
    x2 = min(ancho, X_FIN)
    y1 = max(0, Y_INICIO)
    y2 = min(alto, Y_FIN)

    imagen_hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    umbral_bajo, umbral_alto = calcular_umbrales_extremos(
        imagen_hsv,
        x1,
        x2
    )

    resultados = []

    for inicio in range(
        y1,
        y2 - ALTURA_FRANJA + 1,
        PASO
    ):

        fin = inicio + ALTURA_FRANJA

        caracteristicas = evaluar_franja(
            imagen,
            imagen_hsv,
            inicio,
            fin,
            x1,
            x2,
            umbral_bajo,
            umbral_alto
        )

        caracteristicas["nivel_ntu"] = float(nivel)
        caracteristicas["archivo"] = nombre
        caracteristicas["umbral_V_bajo"] = float(umbral_bajo)
        caracteristicas["umbral_V_alto"] = float(umbral_alto)

        resultados.append(caracteristicas)

    # Visualización.
    visualizacion = crear_visualizacion(imagen)

    cv2.imshow(
        f"{nivel} NTU - {nombre}",
        visualizacion
    )

    print("\n" + "=" * 55)
    print(f"Nivel: {nivel} NTU")
    print(f"Imagen: {nombre}")
    print(f"Dimensiones: {ancho} x {alto}")
    print(f"Umbral V bajo (percentil 5): {umbral_bajo:.2f}")
    print(f"Umbral V alto (percentil 95): {umbral_alto:.2f}")
    print(f"Franjas evaluadas: {len(resultados)}")

    print("\nResultados por franja:")

    for r in resultados:

        print(
            f"Filas {r['fila_inicio']:3d}-{r['fila_fin']:3d} | "
            f"V media: {r['V_media']:7.2f} | "
            f"V std: {r['V_std']:7.2f} | "
            f"Laplaciano: {r['var_laplaciano']:9.2f} | "
            f"Oscuros: {r['pixeles_oscuros_pct']:6.2f}% | "
            f"Brillantes: {r['pixeles_brillantes_pct']:6.2f}%"
        )

    print("\nPulsa cualquier tecla para continuar.")
    print("Pulsa ESC para terminar.")

    tecla = cv2.waitKey(0)

    cv2.destroyAllWindows()

    if tecla == 27:
        return resultados, True

    return resultados, False


# ============================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    if not BASE_DATASET.exists():

        print("No se encontró la carpeta Formazine:")
        print(BASE_DATASET)
        return

    todos_resultados = []

    for nivel, nombre in IMAGENES:

        resultado = procesar_imagen(
            nivel,
            nombre
        )

        if resultado is None:
            continue

        filas, terminar = resultado

        todos_resultados.extend(filas)

        if terminar:
            break

    # Guardar resultados.
    if todos_resultados:

        df = pd.DataFrame(todos_resultados)

        ruta_csv = (
            CARPETA_RESULTADOS
            / "evaluacion_regiones.csv"
        )

        df.to_csv(
            ruta_csv,
            sep=";",
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 55)
        print("EVALUACIÓN TERMINADA")
        print(f"Franjas evaluadas: {len(df)}")
        print("Archivo CSV generado:")
        print(ruta_csv)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
