
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

ARCHIVO_FRANJAS = (
    BASE_PROYECTO
    / "src"
    / "resultados_evaluacion_regiones"
    / "evaluacion_regiones.csv"
)

CARPETA_RESULTADOS = (
    BASE_PROYECTO
    / "src"
    / "resultados_comparacion_regiones"
)

CARPETA_IMAGENES = CARPETA_RESULTADOS / "visualizaciones"

CARPETA_IMAGENES.mkdir(
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

# Región horizontal utilizada para el análisis.
X_INICIO = 20
X_FIN = 224

# Intervalos candidatos.
ALTURA_REGION = 40
PASO_VERTICAL = 5

Y_INICIO = 20
Y_FIN = 224

# Número de regiones que se mostrarán por imagen.
NUM_CANDIDATAS = 3


# ============================================================
# 2. CALCULAR UMBRALES DE INTENSIDAD
# ============================================================

def calcular_umbrales(imagen_hsv, x1, x2):

    canal_v = imagen_hsv[:, :, 2]

    roi = canal_v[
        Y_INICIO:Y_FIN,
        x1:x2
    ]

    umbral_bajo = np.percentile(roi, 5)
    umbral_alto = np.percentile(roi, 95)

    return umbral_bajo, umbral_alto


# ============================================================
# 3. EVALUAR UNA REGIÓN RECTANGULAR
# ============================================================

def evaluar_region(
    imagen,
    imagen_hsv,
    inicio,
    fin,
    x1,
    x2,
    umbral_bajo,
    umbral_alto
):

    region = imagen[
        inicio:fin,
        x1:x2
    ]

    region_hsv = imagen_hsv[
        inicio:fin,
        x1:x2
    ]

    gris = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    canal_v = region_hsv[:, :, 2]

    # Intensidad media y desviación estándar.
    v_media = float(np.mean(canal_v))
    v_std = float(np.std(canal_v))

    # Variación espacial de alta frecuencia.
    laplaciano = cv2.Laplacian(
        gris,
        cv2.CV_64F
    )

    var_laplaciano = float(
        np.var(laplaciano)
    )

    # Porcentaje de píxeles extremos.
    oscuros_pct = float(
        np.mean(canal_v <= umbral_bajo) * 100
    )

    brillantes_pct = float(
        np.mean(canal_v >= umbral_alto) * 100
    )

    return {
        "fila_inicio": inicio,
        "fila_fin": fin,
        "fila_centro": (inicio + fin) / 2,
        "V_media": v_media,
        "V_std": v_std,
        "var_laplaciano": var_laplaciano,
        "pixeles_oscuros_pct": oscuros_pct,
        "pixeles_brillantes_pct": brillantes_pct
    }


# ============================================================
# 4. GENERAR TODAS LAS REGIONES CANDIDATAS
# ============================================================

def generar_regiones(imagen, nivel, nombre):

    alto, ancho = imagen.shape[:2]

    x1 = min(max(X_INICIO, 0), ancho)
    x2 = min(X_FIN, ancho)

    y1 = min(Y_INICIO, alto)
    y2 = min(Y_FIN, alto)

    if x2 <= x1 or y2 - y1 < ALTURA_REGION:
        print(f"Dimensiones insuficientes: {nombre}")
        return []

    imagen_hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    umbral_bajo, umbral_alto = calcular_umbrales(
        imagen_hsv,
        x1,
        x2
    )

    resultados = []

    for inicio in range(
        y1,
        y2 - ALTURA_REGION + 1,
        PASO_VERTICAL
    ):

        fin = inicio + ALTURA_REGION

        datos = evaluar_region(
            imagen,
            imagen_hsv,
            inicio,
            fin,
            x1,
            x2,
            umbral_bajo,
            umbral_alto
        )

        datos["nivel_ntu"] = float(nivel)
        datos["archivo"] = nombre

        datos["umbral_V_bajo"] = float(umbral_bajo)
        datos["umbral_V_alto"] = float(umbral_alto)

        resultados.append(datos)

    return resultados


# ============================================================
# 5. CALCULAR PUNTUACIÓN EXPLORATORIA
# ============================================================

def puntuar_regiones(df):

    df = df.copy()

    # El cálculo se realiza por imagen.
    # Los valores bajos representan menor variabilidad
    # o menor presencia de píxeles extremos.

    grupos = []

    for (nivel, archivo), grupo in df.groupby(
        ["nivel_ntu", "archivo"]
    ):

        grupo = grupo.copy()

        grupo["rango_V_std"] = (
            grupo["V_std"].rank(pct=True)
        )

        grupo["rango_laplaciano"] = (
            grupo["var_laplaciano"].rank(pct=True)
        )

        grupo["rango_oscuros"] = (
            grupo["pixeles_oscuros_pct"].rank(pct=True)
        )

        grupo["rango_brillantes"] = (
            grupo["pixeles_brillantes_pct"].rank(pct=True)
        )

        # Puntuación exploratoria.
        # Cuanto menor sea, mayor homogeneidad relativa.
        grupo["puntuacion"] = (
            0.35 * grupo["rango_V_std"]
            + 0.30 * grupo["rango_laplaciano"]
            + 0.175 * grupo["rango_oscuros"]
            + 0.175 * grupo["rango_brillantes"]
        )

        grupo = grupo.sort_values(
            "puntuacion",
            ascending=True
        )

        grupo["posicion"] = range(
            1,
            len(grupo) + 1
        )

        grupos.append(grupo)

    return pd.concat(
        grupos,
        ignore_index=True
    )


# ============================================================
# 6. CREAR VISUALIZACIÓN DE LAS TRES CANDIDATAS
# ============================================================

def crear_visualizacion(
    imagen,
    candidatas,
    nombre_salida
):

    visualizacion = imagen.copy()

    colores = [
        (0, 255, 0),       # Primera candidata: verde
        (0, 165, 255),     # Segunda candidata: naranja
        (255, 0, 255)      # Tercera candidata: magenta
    ]

    for _, candidata in candidatas.iterrows():

        posicion = int(candidata["posicion"])

        if posicion > NUM_CANDIDATAS:
            continue

        color = colores[posicion - 1]

        inicio = int(candidata["fila_inicio"])
        fin = int(candidata["fila_fin"])

        cv2.rectangle(
            visualizacion,
            (X_INICIO, inicio),
            (X_FIN, fin),
            color,
            2
        )

        texto = (
            f"#{posicion} "
            f"{inicio}-{fin} "
            f"S={candidata['puntuacion']:.3f}"
        )

        cv2.putText(
            visualizacion,
            texto,
            (X_INICIO + 3, max(15, inicio - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA
        )

    ruta_salida = (
        CARPETA_IMAGENES
        / nombre_salida
    )

    cv2.imwrite(
        str(ruta_salida),
        visualizacion
    )

    return ruta_salida


# ============================================================
# 7. PROCESAR UNA IMAGEN
# ============================================================

def procesar_imagen(nivel, nombre):

    ruta = BASE_DATASET / nivel / nombre

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"No se pudo abrir: {ruta}")
        return []

    resultados = generar_regiones(
        imagen,
        nivel,
        nombre
    )

    return resultados


# ============================================================
# 8. MOSTRAR LAS TRES MEJORES CANDIDATAS
# ============================================================

def mostrar_resultados(df):

    for (nivel, nombre), grupo in df.groupby(
        ["nivel_ntu", "archivo"]
    ):

        mejores = grupo.nsmallest(
            NUM_CANDIDATAS,
            "puntuacion"
        )

        print("\n" + "=" * 70)
        print(f"IMAGEN: {nombre} | NIVEL: {nivel} NTU")
        print("=" * 70)

        for _, r in mejores.iterrows():

            print(
                f"Candidata {int(r['posicion'])}: "
                f"filas {int(r['fila_inicio'])}-"
                f"{int(r['fila_fin'])} | "
                f"Puntuación: {r['puntuacion']:.4f}"
            )

            print(
                f"  V media: {r['V_media']:.2f} | "
                f"V std: {r['V_std']:.2f} | "
                f"Laplaciano: {r['var_laplaciano']:.2f}"
            )

            print(
                f"  Oscuros: {r['pixeles_oscuros_pct']:.2f}% | "
                f"Brillantes: {r['pixeles_brillantes_pct']:.2f}%"
            )


# ============================================================
# 9. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    if not BASE_DATASET.exists():

        print("No se encontró la carpeta Formazine:")
        print(BASE_DATASET)
        return

    if not ARCHIVO_FRANJAS.exists():

        print("No se encontró el CSV de evaluación:")
        print(ARCHIVO_FRANJAS)
        return

    # Comprobar que el CSV anterior tiene la estructura esperada.
    df_franjas = pd.read_csv(
        ARCHIVO_FRANJAS,
        sep=";"
    )

    columnas_necesarias = {
        "nivel_ntu",
        "archivo",
        "fila_inicio",
        "fila_fin",
        "V_media",
        "V_std",
        "var_laplaciano"
    }

    if not columnas_necesarias.issubset(
        df_franjas.columns
    ):

        print("El CSV no contiene todas las columnas necesarias.")
        return

    todos_resultados = []

    for nivel, nombre in IMAGENES:

        print(f"\nProcesando {nombre}...")

        resultados = procesar_imagen(
            nivel,
            nombre
        )

        todos_resultados.extend(resultados)

    if not todos_resultados:

        print("No se generaron regiones candidatas.")
        return

    df = pd.DataFrame(todos_resultados)

    # Calcular puntuación y posición relativa.
    df = puntuar_regiones(df)

    # Guardar todas las regiones candidatas.
    ruta_csv = (
        CARPETA_RESULTADOS
        / "comparacion_regiones.csv"
    )

    df.to_csv(
        ruta_csv,
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    # Crear visualizaciones.
    for (nivel, nombre), grupo in df.groupby(
        ["nivel_ntu", "archivo"]
    ):

        imagen = cv2.imread(
            str(BASE_DATASET / str(nivel) / nombre)
        )

        if imagen is None:
            continue

        mejores = grupo.nsmallest(
            NUM_CANDIDATAS,
            "puntuacion"
        )

        crear_visualizacion(
    imagen,
    mejores,
    f"{str(nivel).replace('.', '_')}NTU_{nombre}"
        )

    # Mostrar resultados.
    mostrar_resultados(df)

    print("\n" + "=" * 70)
    print("COMPARACIÓN TERMINADA")
    print(f"Regiones evaluadas: {len(df)}")
    print(f"CSV generado: {ruta_csv}")
    print(f"Imágenes generadas: {CARPETA_IMAGENES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
