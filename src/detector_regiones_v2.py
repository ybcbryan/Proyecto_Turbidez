
from pathlib import Path
import cv2
import numpy as np

# ============================================================
# 1. RUTAS Y CONFIGURACIÓN
# ============================================================

BASE_PROYECTO = Path(__file__).resolve().parents[1]

BASE_DATASET = (
    BASE_PROYECTO
    / "WaTur-Water-Turbidity-Dataset-main"
    / "Formazine"
)

# Las mismas diez imágenes que ya revisamos.
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

FILA_INICIO = 25
FILA_FIN = 180

# Número máximo de transiciones que mostraremos
MAX_TRANSICIONES = 6

# Separación mínima entre transiciones candidatas
SEPARACION_CANDIDATAS = 8

# Tamaño de las ventanas
ANCHO_PANEL = 400
ALTO_PANEL = 400


# ============================================================
# 2. CALCULAR EL PERFIL DE INTENSIDAD
# ============================================================

def calcular_perfil(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    perfil = np.mean(
        gris,
        axis=1
    ).astype(np.float32)

    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1, 9),
        0
    ).flatten()

    return perfil_suave


# ============================================================
# 3. DETECTAR TRANSICIONES CANDIDATAS
# ============================================================

def obtener_candidatas(perfil):

    inicio = FILA_INICIO
    fin = min(FILA_FIN, len(perfil) - 1)

    diferencias = np.diff(
        perfil[inicio:fin]
    )

    positivas = []
    negativas = []

    for i, cambio in enumerate(diferencias):

        fila = inicio + i + 1

        if cambio > 0:
            positivas.append(
                (fila, float(cambio))
            )

        elif cambio < 0:
            negativas.append(
                (fila, float(cambio))
            )

    def seleccionar_picos(lista):

        # Ordenar por magnitud del cambio
        lista = sorted(
            lista,
            key=lambda elemento: abs(elemento[1]),
            reverse=True
        )

        seleccionadas = []

        for fila, cambio in lista:

            demasiado_cerca = any(
                abs(fila - fila_existente)
                < SEPARACION_CANDIDATAS
                for fila_existente, _ in seleccionadas
            )

            if not demasiado_cerca:
                seleccionadas.append(
                    (fila, cambio)
                )

            if len(seleccionadas) >= MAX_TRANSICIONES:
                break

        return sorted(
            seleccionadas,
            key=lambda elemento: elemento[0]
        )

    positivas = seleccionar_picos(positivas)
    negativas = seleccionar_picos(negativas)

    return positivas, negativas


# ============================================================
# 4. DIBUJAR LAS TRANSICIONES SOBRE LA IMAGEN
# ============================================================

def dibujar_transiciones(imagen, positivas, negativas):

    resultado = imagen.copy()

    # Subidas: rojo
    for fila, cambio in positivas:

        cv2.line(
            resultado,
            (0, fila),
            (resultado.shape[1] - 1, fila),
            (0, 0, 255),
            2
        )

        cv2.putText(
            resultado,
            f"+ {fila}",
            (5, max(15, fila - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    # Caídas: azul
    for fila, cambio in negativas:

        cv2.line(
            resultado,
            (0, fila),
            (resultado.shape[1] - 1, fila),
            (255, 0, 0),
            2
        )

        cv2.putText(
            resultado,
            f"- {fila}",
            (resultado.shape[1] - 55, max(15, fila - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    return resultado


# ============================================================
# 5. DIBUJAR EL PERFIL DE INTENSIDAD
# ============================================================

def dibujar_perfil(perfil, positivas, negativas):

    lienzo = np.zeros(
        (ALTO_PANEL, ANCHO_PANEL, 3),
        dtype=np.uint8
    )

    inicio = FILA_INICIO
    fin = min(FILA_FIN, len(perfil) - 1)

    valores = perfil[inicio:fin]

    minimo = float(np.min(valores))
    maximo = float(np.max(valores))

    if maximo == minimo:
        maximo += 1

    # Ejes
    cv2.line(
        lienzo,
        (40, 20),
        (40, ALTO_PANEL - 30),
        (180, 180, 180),
        1
    )

    cv2.line(
        lienzo,
        (40, ALTO_PANEL - 30),
        (ANCHO_PANEL - 10, ALTO_PANEL - 30),
        (180, 180, 180),
        1
    )

    puntos = []

    for i, valor in enumerate(valores):

        x = 40 + int(
            i * (ANCHO_PANEL - 60)
            / max(1, len(valores) - 1)
        )

        y = ALTO_PANEL - 30 - int(
            (valor - minimo)
            / (maximo - minimo)
            * (ALTO_PANEL - 60)
        )

        puntos.append((x, y))

    for i in range(len(puntos) - 1):

        cv2.line(
            lienzo,
            puntos[i],
            puntos[i + 1],
            (0, 255, 255),
            2
        )

    # Marcar transiciones positivas
    for fila, cambio in positivas:

        x = 40 + int(
            (fila - inicio)
            * (ANCHO_PANEL - 60)
            / max(1, fin - inicio - 1)
        )

        cv2.line(
            lienzo,
            (x, 20),
            (x, ALTO_PANEL - 30),
            (0, 0, 255),
            1
        )

    # Marcar transiciones negativas
    for fila, cambio in negativas:

        x = 40 + int(
            (fila - inicio)
            * (ANCHO_PANEL - 60)
            / max(1, fin - inicio - 1)
        )

        cv2.line(
            lienzo,
            (x, 20),
            (x, ALTO_PANEL - 30),
            (255, 0, 0),
            1
        )

    cv2.putText(
        lienzo,
        "PERFIL DE INTENSIDAD",
        (45, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        lienzo,
        f"Min: {minimo:.1f}  Max: {maximo:.1f}",
        (45, ALTO_PANEL - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    return lienzo


# ============================================================
# 6. PROCESAR CADA IMAGEN
# ============================================================

def revisar_imagen(nivel, nombre):

    ruta = BASE_DATASET / nivel / nombre

    imagen = cv2.imread(str(ruta))

    if imagen is None:

        print("No se pudo abrir:", ruta)
        return

    perfil = calcular_perfil(imagen)

    positivas, negativas = obtener_candidatas(perfil)

    imagen_fronteras = dibujar_transiciones(
        imagen,
        positivas,
        negativas
    )

    grafico = dibujar_perfil(
        perfil,
        positivas,
        negativas
    )

    imagen_reducida = cv2.resize(
        imagen_fronteras,
        (ANCHO_PANEL, ALTO_PANEL)
    )

    comparacion = np.hstack([
        imagen_reducida,
        grafico
    ])

    print("\n" + "=" * 55)
    print(f"Nivel: {nivel} NTU")
    print(f"Imagen: {nombre}")

    print("\nTRANSICIONES POSITIVAS:")

    for fila, cambio in positivas:
        print(
            f"Fila {fila}: cambio de intensidad +{cambio:.3f}"
        )

    print("\nTRANSICIONES NEGATIVAS:")

    for fila, cambio in negativas:
        print(
            f"Fila {fila}: cambio de intensidad {cambio:.3f}"
        )

    print("\nRojo = subida | Azul = caída")

    cv2.imshow(
        f"{nivel} NTU - {nombre}",
        comparacion
    )

    print("\nPulsa cualquier tecla para continuar.")
    print("Pulsa ESC para terminar.")

    tecla = cv2.waitKey(0)

    cv2.destroyAllWindows()

    return tecla


# ============================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    if not BASE_DATASET.exists():

        print("No se encontró la carpeta Formazine.")
        print(BASE_DATASET)
        return

    for nivel, nombre in IMAGENES:

        tecla = revisar_imagen(
            nivel,
            nombre
        )

        if tecla == 27:
            break

    cv2.destroyAllWindows()

    print("\nANÁLISIS DE TRANSICIONES TERMINADO.")


if __name__ == "__main__":
    main()
