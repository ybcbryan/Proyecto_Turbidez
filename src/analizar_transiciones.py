from pathlib import Path
import cv2
import numpy as np

# ==========================================
# 1. RUTA DEL DATASET
# ==========================================

base = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"

# ==========================================
# 2. IMÁGENES QUE VAMOS A ANALIZAR
# ==========================================

pruebas = [
    ("0.5", "326.jpg"),
    ("0.5", "652.jpg"),
    ("0.5", "978.jpg"),
    ("0.5", "1304.jpg"),
    ("1", "591.jpg")
]

# ==========================================
# 3. CORREGIR ORIENTACIÓN
# ==========================================

def corregir_orientacion(imagen, nombre_archivo):
    numero = int(Path(nombre_archivo).stem)

    if numero % 2 == 1:
        imagen = cv2.flip(imagen, 0)

    return imagen

# ==========================================
# 4. CALCULAR PERFIL DE INTENSIDAD
# ==========================================

def calcular_perfil(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Intensidad promedio de cada fila
    perfil = np.mean(gris, axis=1)

    return perfil

# ==========================================
# 5. DIBUJAR GRÁFICAS CON OPENCV
# ==========================================

def dibujar_grafica(perfil, titulo, ancho=800, alto=300):
    margen_izq = 55
    margen_der = 20
    margen_sup = 35
    margen_inf = 35

    grafica = np.full(
        (alto, ancho, 3),
        255,
        dtype=np.uint8
    )

    x0 = margen_izq
    y0 = alto - margen_inf
    x1 = ancho - margen_der
    y1 = margen_sup

    # Título
    cv2.putText(
        grafica,
        titulo,
        (margen_izq, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1
    )

    # Ejes
    cv2.line(grafica, (x0, y1), (x0, y0), (0, 0, 0), 1)
    cv2.line(grafica, (x0, y0), (x1, y0), (0, 0, 0), 1)

    # Escala vertical
    minimo = float(np.min(perfil))
    maximo = float(np.max(perfil))

    if maximo == minimo:
        maximo = minimo + 1

    cv2.putText(
        grafica,
        f"{maximo:.1f}",
        (3, y1 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1
    )

    cv2.putText(
        grafica,
        f"{minimo:.1f}",
        (3, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1
    )

    # Convertir valores del perfil a coordenadas
    puntos = []

    for i, valor in enumerate(perfil):
        x = x0 + int(i * (x1 - x0) / max(1, len(perfil) - 1))

        y = y0 - int(
            (valor - minimo) * (y0 - y1) / (maximo - minimo)
        )

        puntos.append((x, y))

    # Dibujar curva
    for i in range(len(puntos) - 1):
        cv2.line(
            grafica,
            puntos[i],
            puntos[i + 1],
            (255, 0, 0),
            2
        )

    # Etiquetas de eje X
    cv2.putText(
        grafica,
        "Fila 0",
        (x0 - 10, alto - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1
    )

    cv2.putText(
        grafica,
        f"Fila {len(perfil) - 1}",
        (x1 - 65, alto - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1
    )

    return grafica

# ==========================================
# 6. BUSCAR TRANSICIONES
# ==========================================

def obtener_transiciones(perfil, cantidad=8):
    # Suavizar el perfil para reducir fluctuaciones pequeñas
    perfil_suave = cv2.GaussianBlur(
        perfil.astype(np.float32).reshape(-1, 1),
        (1, 9),
        0
    ).flatten()

    # Diferencia entre filas consecutivas
    diferencias = np.diff(perfil_suave)

    # Transiciones más fuertes hacia menor intensidad
    caidas = np.argsort(diferencias)[:cantidad]

    # Transiciones más fuertes hacia mayor intensidad
    subidas = np.argsort(diferencias)[-cantidad:][::-1]

    return perfil_suave, diferencias, caidas, subidas

# ==========================================
# 7. MOSTRAR IMÁGENES Y RESULTADOS
# ==========================================

if not base.exists():
    print("ERROR: No se encontró la carpeta Formazine.")
    print("Ruta revisada:")
    print(base)
    raise SystemExit

for nivel, nombre in pruebas:

    ruta = base / nivel / nombre

    if not ruta.exists():
        print(f"No se encontró: {ruta}")
        continue

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"No se pudo leer: {ruta}")
        continue

    imagen = corregir_orientacion(imagen, nombre)

    alto, ancho = imagen.shape[:2]

    perfil = calcular_perfil(imagen)

    perfil_suave, diferencias, caidas, subidas = (
        obtener_transiciones(perfil)
    )

    # ======================================
    # 8. MARCAR LAS TRANSICIONES EN LA FOTO
    # ======================================

    imagen_marcada = imagen.copy()

    # Mostrar las caídas más pronunciadas en rojo
    for indice in caidas[:4]:
        fila = int(indice + 1)

        cv2.line(
            imagen_marcada,
            (0, fila),
            (ancho - 1, fila),
            (0, 0, 255),
            1
        )

    # Mostrar las subidas más pronunciadas en verde
    for indice in subidas[:4]:
        fila = int(indice + 1)

        cv2.line(
            imagen_marcada,
            (0, fila),
            (ancho - 1, fila),
            (0, 255, 0),
            1
        )

    cv2.putText(
        imagen_marcada,
        f"{nivel} NTU - {nombre}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # ======================================
    # 9. GRÁFICA DEL PERFIL
    # ======================================

    grafica = dibujar_grafica(
        perfil_suave,
        "Intensidad promedio por fila"
    )

    # ======================================
    # 10. GRÁFICA DE DIFERENCIAS
    # ======================================

    grafica_dif = dibujar_grafica(
        diferencias,
        "Cambio de intensidad entre filas"
    )

    # ======================================
    # 11. UNIR VISUALIZACIONES
    # ======================================

    # Ajustar imagen a una altura similar a las gráficas
    alto_grafica = 300
    ancho_imagen = int(
        imagen_marcada.shape[1] *
        alto_grafica / imagen_marcada.shape[0]
    )

    imagen_pequena = cv2.resize(
        imagen_marcada,
        (ancho_imagen, alto_grafica),
        interpolation=cv2.INTER_AREA
    )

    columna_izquierda = np.vstack((
        imagen_pequena,
        np.full((alto_grafica, ancho_imagen, 3), 255, dtype=np.uint8)
    ))

    columna_derecha = np.vstack((
        grafica,
        grafica_dif
    ))

    comparacion = np.hstack((
        columna_izquierda,
        columna_derecha
    ))

    # ======================================
    # 12. IMPRIMIR TRANSICIONES
    # ======================================

    print("\n===================================")
    print(f"Nivel: {nivel} NTU")
    print(f"Archivo: {nombre}")
    print("Orientación aplicada según paridad.")
    print("-----------------------------------")

    print("Caídas más pronunciadas:")
    for indice in caidas[:8]:
        fila = int(indice + 1)
        cambio = diferencias[indice]
        print(f"Fila {fila}: cambio {cambio:.2f}")

    print("Subidas más pronunciadas:")
    for indice in subidas[:8]:
        fila = int(indice + 1)
        cambio = diferencias[indice]
        print(f"Fila {fila}: cambio {cambio:.2f}")

    # ======================================
    # 13. MOSTRAR VENTANA
    # ======================================

    ventana = f"Analisis - {nivel} NTU - {nombre}"

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 1250, 650)
    cv2.imshow(ventana, comparacion)

    print("\nObserva la imagen y las dos gráficas.")
    print("Pulsa una tecla para continuar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nAnálisis terminado.")