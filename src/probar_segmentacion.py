from pathlib import Path
import cv2
import numpy as np

# ==========================================
# 1. RUTA DEL DATASET
# ==========================================

base = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"

niveles = ["0", "0.5", "5", "10", "40"]

# ==========================================
# 2. IMÁGENES QUE VAMOS A PROBAR
# ==========================================

pruebas = [
    ("0", "0.jpg"),
    ("0.5", "0.jpg"),
    ("5", "0.jpg"),
    ("10", "1816.jpg"),
    ("40", "4.jpg")
]

# Margen entre la frontera detectada y la máscara
margen = 8

# ==========================================
# 3. AJUSTAR LA VISUALIZACIÓN
# ==========================================

def ajustar_a_pantalla(imagen, max_ancho=1200, max_alto=700):
    alto, ancho = imagen.shape[:2]

    escala = min(
        max_ancho / ancho,
        max_alto / alto
    )

    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))

    imagen_reducida = cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    )

    return imagen_reducida

# ==========================================
# 4. DETECTAR LA FRONTERA
# ==========================================

def detectar_frontera(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Promedio de intensidad de cada fila
    perfil = np.mean(gris, axis=1).astype(np.float32)

    # Suavizar el perfil para reducir pequeñas variaciones
    perfil_columna = perfil.reshape(-1, 1)

    perfil_suave = cv2.GaussianBlur(
        perfil_columna,
        (1, 9),
        0
    ).flatten()

    # Buscar el cambio más fuerte dentro de este intervalo
    inicio = 25
    fin = min(180, len(perfil_suave) - 1)

    diferencias = np.diff(perfil_suave[inicio:fin])

    posicion = np.argmin(diferencias)

    frontera = inicio + posicion

    return frontera

# ==========================================
# 5. PROCESAR Y MOSTRAR CADA IMAGEN
# ==========================================

if not base.exists():
    print("ERROR: No se encontró la carpeta del dataset.")
    print("Ruta revisada:")
    print(base)
    raise SystemExit

for nivel, nombre_archivo in pruebas:

    ruta = base / nivel / nombre_archivo

    if not ruta.exists():
        print(f"No se encontró: {ruta}")
        continue

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"No se pudo leer: {ruta}")
        continue

    # Corregir orientación según el número del archivo
    numero = int(Path(nombre_archivo).stem)

    if numero % 2 == 1:
        imagen = cv2.flip(imagen, 0)

    alto, ancho = imagen.shape[:2]

    # Detectar la frontera
    frontera = detectar_frontera(imagen)

    # Definir el límite superior de la máscara
    limite = max(1, frontera - margen)

    # Crear máscara: blanco = región que conservamos
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    mascara[:limite, :] = 255

    # Crear superposición verde en la región conservada
    superpuesta = imagen.copy()

    capa_verde = np.zeros_like(imagen)
    capa_verde[:, :] = (0, 255, 0)

    region = mascara == 255

    superpuesta[region] = cv2.addWeighted(
        imagen[region],
        0.65,
        capa_verde[region],
        0.35,
        0
    )

    # Dibujar la frontera estimada en amarillo
    cv2.line(
        superpuesta,
        (0, frontera),
        (ancho - 1, frontera),
        (0, 255, 255),
        2
    )

    # Dibujar el límite de la máscara en azul
    cv2.line(
        superpuesta,
        (0, limite),
        (ancho - 1, limite),
        (255, 0, 0),
        2
    )

    # Convertir máscara a BGR para mostrarla junto a las otras
    mascara_color = cv2.cvtColor(
        mascara,
        cv2.COLOR_GRAY2BGR
    )

    # Añadir títulos a cada panel
    cv2.putText(
        imagen,
        f"{nivel} NTU - Original",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        mascara_color,
        f"Limite: {limite}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1
    )

    cv2.putText(
        superpuesta,
        "Superposicion",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # Unir los tres paneles horizontalmente
    comparacion = np.hstack((
        imagen,
        mascara_color,
        superpuesta
    ))

    # Calcular porcentaje conservado
    pixeles_validos = cv2.countNonZero(mascara)
    total_pixeles = alto * ancho

    porcentaje = (
        pixeles_validos / total_pixeles
    ) * 100

    # Mostrar resultados
    print("-----------------------------------")
    print(f"Nivel: {nivel} NTU")
    print(f"Archivo: {nombre_archivo}")
    print(f"Frontera estimada: {frontera}")
    print(f"Limite de mascara: {limite}")
    print(f"Area conservada: {porcentaje:.2f}%")

    # Ajustar la comparación al tamaño de pantalla
    comparacion = ajustar_a_pantalla(comparacion)

    ventana = f"Prueba de mascara - {nivel} NTU"

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        ventana,
        comparacion.shape[1],
        comparacion.shape[0]
    )

    cv2.imshow(ventana, comparacion)

    print("Presiona una tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Pruebas finalizadas.")