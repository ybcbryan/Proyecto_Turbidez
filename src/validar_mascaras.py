from pathlib import Path
import cv2
import numpy as np

# ==========================================
# 1. RUTA DEL DATASET
# ==========================================
base = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"
# ==========================================
# 2. CONFIGURACIÓN
# ==========================================

niveles = ["0", "0.5", "1", "2.5", "4",
           "5", "7.5", "10", "40"]

cantidad_por_nivel = 5
margen = 8

# ==========================================
# 3. DETECTOR DE FRONTERA
# ==========================================

def detectar_frontera(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    perfil = np.mean(gris, axis=1).astype(np.float32)
    perfil_columna = perfil.reshape(-1, 1)

    perfil_suave = cv2.GaussianBlur(
        perfil_columna,
        (1, 9),
        0
    ).flatten()

    inicio = 25
    fin = min(180, len(perfil_suave) - 1)

    diferencias = np.diff(perfil_suave[inicio:fin])
    posicion = np.argmin(diferencias)

    return inicio + posicion

# ==========================================
# 4. AJUSTAR IMÁGENES PARA VISUALIZAR
# ==========================================

def ajustar_a_pantalla(imagen, max_ancho=1200, max_alto=650):
    alto, ancho = imagen.shape[:2]

    escala = min(
        max_ancho / ancho,
        max_alto / alto,
        1.0
    )

    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))

    return cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )

# ==========================================
# 5. COMPROBAR LA RUTA
# ==========================================

if not base.exists():
    print("ERROR: No se encontró la carpeta Formazine.")
    print(base)
    raise SystemExit

# ==========================================
# 6. RECORRER LOS NIVELES
# ==========================================

for nivel in niveles:

    carpeta = base / nivel

    if not carpeta.exists():
        print(f"No existe la carpeta: {carpeta}")
        continue

    archivos = sorted(
        carpeta.glob("*.jpg"),
        key=lambda archivo: int(archivo.stem)
        if archivo.stem.isdigit() else -1
    )

    if len(archivos) == 0:
        print(f"No hay imágenes JPG en {carpeta}")
        continue

    # Elegir imágenes repartidas a lo largo de la carpeta
    indices = np.linspace(
        0,
        len(archivos) - 1,
        min(cantidad_por_nivel, len(archivos)),
        dtype=int
    )

    seleccionadas = [archivos[i] for i in indices]

    print("\n===================================")
    print(f"VALIDANDO NIVEL: {nivel} NTU")
    print(f"Imágenes en carpeta: {len(archivos)}")
    print("===================================")

    # ==========================================
    # 7. PROCESAR CADA IMAGEN SELECCIONADA
    # ==========================================

    for archivo in seleccionadas:

        imagen = cv2.imread(str(archivo))

        if imagen is None:
            print(f"No se pudo leer: {archivo.name}")
            continue

        # Corregir orientación por paridad del nombre
        numero = int(archivo.stem)

        if numero % 2 == 1:
            imagen = cv2.flip(imagen, 0)

        alto, ancho = imagen.shape[:2]

        # Detectar frontera y definir máscara
        frontera = detectar_frontera(imagen)
        limite = max(1, frontera - margen)

        mascara = np.zeros((alto, ancho), dtype=np.uint8)
        mascara[:limite, :] = 255

        # Crear imagen de superposición
        superpuesta = imagen.copy()
        capa_verde = np.zeros_like(imagen)
        capa_verde[:] = (0, 255, 0)

        region = mascara == 255

        superpuesta[region] = cv2.addWeighted(
            imagen[region],
            0.65,
            capa_verde[region],
            0.35,
            0
        )

        # Dibujar frontera amarilla y límite azul
        cv2.line(
            superpuesta,
            (0, frontera),
            (ancho - 1, frontera),
            (0, 255, 255),
            2
        )

        cv2.line(
            superpuesta,
            (0, limite),
            (ancho - 1, limite),
            (255, 0, 0),
            2
        )

        mascara_color = cv2.cvtColor(
            mascara,
            cv2.COLOR_GRAY2BGR
        )

        # Etiquetas
        cv2.putText(
            imagen,
            f"{nivel} NTU - {archivo.name}",
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

        comparacion = np.hstack((
            imagen,
            mascara_color,
            superpuesta
        ))

        porcentaje = (
            cv2.countNonZero(mascara) / (alto * ancho)
        ) * 100

        print(
            f"{nivel} NTU | {archivo.name} | "
            f"frontera: {frontera} | "
            f"limite: {limite} | "
            f"área: {porcentaje:.2f}%"
        )

        comparacion = ajustar_a_pantalla(comparacion)

        ventana = f"Validacion - {nivel} NTU - {archivo.name}"

        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.imshow(ventana, comparacion)

        print("Pulsa una tecla para ver la siguiente imagen.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("\nValidación terminada.")