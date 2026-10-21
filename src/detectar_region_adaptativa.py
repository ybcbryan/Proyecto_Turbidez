from pathlib import Path
import cv2
import numpy as np


# ============================================================
# 1. RUTA DEL DATASET
# ============================================================

# Conservamos la misma lógica de ruta que tu script original:
# si este archivo está dentro de una subcarpeta como src,
# parents[1] apunta a la carpeta principal del proyecto.

base = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"

# ============================================================
# 2. IMÁGENES DE PRUEBA
# ============================================================

pruebas = [
    ("0.5", "326.jpg"),
    ("1", "591.jpg"),
    ("40", "4.jpg"),
]


# ============================================================
# 3. CONFIGURACIÓN
# ============================================================

margen = 8

fila_inicio = 25
fila_fin = 180

separacion_minima = 20
umbral_transicion = 1.0

percentil_oscuro = 5
percentil_brillante = 98

tamano_kernel = 3


# ============================================================
# 4. AJUSTAR IMAGEN A LA PANTALLA
# ============================================================

def ajustar_a_pantalla(imagen, max_ancho=1500, max_alto=650):

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


# ============================================================
# 5. CALCULAR PERFIL DE INTENSIDAD
# ============================================================

def calcular_perfil(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # Intensidad promedio de cada fila
    perfil = np.mean(
        gris,
        axis=1
    ).astype(np.float32)

    # Suavizado del perfil
    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1, 9),
        0
    ).flatten()

    return perfil_suave


# ============================================================
# 6. BUSCAR FRONTERAS
# ============================================================

def buscar_fronteras(perfil):

    inicio = fila_inicio
    fin = min(
        fila_fin,
        len(perfil) - 1
    )

    perfil_zona = perfil[inicio:fin]

    # Diferencias entre filas consecutivas
    diferencias = np.diff(perfil_zona)

    # Candidatas a subida: oscuro -> claro
    indices_subidas = np.argsort(
        diferencias
    )[::-1]

    # Candidatas a caída: claro -> oscuro
    indices_caidas = np.argsort(
        diferencias
    )

    subida = None
    subida_valor = None

    caida = None
    caida_valor = None

    for indice in indices_subidas:

        valor = diferencias[indice]

        if valor <= 0:
            break

        subida = inicio + indice + 1
        subida_valor = float(valor)
        break

    for indice in indices_caidas:

        valor = diferencias[indice]

        if valor >= 0:
            break

        caida = inicio + indice + 1
        caida_valor = float(valor)
        break

    return (
        subida,
        subida_valor,
        caida,
        caida_valor
    )


# ============================================================
# 7. DETERMINAR TIPO DE REGIÓN
# ============================================================

def determinar_region(
    subida,
    subida_valor,
    caida,
    caida_valor
):

    subida_valida = (
        subida is not None
        and subida_valor is not None
        and subida_valor >= umbral_transicion
    )

    caida_valida = (
        caida is not None
        and caida_valor is not None
        and abs(caida_valor) >= umbral_transicion
    )

    # Caso 1: oscuro -> claro -> oscuro
    if (
        subida_valida
        and caida_valida
        and subida < caida
        and (caida - subida) >= separacion_minima
    ):

        return {
            "tipo": "REGION_CLARA_INTERMEDIA",
            "frontera_1": subida,
            "frontera_2": caida
        }

    # Caso 2: oscuro arriba y claro abajo
    if subida_valida and not caida_valida:

        return {
            "tipo": "OSCURO_ARRIBA_CLARO_ABAJO",
            "frontera": subida
        }

    # Caso 3: claro arriba y oscuro abajo
    if caida_valida and not subida_valida:

        return {
            "tipo": "CLARO_ARRIBA_OSCURO_ABAJO",
            "frontera": caida
        }

    # Caso 4: ambas transiciones son fuertes,
    # pero no forman una región intermedia válida.
    if subida_valida and caida_valida:

        # Elegir la transición más fuerte es una regla provisional.
        # En imágenes con transiciones próximas puede equivocarse.
        if subida_valor >= abs(caida_valor):

            return {
                "tipo": "OSCURO_ARRIBA_CLARO_ABAJO",
                "frontera": subida
            }

        return {
            "tipo": "CLARO_ARRIBA_OSCURO_ABAJO",
            "frontera": caida
        }

    # Caso 5: no se encontró una transición válida
    return {
        "tipo": "NO_DETERMINADO"
    }


# ============================================================
# 8. CREAR MÁSCARA GEOMÉTRICA
# ============================================================

def crear_mascara_geometrica(imagen, resultado):

    alto, ancho = imagen.shape[:2]

    mascara = np.zeros(
        (alto, ancho),
        dtype=np.uint8
    )

    tipo = resultado["tipo"]

    # Caso A: oscuro arriba -> claro abajo
    if tipo == "OSCURO_ARRIBA_CLARO_ABAJO":

        frontera = resultado["frontera"]

        limite = min(
            alto - 1,
            frontera + margen
        )

        mascara[limite:, :] = 255

        return mascara, limite, None

    # Caso B: claro arriba -> oscuro abajo
    if tipo == "CLARO_ARRIBA_OSCURO_ABAJO":

        frontera = resultado["frontera"]

        limite = max(
            1,
            frontera - margen
        )

        mascara[:limite, :] = 255

        return mascara, limite, None

    # Caso C: región clara entre dos fronteras
    if tipo == "REGION_CLARA_INTERMEDIA":

        frontera_1 = resultado["frontera_1"]
        frontera_2 = resultado["frontera_2"]

        limite_superior = frontera_1 + margen
        limite_inferior = frontera_2 - margen

        if limite_inferior > limite_superior:

            mascara[
                limite_superior:limite_inferior,
                :
            ] = 255

        return (
            mascara,
            limite_superior,
            limite_inferior
        )

    # Si no se detecta una frontera fiable,
    # no seleccionamos píxeles automáticamente.
    return mascara, None, None


# ============================================================
# 9. FILTRADO ADAPTATIVO DENTRO DE LA MÁSCARA GEOMÉTRICA
# ============================================================

def filtrar_dentro_roi(imagen, mascara_geometrica):

    mascara_final = np.zeros_like(
        mascara_geometrica
    )

    pixeles_roi = mascara_geometrica > 0

    if cv2.countNonZero(mascara_geometrica) == 0:

        return mascara_final, None, None

    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    valor = hsv[:, :, 2]

    valores_validos = valor[pixeles_roi]

    limite_oscuro = float(
        np.percentile(
            valores_validos,
            percentil_oscuro
        )
    )

    limite_brillante = float(
        np.percentile(
            valores_validos,
            percentil_brillante
        )
    )

    # Conservar los píxeles que están dentro de la ROI
    # y entre los límites adaptativos de brillo.
    candidatos = (
        pixeles_roi
        & (valor >= limite_oscuro)
        & (valor <= limite_brillante)
    )

    mascara_final[candidatos] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (tamano_kernel, tamano_kernel)
    )

    # Eliminar pequeños puntos aislados
    mascara_final = cv2.morphologyEx(
        mascara_final,
        cv2.MORPH_OPEN,
        kernel
    )

    # Cerrar pequeñas discontinuidades
    mascara_final = cv2.morphologyEx(
        mascara_final,
        cv2.MORPH_CLOSE,
        kernel
    )

    return (
        mascara_final,
        limite_oscuro,
        limite_brillante
    )


# ============================================================
# 10. CREAR SUPERPOSICIÓN
# ============================================================

def crear_superposicion(
    imagen,
    mascara,
    color=(0, 255, 0)
):

    resultado = imagen.copy()

    capa_color = np.zeros_like(imagen)
    capa_color[:] = color

    # Mezclar las imágenes completas primero.
    # Así evitamos el error anterior de addWeighted.
    mezcla = cv2.addWeighted(
        imagen,
        0.65,
        capa_color,
        0.35,
        0
    )

    # Aplicar la mezcla solamente a los píxeles seleccionados
    region = mascara > 0

    resultado[region] = mezcla[region]

    return resultado


# ============================================================
# 11. PREPARAR VISUALIZACIONES
# ============================================================

def poner_titulo(imagen, titulo):

    salida = imagen.copy()

    cv2.rectangle(
        salida,
        (0, 0),
        (salida.shape[1], 25),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        salida,
        titulo,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return salida


def mascara_a_color(mascara):

    return cv2.cvtColor(
        mascara,
        cv2.COLOR_GRAY2BGR
    )


# ============================================================
# 12. PROCESAMIENTO PRINCIPAL
# ============================================================

if not base.exists():

    print("ERROR: No se encontró la carpeta Formazine.")
    print("Ruta utilizada:", base)

    raise SystemExit


for nivel, nombre in pruebas:

    ruta = base / nivel / nombre

    if not ruta.exists():

        print(f"\nNo se encontró: {ruta}")
        continue

    imagen = cv2.imread(
        str(ruta)
    )

    if imagen is None:

        print(f"\nNo se pudo leer: {ruta}")
        continue

    # No volteamos la imagen.
    # El detector analiza el contenido tal como está.
    perfil = calcular_perfil(imagen)

    (
        subida,
        subida_valor,
        caida,
        caida_valor
    ) = buscar_fronteras(perfil)

    resultado = determinar_region(
        subida,
        subida_valor,
        caida,
        caida_valor
    )

    # Crear máscara geométrica
    (
        mascara_geometrica,
        limite_1,
        limite_2
    ) = crear_mascara_geometrica(
        imagen,
        resultado
    )

    # Aplicar filtrado adaptativo dentro de la ROI
    (
        mascara_final,
        limite_oscuro,
        limite_brillante
    ) = filtrar_dentro_roi(
        imagen,
        mascara_geometrica
    )

    # Crear superposiciones
    superpuesta_geometrica = crear_superposicion(
        imagen,
        mascara_geometrica
    )

    superpuesta_final = crear_superposicion(
        imagen,
        mascara_final
    )

    # Preparar los cuatro paneles
    panel_original = poner_titulo(
        imagen,
        f"Original: {nivel} NTU - {nombre}"
    )

    panel_geometrico = poner_titulo(
        mascara_a_color(mascara_geometrica),
        "1. Mascara geometrica"
    )

    panel_filtrado = poner_titulo(
        mascara_a_color(mascara_final),
        "2. Mascara final (filtrado adaptativo)"
    )

    panel_superpuesto = poner_titulo(
        superpuesta_final,
        "3. Region final sobre la imagen"
    )

    comparacion = np.hstack((
        panel_original,
        panel_geometrico,
        panel_filtrado,
        panel_superpuesto
    ))

    # Calcular porcentajes de área conservada
    total_pixeles = (
        imagen.shape[0] *
        imagen.shape[1]
    )

    area_geometrica = cv2.countNonZero(
        mascara_geometrica
    )

    area_final = cv2.countNonZero(
        mascara_final
    )

    porcentaje_geometrico = (
        100 * area_geometrica / total_pixeles
    )

    porcentaje_final = (
        100 * area_final / total_pixeles
    )

    # Mostrar resultados en consola
    print("\n" + "=" * 55)
    print(f"Nivel: {nivel} NTU")
    print(f"Archivo: {nombre}")

    if subida is not None:
        print(
            f"Subida detectada: {subida} "
            f"(cambio={subida_valor:.2f})"
        )
    else:
        print("Subida detectada: ninguna")

    if caida is not None:
        print(
            f"Caída detectada: {caida} "
            f"(cambio={caida_valor:.2f})"
        )
    else:
        print("Caída detectada: ninguna")

    print(
        f"Tipo detectado: {resultado['tipo']}"
    )

    print(
        f"Área de máscara geométrica: "
        f"{porcentaje_geometrico:.2f}%"
    )

    print(
        f"Área tras filtrado adaptativo: "
        f"{porcentaje_final:.2f}%"
    )

    if limite_oscuro is not None:

        print(
            f"Límite adaptativo oscuro (V): "
            f"{limite_oscuro:.1f}"
        )

        print(
            f"Límite adaptativo brillante (V): "
            f"{limite_brillante:.1f}"
        )

    if resultado["tipo"] == "NO_DETERMINADO":

        print(
            "AVISO: no se detectó una frontera fiable. "
            "Revisa esta imagen."
        )

    if (
        subida is not None
        and caida is not None
        and abs(subida - caida) < separacion_minima
    ):

        print(
            "AVISO: las transiciones están muy próximas. "
            "La frontera puede ser un cambio local y debe revisarse."
        )

    print("=" * 55)

    # Mostrar comparación
    comparacion = ajustar_a_pantalla(
        comparacion
    )

    ventana = (
        f"Seleccion de region - "
        f"{nivel} NTU - {nombre}"
    )

    cv2.namedWindow(
        ventana,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        ventana,
        1500,
        650
    )

    cv2.imshow(
        ventana,
        comparacion
    )

    print(
        "Pulsa una tecla para pasar a la siguiente imagen..."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("\nPruebas terminadas.")