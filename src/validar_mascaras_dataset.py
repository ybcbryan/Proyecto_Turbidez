from pathlib import Path
import csv
import cv2
import numpy as np


# ============================================================
# 1. RUTAS
# ============================================================

# Estructura esperada:
# C:\Proyecto\src\validar_mascaras_dataset.py
# C:\Proyecto\WaTur-Water-Turbidity-Dataset-main\Formazine\...

BASE_PROYECTO = Path(__file__).resolve().parents[1]

BASE_DATASET = (
    BASE_PROYECTO
    / "WaTur-Water-Turbidity-Dataset-main"
    / "Formazine"
)

CARPETA_RESULTADOS = (
    BASE_PROYECTO / "src" / "resultados_validacion"
)

CSV_SALIDA = CARPETA_RESULTADOS / "validacion_mascaras.csv"


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

NIVELES = ["0", "0.5", "1", "2.5", "4", "5", "7.5", "10", "40"]

# 10 imágenes aleatorias por nivel.
# Cambiar a None para procesar todas las imágenes.
IMAGENES_POR_NIVEL = 10

SEMILLA = 42

# Parámetros geométricos
MARGEN = 8
FILA_INICIO = 25
FILA_FIN = 180
SEPARACION_MINIMA = 20
UMBRAL_TRANSICION = 1.0

# Filtrado adaptativo de intensidad V en HSV
PERCENTIL_OSCURO = 5
PERCENTIL_BRILLANTE = 98
TAMANO_KERNEL = 3

# Criterios provisionales de revisión
AREA_MINIMA_PORCENTAJE = 10.0
AREA_MAXIMA_PORCENTAJE = 90.0

# Mostrar cada imagen y su máscara
MOSTRAR_IMAGENES = True


# ============================================================
# 3. PERFIL DE INTENSIDAD
# ============================================================

def calcular_perfil(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    perfil = np.mean(gris, axis=1).astype(np.float32)

    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1, 9),
        0
    ).flatten()

    return perfil_suave


# ============================================================
# 4. DETECTAR TRANSICIONES
# ============================================================

def buscar_fronteras(perfil):
    inicio = FILA_INICIO
    fin = min(FILA_FIN, len(perfil) - 1)

    if fin - inicio < 3:
        return None, None, None, None

    diferencias = np.diff(perfil[inicio:fin])

    if diferencias.size == 0:
        return None, None, None, None

    indice_subida = int(np.argmax(diferencias))
    indice_caida = int(np.argmin(diferencias))

    valor_subida = float(diferencias[indice_subida])
    valor_caida = float(diferencias[indice_caida])

    subida = (
        inicio + indice_subida + 1
        if valor_subida > 0 else None
    )

    caida = (
        inicio + indice_caida + 1
        if valor_caida < 0 else None
    )

    subida_valor = (
        valor_subida if subida is not None else None
    )

    caida_valor = (
        valor_caida if caida is not None else None
    )

    return subida, subida_valor, caida, caida_valor


def determinar_region(
    subida,
    subida_valor,
    caida,
    caida_valor
):
    subida_valida = (
        subida is not None
        and subida_valor is not None
        and subida_valor >= UMBRAL_TRANSICION
    )

    caida_valida = (
        caida is not None
        and caida_valor is not None
        and abs(caida_valor) >= UMBRAL_TRANSICION
    )

    # Dos fronteras separadas: posible región clara intermedia
    if (
        subida_valida
        and caida_valida
        and subida < caida
        and (caida - subida) >= SEPARACION_MINIMA
    ):
        return {
            "tipo": "REGION_CLARA_INTERMEDIA",
            "frontera_1": subida,
            "frontera_2": caida
        }

    # Zona oscura arriba y agua candidata abajo
    if subida_valida and not caida_valida:
        return {
            "tipo": "OSCURO_ARRIBA_CLARO_ABAJO",
            "frontera": subida
        }

    # Agua candidata arriba y zona oscura abajo
    if caida_valida and not subida_valida:
        return {
            "tipo": "CLARO_ARRIBA_OSCURO_ABAJO",
            "frontera": caida
        }

    # Si ambas son válidas pero no delimitan una región intermedia,
    # se conserva la transición de mayor magnitud.
    if subida_valida and caida_valida:
        if subida_valor >= abs(caida_valor):
            return {
                "tipo": "OSCURO_ARRIBA_CLARO_ABAJO",
                "frontera": subida
            }

        return {
            "tipo": "CLARO_ARRIBA_OSCURO_ABAJO",
            "frontera": caida
        }

    return {"tipo": "NO_DETERMINADO"}


# ============================================================
# 5. CREAR MÁSCARA GEOMÉTRICA
# ============================================================

def crear_mascara_geometrica(imagen, resultado):
    alto, ancho = imagen.shape[:2]

    mascara = np.zeros(
        (alto, ancho),
        dtype=np.uint8
    )

    tipo = resultado["tipo"]

    if tipo == "OSCURO_ARRIBA_CLARO_ABAJO":
        limite = min(
            alto - 1,
            resultado["frontera"] + MARGEN
        )

        mascara[limite:, :] = 255

        return mascara, limite, None

    if tipo == "CLARO_ARRIBA_OSCURO_ABAJO":
        limite = max(
            1,
            resultado["frontera"] - MARGEN
        )

        mascara[:limite, :] = 255

        return mascara, limite, None

    if tipo == "REGION_CLARA_INTERMEDIA":
        superior = max(
            0,
            min(alto, resultado["frontera_1"] + MARGEN)
        )

        inferior = max(
            0,
            min(alto, resultado["frontera_2"] - MARGEN)
        )

        if inferior > superior:
            mascara[superior:inferior, :] = 255

        return mascara, superior, inferior

    return mascara, None, None


# ============================================================
# 6. FILTRADO ADAPTATIVO
# ============================================================

def filtrar_dentro_roi(imagen, mascara_geometrica):
    mascara_final = np.zeros_like(mascara_geometrica)

    region = mascara_geometrica > 0

    if cv2.countNonZero(mascara_geometrica) == 0:
        return mascara_final, None, None

    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    valor = hsv[:, :, 2]
    valores_region = valor[region]

    limite_oscuro = float(
        np.percentile(
            valores_region,
            PERCENTIL_OSCURO
        )
    )

    limite_brillante = float(
        np.percentile(
            valores_region,
            PERCENTIL_BRILLANTE
        )
    )

    candidatos = (
        region
        & (valor >= limite_oscuro)
        & (valor <= limite_brillante)
    )

    mascara_final[candidatos] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TAMANO_KERNEL, TAMANO_KERNEL)
    )

    mascara_final = cv2.morphologyEx(
        mascara_final,
        cv2.MORPH_OPEN,
        kernel
    )

    mascara_final = cv2.morphologyEx(
        mascara_final,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mascara_final, limite_oscuro, limite_brillante


# ============================================================
# 7. MEDIDAS Y VISUALIZACIÓN
# ============================================================

def porcentaje_area(mascara):
    total = mascara.shape[0] * mascara.shape[1]

    if total == 0:
        return 0.0

    return (
        100.0
        * cv2.countNonZero(mascara)
        / total
    )


def superponer_mascara(imagen, mascara):
    capa_verde = np.zeros_like(imagen)
    capa_verde[:] = (0, 255, 0)

    mezcla = cv2.addWeighted(
        imagen,
        0.65,
        capa_verde,
        0.35,
        0
    )

    resultado = imagen.copy()
    seleccion = mascara > 0
    resultado[seleccion] = mezcla[seleccion]

    return resultado


def medir_imagen(imagen):
    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    return {
        "brillo_medio_gris": float(np.mean(gris)),
        "brillo_mediano_gris": float(np.median(gris)),
        "desviacion_gris": float(np.std(gris)),
        "valor_medio_HSV": float(np.mean(hsv[:, :, 2])),
        "saturacion_media_HSV": float(np.mean(hsv[:, :, 1]))
    }


def obtener_imagenes(carpeta, cantidad, semilla):
    extensiones = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    archivos = sorted(
        archivo
        for archivo in carpeta.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in extensiones
    )

    if cantidad is None or len(archivos) <= cantidad:
        return archivos

    generador = np.random.default_rng(semilla)

    indices = sorted(
        generador.choice(
            len(archivos),
            size=cantidad,
            replace=False
        ).tolist()
    )

    return [archivos[i] for i in indices]


# ============================================================
# 8. CLASIFICACIÓN PROVISIONAL
# ============================================================

def clasificar_resultado(
    tipo,
    area_final,
    advertencia_transiciones
):
    if tipo == "NO_DETERMINADO" or area_final <= 0:
        return (
            "RECHAZADA_SIN_REGION",
            "No se detectó una región utilizable."
        )

    if advertencia_transiciones:
        return (
            "REVISION_MANUAL",
            "Las transiciones están demasiado próximas."
        )

    if area_final < AREA_MINIMA_PORCENTAJE:
        return (
            "REVISION_MANUAL",
            "Área conservada inferior al criterio provisional."
        )

    if area_final > AREA_MAXIMA_PORCENTAJE:
        return (
            "REVISION_MANUAL",
            "Área conservada superior al criterio provisional."
        )

    return (
        "CANDIDATA_A_ACEPTACION",
        "Cumple criterios iniciales; requiere revisión visual."
    )


# ============================================================
# 9. PROCESAR EL DATASET Y GUARDAR CSV
# ============================================================

def main():
    if not BASE_DATASET.exists():
        print("ERROR: no se encontró la carpeta Formazine.")
        print("Ruta buscada:")
        print(BASE_DATASET)
        print(
            "\nComprueba que la carpeta del dataset esté "
            "al mismo nivel que la carpeta src."
        )
        return

    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    columnas = [
        "nivel_ntu",
        "archivo",
        "ruta",
        "ancho",
        "alto",
        "tipo_detectado",
        "subida_fila",
        "subida_cambio",
        "caida_fila",
        "caida_cambio",
        "separacion_transiciones",
        "area_geometrica_pct",
        "area_final_pct",
        "limite_oscuro_V",
        "limite_brillante_V",
        "brillo_medio_gris",
        "brillo_mediano_gris",
        "desviacion_gris",
        "valor_medio_HSV",
        "saturacion_media_HSV",
        "advertencia_transiciones",
        "decision_provisional",
        "motivo"
    ]

    registros = []
    detener = False

    for indice_nivel, nivel in enumerate(NIVELES):
        carpeta_nivel = BASE_DATASET / nivel

        if not carpeta_nivel.exists():
            print(
                f"\nAVISO: no existe la carpeta del nivel {nivel}:"
            )
            print(carpeta_nivel)
            continue

        archivos = obtener_imagenes(
            carpeta_nivel,
            IMAGENES_POR_NIVEL,
            SEMILLA + indice_nivel
        )

        print(
            f"\n{'=' * 55}\n"
            f"Nivel: {nivel} NTU\n"
            f"Imágenes seleccionadas: {len(archivos)}"
        )

        for archivo in archivos:
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                print("No se pudo leer:", archivo.name)
                continue

            alto, ancho = imagen.shape[:2]

            perfil = calcular_perfil(imagen)

            subida, subida_valor, caida, caida_valor = (
                buscar_fronteras(perfil)
            )

            region = determinar_region(
                subida,
                subida_valor,
                caida,
                caida_valor
            )

            mascara_geometrica, _, _ = (
                crear_mascara_geometrica(
                    imagen,
                    region
                )
            )

            mascara_final, limite_oscuro, limite_brillante = (
                filtrar_dentro_roi(
                    imagen,
                    mascara_geometrica
                )
            )

            area_geometrica = porcentaje_area(
                mascara_geometrica
            )

            area_final = porcentaje_area(
                mascara_final
            )

            if subida is not None and caida is not None:
                separacion = abs(subida - caida)
            else:
                separacion = None

            advertencia = (
                separacion is not None
                and separacion < SEPARACION_MINIMA
            )

            decision, motivo = clasificar_resultado(
                region["tipo"],
                area_final,
                advertencia
            )

            medidas = medir_imagen(imagen)

            registro = {
                "nivel_ntu": nivel,
                "archivo": archivo.name,
                "ruta": str(archivo),
                "ancho": ancho,
                "alto": alto,
                "tipo_detectado": region["tipo"],
                "subida_fila": (
                    subida if subida is not None else ""
                ),
                "subida_cambio": (
                    round(subida_valor, 4)
                    if subida_valor is not None else ""
                ),
                "caida_fila": (
                    caida if caida is not None else ""
                ),
                "caida_cambio": (
                    round(caida_valor, 4)
                    if caida_valor is not None else ""
                ),
                "separacion_transiciones": (
                    separacion if separacion is not None else ""
                ),
                "area_geometrica_pct": round(
                    area_geometrica, 4
                ),
                "area_final_pct": round(
                    area_final, 4
                ),
                "limite_oscuro_V": (
                    round(limite_oscuro, 2)
                    if limite_oscuro is not None else ""
                ),
                "limite_brillante_V": (
                    round(limite_brillante, 2)
                    if limite_brillante is not None else ""
                ),
                "brillo_medio_gris": round(
                    medidas["brillo_medio_gris"], 4
                ),
                "brillo_mediano_gris": round(
                    medidas["brillo_mediano_gris"], 4
                ),
                "desviacion_gris": round(
                    medidas["desviacion_gris"], 4
                ),
                "valor_medio_HSV": round(
                    medidas["valor_medio_HSV"], 4
                ),
                "saturacion_media_HSV": round(
                    medidas["saturacion_media_HSV"], 4
                ),
                "advertencia_transiciones": advertencia,
                "decision_provisional": decision,
                "motivo": motivo
            }

            registros.append(registro)

            print(
                f"{archivo.name}: "
                f"{region['tipo']} | "
                f"área final={area_final:.2f}% | "
                f"{decision}"
            )

            if MOSTRAR_IMAGENES:
                mascara_color = cv2.cvtColor(
                    mascara_final,
                    cv2.COLOR_GRAY2BGR
                )

                superpuesta = superponer_mascara(
                    imagen,
                    mascara_final
                )

                escala = min(
                    400 / ancho,
                    300 / alto,
                    1.0
                )

                tamano = (
                    max(1, int(ancho * escala)),
                    max(1, int(alto * escala))
                )

                original_reducida = cv2.resize(
                    imagen, tamano
                )

                mascara_reducida = cv2.resize(
                    mascara_color, tamano
                )

                superpuesta_reducida = cv2.resize(
                    superpuesta, tamano
                )

                comparacion = np.hstack([
                    original_reducida,
                    mascara_reducida,
                    superpuesta_reducida
                ])

                nombre_ventana = (
                    f"{nivel} NTU - {archivo.name} "
                    f"| Original / Mascara / Superpuesta"
                )

                cv2.imshow(
                    nombre_ventana,
                    comparacion
                )

                print(
                    "Pulsa cualquier tecla para continuar; "
                    "ESC para detener la revisión."
                )

                tecla = cv2.waitKey(0)
                cv2.destroyAllWindows()

                if tecla == 27:
                    detener = True
                    break

        if detener:
            break

    with CSV_SALIDA.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo_csv:

        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=columnas,
            delimiter=";"
        )

        escritor.writeheader()
        escritor.writerows(registros)

    print("\n" + "=" * 60)
    print("VALIDACIÓN TERMINADA")
    print("Imágenes procesadas:", len(registros))
    print("CSV guardado en:")
    print(CSV_SALIDA)

    conteos = {}

    for registro in registros:
        decision = registro["decision_provisional"]
        conteos[decision] = conteos.get(decision, 0) + 1

    print("\nResumen de decisiones:")

    for decision, cantidad in sorted(conteos.items()):
        print(f"  {decision}: {cantidad}")

    print(
        "\nIMPORTANTE: las decisiones son provisionales. "
        "No se debe rechazar una imagen únicamente por ser oscura "
        "sin revisar si la región de agua es realmente utilizable."
    )