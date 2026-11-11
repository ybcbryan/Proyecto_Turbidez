
from pathlib import Path
import sys
import cv2
import numpy as np

# ------------------------------------------------------------
# 1. Importar las funciones del detector existente
# ------------------------------------------------------------

CARPETA_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_SRC))

from validar_mascaras_dataset import (
    BASE_DATASET,
    calcular_perfil,
    buscar_fronteras,
    determinar_region,
    crear_mascara_geometrica,
    filtrar_dentro_roi,
    porcentaje_area,
    superponer_mascara,
)

# ------------------------------------------------------------
# 2. Imágenes que vamos a revisar
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 3. Procesar y mostrar cada imagen
# ------------------------------------------------------------

def revisar_imagen(nivel, nombre):
    ruta = BASE_DATASET / nivel / nombre

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"No se pudo abrir: {ruta}")
        return

    perfil = calcular_perfil(imagen)

    subida, subida_valor, caida, caida_valor = (
        buscar_fronteras(perfil)
    )

    region = determinar_region(
        subida,
        subida_valor,
        caida,
        caida_valor,
    )

    mascara_geometrica, limite1, limite2 = (
        crear_mascara_geometrica(imagen, region)
    )

    mascara_final, limite_oscuro, limite_brillante = (
        filtrar_dentro_roi(imagen, mascara_geometrica)
    )

    area_geometrica = porcentaje_area(mascara_geometrica)
    area_final = porcentaje_area(mascara_final)

    # Mostrar la máscara final en blanco y negro.
    mascara_color = cv2.cvtColor(
        mascara_final, cv2.COLOR_GRAY2BGR
    )

    # Superponer la máscara final sobre la fotografía.
    superpuesta = superponer_mascara(imagen, mascara_final)

    # Dibujar las fronteras detectadas sobre una copia.
    fronteras = imagen.copy()

    if subida is not None:
        cv2.line(
            fronteras,
            (0, subida),
            (fronteras.shape[1] - 1, subida),
            (0, 0, 255),
            2,
        )

    if caida is not None:
        cv2.line(
            fronteras,
            (0, caida),
            (fronteras.shape[1] - 1, caida),
            (255, 0, 0),
            2,
        )

    # Añadir etiquetas para identificar cada panel.
    paneles = [
        ("ORIGINAL", imagen),
        ("FRONTERAS", fronteras),
        ("MASCARA FINAL", mascara_color),
        ("SUPERPOSICION", superpuesta),
    ]

    paneles_redimensionados = []

    for titulo, panel in paneles:
        panel = panel.copy()

        cv2.putText(
            panel,
            titulo,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panel = cv2.resize(panel, (244, 244))
        paneles_redimensionados.append(panel)

    comparacion = np.hstack(paneles_redimensionados)

    print("\n" + "-" * 55)
    print(f"Nivel: {nivel} NTU")
    print(f"Archivo: {nombre}")
    print(f"Tipo detectado: {region['tipo']}")
    print(f"Área geométrica: {area_geometrica:.4f}%")
    print(f"Área final: {area_final:.4f}%")
    print(f"Frontera de subida: {subida}")
    print(f"Frontera de caída: {caida}")

    cv2.imshow(
        f"{nivel} NTU - {nombre}",
        comparacion,
    )

    print("Pulsa una tecla para pasar a la siguiente imagen.")
    print("Pulsa ESC para terminar.")

    tecla = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return tecla


def main():
    for nivel, nombre in IMAGENES:
        tecla = revisar_imagen(nivel, nombre)

        if tecla == 27:
            break

    cv2.destroyAllWindows()
    print("\nRevisión terminada.")


if __name__ == "__main__":
    main()
