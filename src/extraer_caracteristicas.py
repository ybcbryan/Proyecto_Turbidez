
from pathlib import Path
import csv
import cv2
import numpy as np

# Importamos las funciones del detector ya validado.
from validar_mascaras_dataset import (
    BASE_DATASET,
    NIVELES,
    calcular_perfil,
    buscar_fronteras,
    determinar_region,
    crear_mascara_geometrica,
    filtrar_dentro_roi,
    porcentaje_area,
)

CARPETA_RESULTADOS = (
    Path(__file__).resolve().parent / "resultados_caracteristicas"
)
CSV_SALIDA = CARPETA_RESULTADOS / "caracteristicas_dataset.csv"

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def calcular_estadisticas(imagen, mascara, espacio, canales):
    """Calcula media y desviación estándar en los píxeles seleccionados."""

    imagen_color = cv2.cvtColor(imagen, espacio)
    pixeles = imagen_color[mascara > 0]

    resultado = {}

    if len(pixeles) == 0:
        for canal in canales:
            resultado[f"{canal}_media"] = ""
            resultado[f"{canal}_std"] = ""
        return resultado

    for indice, canal in enumerate(canales):
        valores = pixeles[:, indice].astype(np.float64)

        resultado[f"{canal}_media"] = round(
            float(np.mean(valores)), 4
        )
        resultado[f"{canal}_std"] = round(
            float(np.std(valores)), 4
        )

    return resultado


def extraer_caracteristicas(imagen, mascara):
    """Extrae estadísticas RGB, HSV y Lab de la región seleccionada."""

    caracteristicas = {}

    espacios = [
        (
            "RGB",
            cv2.COLOR_BGR2RGB,
            ["R", "G", "B"],
        ),
        (
            "HSV",
            cv2.COLOR_BGR2HSV,
            ["H", "S", "V"],
        ),
        (
            "Lab",
            cv2.COLOR_BGR2LAB,
            ["L", "a", "b"],
        ),
    ]

    for nombre, conversion, canales in espacios:
        estadisticas = calcular_estadisticas(
            imagen, mascara, conversion, canales
        )

        for clave, valor in estadisticas.items():
            caracteristicas[f"{nombre}_{clave}"] = valor

    return caracteristicas


def main():
    if not BASE_DATASET.exists():
        print("ERROR: no se encontró el dataset:")
        print(BASE_DATASET)
        return

    CARPETA_RESULTADOS.mkdir(
        parents=True, exist_ok=True
    )

    registros = []

    for nivel in NIVELES:
        carpeta = BASE_DATASET / nivel

        if not carpeta.exists():
            print(f"AVISO: no existe la carpeta {carpeta}")
            continue

        archivos = sorted(
            archivo for archivo in carpeta.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES
        )

        print(f"\nNivel {nivel} NTU: {len(archivos)} imágenes")

        for numero, archivo in enumerate(archivos, start=1):
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                print("No se pudo leer:", archivo.name)
                continue

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

            mascara_geometrica, _, _ = crear_mascara_geometrica(
                imagen, region
            )

            mascara_final, _, _ = filtrar_dentro_roi(
                imagen, mascara_geometrica
            )

            area_final = porcentaje_area(mascara_final)

            if cv2.countNonZero(mascara_final) == 0:
                print(
                    f"AVISO: máscara vacía en {nivel}/{archivo.name}"
                )
                continue

            caracteristicas = extraer_caracteristicas(
                imagen, mascara_final
            )

            registro = {
                "nivel_ntu": float(nivel),
                "archivo": archivo.name,
                "ruta": str(archivo),
                "tipo_region": region["tipo"],
                "area_mascara_pct": round(area_final, 4),
            }

            registro.update(caracteristicas)
            registros.append(registro)

            if numero % 200 == 0:
                print(f"  Procesadas {numero} imágenes...")

    if not registros:
        print("ERROR: no se generaron registros.")
        return

    columnas = list(registros[0].keys())

    with CSV_SALIDA.open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo_csv:
        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=columnas,
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(registros)

    print("\nEXTRACCIÓN TERMINADA")
    print("Imágenes con características:", len(registros))
    print("Archivo generado:")
    print(CSV_SALIDA)


if __name__ == "__main__":
    main()
