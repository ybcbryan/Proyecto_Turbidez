from pathlib import Path

# ============================================================
# RUTA A LA CARPETA FORMAZINE/0
# ============================================================

carpeta = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine" / "0"

# Buscar imágenes JPG
imagenes = list(carpeta.glob("*.jpg"))

# Convertir nombres a números
numeros = []

for imagen in imagenes:
    try:
        numero = int(imagen.stem)
        numeros.append(numero)
    except ValueError:
        print(f"Archivo ignorado: {imagen.name}")

# Orden numérico
numeros.sort()

print("========================================")
print("INSPECCIÓN DEL DATASET")
print("========================================")

print(f"Cantidad de imágenes: {len(numeros)}")

if not numeros:
    print(f"No se encontraron imágenes JPG en: {carpeta}")
    raise SystemExit(0)

print(f"Número menor: {numeros[0]}")
print(f"Número mayor: {numeros[-1]}")

# Buscar números faltantes
faltantes = []

for numero in range(numeros[0], numeros[-1] + 1):
    if numero not in numeros:
        faltantes.append(numero)

print(f"\nCantidad de números faltantes: {len(faltantes)}")

if faltantes:
    print("Números faltantes:")
    print(faltantes)
else:
    print("No hay números faltantes.")

# Mostrar primeras imágenes en orden numérico
print("\nPrimeras 10:")
for numero in numeros[:10]:
    print(f"{numero}.jpg")

# Mostrar últimas imágenes
print("\nÚltimas 10:")
for numero in numeros[-10:]:
    print(f"{numero}.jpg")