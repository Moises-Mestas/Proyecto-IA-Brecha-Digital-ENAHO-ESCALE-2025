
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import csv

# 1. Ubicar la carpeta principal del proyecto
BASE = Path(__file__).resolve().parent.parent

# 2. Definir dónde se encuentran nuestros datos
CARPETAS = {
    "INEI": BASE / "data" / "raw" / "inei",
    "MINEDU": BASE / "data" / "raw" / "minedu",
}

# 3. Preparar la carpeta donde guardaremos el inventario
SALIDA = BASE / "outputs" / "inventario_zip.csv"
SALIDA.parent.mkdir(parents=True, exist_ok=True)

inventario = []

# 4. Recorrer las carpetas de cada institución
for institucion, carpeta in CARPETAS.items():

    print(f"\n========== {institucion} ==========")

    if not carpeta.exists():
        print(f"No existe la carpeta: {carpeta}")
        continue

    archivos_zip = sorted(
        archivo for archivo in carpeta.rglob("*")
        if archivo.is_file() and archivo.suffix.lower() == ".zip"
    )

    print(f"Archivos ZIP encontrados: {len(archivos_zip)}")

    # 5. Inspeccionar cada ZIP sin descomprimirlo
    for ruta_zip in archivos_zip:

        print(f"\nZIP: {ruta_zip.name}")

        try:
            with ZipFile(ruta_zip, "r") as zip_actual:
                archivos = [
                    info for info in zip_actual.infolist()
                    if not info.is_dir()
                ]

                print(f"Archivos internos: {len(archivos)}")

                # Mostrar hasta 12 archivos internos
                for info in archivos[:12]:
                    print(f"  - {info.filename}")

                if len(archivos) > 12:
                    print("  ... (hay más archivos)")

                # Guardar todos los nombres en el inventario
                for info in archivos:
                    inventario.append({
                        "institucion": institucion,
                        "zip_origen": ruta_zip.name,
                        "archivo_interno": info.filename,
                        "extension": Path(info.filename).suffix.lower(),
                        "tamano_bytes": info.file_size,
                    })

        except (BadZipFile, OSError) as error:
            print(f"Error al leer el ZIP: {error}")

# 6. Crear un CSV con el inventario completo
columnas = [
    "institucion",
    "zip_origen",
    "archivo_interno",
    "extension",
    "tamano_bytes",
]

with open(SALIDA, "w", newline="", encoding="utf-8-sig") as archivo:
    escritor = csv.DictWriter(archivo, fieldnames=columnas)
    escritor.writeheader()
    escritor.writerows(inventario)

print("\n========== RESULTADO FINAL ==========")
print(f"Total de archivos internos registrados: {len(inventario)}")
print(f"Inventario guardado en: {SALIDA}")
print("Inspección terminada correctamente.")
