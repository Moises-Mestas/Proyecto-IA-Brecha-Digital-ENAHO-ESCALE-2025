
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import struct

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
SALIDA = BASE / "outputs" / "auditoria_02.txt"

# Archivos principales identificados en la práctica 1
ENAHO = {
    "100": ("1041-Modulo1474.zip", "enaho01-2021-2025-100-panel.csv"),
    "200": ("1041-Modulo1479.zip", "enaho01-2021-2025-200-panel.csv"),
    "300": ("1041-Modulo1475.zip", "enaho01a-2021-2025-300-panel.csv"),
    "SUMARIA": ("1041-Modulo1478.zip", "sumaria-2021-2025-panel.csv"),
}

# Variables que menciona el examen
ESPERADAS = {
    "100": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR",
            "UBIGEO", "ESTRATO", "P1144", "P1121"],
    "200": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR",
            "CODPERSO", "P207", "P208A"],
    "300": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR",
            "CODPERSO", "P300A", "P306", "P307",
            "P308A", "P308B", "P314A"],
    "SUMARIA": ["AÑO", "CONGLOME", "VIVIENDA",
                "HOGAR", "POBREZA", "INGHOG2D",
                "GASHOG2D", "FACTOR07"],
}

reporte = []

def mostrar(texto=""):
    print(texto)
    reporte.append(texto)

# Inspeccionar los CSV sin extraerlos
for modulo, (nombre_zip, nombre_csv) in ENAHO.items():
    mostrar(f"\n========== ENAHO {modulo} ==========")

    ruta_zip = RAW / "inei" / nombre_zip

    try:
        with ZipFile(ruta_zip) as z:
            ruta_interna = next(
                (n for n in z.namelist()
                 if Path(n).name.lower() == nombre_csv.lower()),
                None
            )

            if ruta_interna is None:
                mostrar("ERROR: No se encontró el CSV.")
                continue

            with z.open(ruta_interna) as archivo:
                with io.TextIOWrapper(
                    archivo, encoding="latin1", newline=""
                ) as texto:

                    cabecera = texto.readline()
                    separador = (
                        ";" if cabecera.count(";") > cabecera.count(",")
                        else ","
                    )

                    columnas = [
                        c.strip().upper()
                        for c in next(csv.reader(
                            [cabecera], delimiter=separador
                        ))
                    ]

                    mostrar(f"Total de columnas: {len(columnas)}")
                    mostrar(f"Separador: {repr(separador)}")

                    presentes = [
                        c for c in ESPERADAS[modulo] if c in columnas
                    ]
                    ausentes = [
                        c for c in ESPERADAS[modulo] if c not in columnas
                    ]

                    mostrar(f"Variables encontradas: {presentes}")
                    mostrar(f"Variables no encontradas: {ausentes}")

                    # Solo una muestra de las primeras filas
                    lector = csv.reader(texto, delimiter=separador)
                    if "AÑO" in columnas:
                        posicion = columnas.index("AÑO")
                        anios = []
                        for i, fila in enumerate(lector):
                            if posicion < len(fila):
                                anios.append(fila[posicion])
                            if i >= 4:
                                break
                        mostrar(f"Años de las primeras filas: {anios}")

                    mostrar("Lista completa de columnas:")
                    mostrar(", ".join(columnas))

    except Exception as error:
        mostrar(f"ERROR: {error}")

# Leer encabezados DBF del MINEDU sin librerías externas
for nombre_zip, nombre_dbf in [
    ("00_Padron.zip", "Padron.dbf"),
    ("00_Padlocal.zip", "Padlocal.dbf"),
]:
    mostrar(f"\n========== MINEDU {nombre_dbf} ==========")

    try:
        with ZipFile(RAW / "minedu" / nombre_zip) as z:
            with z.open(nombre_dbf) as archivo:
                cabecera = archivo.read(32)

                registros = struct.unpack("<I", cabecera[4:8])[0]
                longitud_cabecera = struct.unpack(
                    "<H", cabecera[8:10]
                )[0]

                columnas = []
                while archivo.tell() < longitud_cabecera:
                    descriptor = archivo.read(32)

                    if not descriptor or descriptor[0] == 13:
                        break

                    nombre = descriptor[:11].split(b"\x00")[0]
                    columnas.append(
                        nombre.decode("latin1").strip().upper()
                    )

                mostrar(f"Registros declarados: {registros}")
                mostrar(f"Total de columnas: {len(columnas)}")
                mostrar("Lista completa de columnas:")
                mostrar(", ".join(columnas))

    except Exception as error:
        mostrar(f"ERROR: {error}")

SALIDA.parent.mkdir(parents=True, exist_ok=True)
SALIDA.write_text("\n".join(reporte), encoding="utf-8")

print("\n========== AUDITORÍA COMPLETADA ==========")
print(f"Reporte guardado en: {SALIDA}")
