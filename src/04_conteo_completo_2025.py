"""Auditoria completa de registros ENAHO 2025 dentro de los ZIP panel 2021-2025.

No extrae los archivos, no modifica los datos originales y no necesita librerias externas.
El analisis recorre TODAS las filas, por lo que puede tomar varios minutos.
"""

from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import csv
import io

BASE = Path(__file__).resolve().parent.parent
INEI = BASE / "data" / "raw" / "inei"
SALIDA = BASE / "outputs" / "validacion_completa_2025.txt"
SALIDA.parent.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "100 Vivienda": (
        "1041-Modulo1474.zip", "enaho01-2021-2025-100-panel.csv",
        ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25", "UBIGEO_25", "ESTRATO_25", "P1121_25", "P1144_25"],
    ),
    "200 Personas": (
        "1041-Modulo1479.zip", "enaho01-2021-2025-200-panel.csv",
        ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25", "CODPERSO_25", "P207_25", "P208A_25"],
    ),
    "300 Educacion": (
        "1041-Modulo1475.zip", "enaho01a-2021-2025-300-panel.csv",
        ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25", "CODPERSO_25", "P208A_25", "P300A_25", "P306_25", "P307_25", "P308A_25", "P308B_25", "P314A_25"],
    ),
    "SUMARIA Hogar": (
        "1041-Modulo1478.zip", "sumaria-2021-2025-panel.csv",
        ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25", "UBIGEO_25", "POBREZA_25", "INGHOG2D_25", "GASHOG2D_25", "FACTOR07_25"],
    ),
}

lineas = []


def reportar(mensaje=""):
    print(mensaje, flush=True)
    lineas.append(mensaje)


def no_vacio(valor):
    return valor is not None and str(valor).strip() != ""


for modulo, (nombre_zip, nombre_csv, variables) in CONFIG.items():
    reportar(f"\n========== {modulo} ==========")
    zip_path = INEI / nombre_zip
    if not zip_path.exists():
        reportar(f"ERROR: falta {zip_path}")
        continue

    try:
        with ZipFile(zip_path) as z:
            nombre_interno = next(
                (n for n in z.namelist() if Path(n).name.lower() == nombre_csv.lower()),
                None,
            )
            if nombre_interno is None:
                reportar(f"ERROR: no aparece {nombre_csv} dentro del ZIP")
                continue

            with z.open(nombre_interno) as datos:
                with io.TextIOWrapper(datos, encoding="utf-8-sig", newline="") as texto:
                    lector = csv.reader(texto)
                    cabeceras = [c.strip().upper() for c in next(lector)]
                    posiciones = {nombre: i for i, nombre in enumerate(cabeceras)}
                    obligatorias = ["AÑO_25"] + variables
                    ausentes = [v for v in obligatorias if v not in posiciones]
                    if ausentes:
                        reportar(f"ERROR: faltan columnas {ausentes}")
                        continue

                    idx_anio = posiciones["AÑO_25"]
                    idx_edad = posiciones.get("P208A_25")
                    contador_anio = Counter()
                    completos = Counter()
                    total = 0
                    cantidad_2025 = 0
                    escolares_6_17 = 0
                    edades_no_validas = 0
                    primera_2025 = None
                    ultima_2025 = None

                    reportar("Leyendo todas las filas sin extraer el ZIP...")
                    for fila in lector:
                        total += 1
                        anio = fila[idx_anio].strip() if idx_anio < len(fila) else ""
                        contador_anio[anio if anio else "(vacio)"] += 1

                        if anio == "2025":
                            cantidad_2025 += 1
                            if primera_2025 is None:
                                primera_2025 = total
                            ultima_2025 = total

                            for nombre in variables:
                                pos = posiciones[nombre]
                                if pos < len(fila) and no_vacio(fila[pos]):
                                    completos[nombre] += 1

                            if idx_edad is not None and idx_edad < len(fila):
                                edad_txt = fila[idx_edad].strip()
                                if edad_txt:
                                    try:
                                        edad = float(edad_txt)
                                        if 6 <= edad <= 17 and edad.is_integer():
                                            escolares_6_17 += 1
                                    except ValueError:
                                        edades_no_validas += 1

                        if total % 100000 == 0:
                            print(
                                f"  Procesadas {total:,} filas; "
                                f"con AÑO_25=2025: {cantidad_2025:,}",
                                flush=True,
                            )

                    reportar(f"Total de filas del CSV: {total:,}")
                    reportar(f"Distribucion COMPLETA de AÑO_25: {dict(contador_anio)}")
                    reportar(f"Filas con AÑO_25 = 2025: {cantidad_2025:,}")
                    reportar(f"Primera fila de datos 2025: {primera_2025}")
                    reportar(f"Ultima fila de datos 2025: {ultima_2025}")
                    for nombre in variables:
                        n = completos[nombre]
                        porcentaje = 100 * n / cantidad_2025 if cantidad_2025 else 0
                        reportar(f"  {nombre}: {n:,}/{cantidad_2025:,} no vacios ({porcentaje:.1f}%)")
                    if idx_edad is not None:
                        reportar(f"Personas de 6 a 17 anos (entre filas 2025): {escolares_6_17:,}")
                        reportar(f"Edades no interpretables: {edades_no_validas:,}")

    except (OSError, UnicodeError, csv.Error, EOFError) as error:
        reportar(f"ERROR leyendo {nombre_zip}: {type(error).__name__}: {error}")

SALIDA.write_text("\n".join(lineas) + "\n", encoding="utf-8")
print(f"\nReporte completo guardado en: {SALIDA}")
