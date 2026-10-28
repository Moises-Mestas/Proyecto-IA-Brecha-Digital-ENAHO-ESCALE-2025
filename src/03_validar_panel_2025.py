"""Auditoría preliminar de columnas y valores 2025 en paneles ENAHO.

Se leen solo las primeras 2000 filas de cada CSV desde su ZIP, sin extraer
ni modificar los archivos originales. NO reemplaza la auditoría anual completa.
"""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import csv
import io

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "raw" / "inei"
REPORTE = BASE / "outputs" / "validacion_panel_2025.txt"
MAX_FILAS = 2000

ARCHIVOS = {
    "100 Vivienda": ("1041-Modulo1474.zip", "enaho01-2021-2025-100-panel.csv"),
    "200 Personas": ("1041-Modulo1479.zip", "enaho01-2021-2025-200-panel.csv"),
    "300 Educación": ("1041-Modulo1475.zip", "enaho01a-2021-2025-300-panel.csv"),
    "SUMARIA Hogar": ("1041-Modulo1478.zip", "sumaria-2021-2025-panel.csv"),
}

VARIABLES = {
    "100 Vivienda": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "UBIGEO", "ESTRATO", "P1121", "P1144"],
    "200 Personas": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "P207", "P208A"],
    "300 Educación": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "P300A", "P306", "P307", "P308A", "P308B", "P314A"],
    "SUMARIA Hogar": ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "UBIGEO", "POBREZA", "INGHOG2D", "GASHOG2D", "FACTOR07"],
}

KEY_HOGAR = ["CONGLOME_25", "VIVIENDA_25", "HOGAR_25"]
KEY_PERSONA = KEY_HOGAR + ["CODPERSO_25"]
lineas = []


def informar(mensaje=""):
    print(mensaje)
    lineas.append(str(mensaje))


def escoger_codificacion(zip_archivo, ruta_interna):
    with zip_archivo.open(ruta_interna) as f:
        muestra = f.read(32768)
    try:
        muestra.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin1"


for nombre, (archivo_zip, archivo_csv) in ARCHIVOS.items():
    informar(f"\n========== {nombre} ==========")
    ruta_zip = DATOS / archivo_zip

    if not ruta_zip.is_file():
        informar(f"NO ENCONTRADO: {ruta_zip}")
        continue

    try:
        with ZipFile(ruta_zip) as z:
            interno = next((x for x in z.namelist() if Path(x).name.lower() == archivo_csv.lower()), None)
            if interno is None:
                informar(f"No se encontró {archivo_csv} dentro del ZIP")
                continue

            encoding = escoger_codificacion(z, interno)
            informar(f"Codificación detectada (muestra): {encoding}")

            with z.open(interno) as f:
                with io.TextIOWrapper(f, encoding=encoding, errors="replace", newline="") as texto:
                    lector = csv.DictReader(texto)
                    lector.fieldnames = [x.lstrip("\ufeff").strip().upper() for x in lector.fieldnames]
                    columnas = set(lector.fieldnames)

                    esperadas = [x + "_25" for x in VARIABLES[nombre]]
                    presentes = [x for x in esperadas if x in columnas]
                    faltantes = [x for x in esperadas if x not in columnas]
                    informar(f"Variables 2025 presentes ({len(presentes)}/{len(esperadas)}): {presentes}")
                    informar(f"Variables 2025 faltantes: {faltantes}")
                    if "AÑO_25" not in columnas:
                        informar("No existe AÑO_25; no se puede confirmar participación 2025")
                        continue

                    edades = Counter()
                    no_vacios = Counter()
                    ejemplos = {x: set() for x in presentes if x not in ["AÑO_25", *KEY_PERSONA, "UBIGEO_25"]}
                    vistos = set()
                    duplicados = 0
                    llave = KEY_PERSONA if nombre in ("200 Personas", "300 Educación") else KEY_HOGAR
                    llave_presente = all(x in columnas for x in llave)
                    total = con_2025 = 0

                    for fila in lector:
                        total += 1
                        y = (fila.get("AÑO_25") or "").strip()
                        edades[y if y else "(vacío)"] += 1
                        if y == "2025":
                            con_2025 += 1
                            for var in presentes:
                                val = (fila.get(var) or "").strip()
                                if val:
                                    no_vacios[var] += 1
                                    if var in ejemplos and len(ejemplos[var]) < 4:
                                        ejemplos[var].add(val)
                            if llave_presente:
                                tupla = tuple((fila.get(x) or "").strip() for x in llave)
                                if all(tupla):
                                    if tupla in vistos:
                                        duplicados += 1
                                    vistos.add(tupla)
                        if total >= MAX_FILAS:
                            break

                    informar(f"Filas inspeccionadas (SOLO muestra inicial): {total}")
                    informar(f"Valores observados en AÑO_25: {dict(edades)}")
                    informar(f"Filas con AÑO_25 = 2025 en la muestra: {con_2025}")
                    informar("Variables no vacías entre filas 2025 de la muestra:")
                    for var in presentes:
                        informar(f"  {var}: {no_vacios[var]}/{con_2025}")
                    if llave_presente:
                        informar(f"Duplicados de llave {llave} en muestra 2025 (valores completos): {duplicados}")
                    else:
                        informar(f"NO ESTÁN TODAS las columnas de llave propuestas: {llave}")
                    informar("Ejemplos de valores (sin códigos de identificación):")
                    for var, valores in ejemplos.items():
                        informar(f"  {var}: {sorted(valores)}")

    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        informar(f"ERROR: {exc}")

REPORTE.parent.mkdir(parents=True, exist_ok=True)
REPORTE.write_text("\n".join(lineas), encoding="utf-8")
informar("\nAuditoría preliminar terminada.")
informar(f"Reporte guardado: {REPORTE}")
