"""Fase 1 del examen IA ENAHO–ESCALE 2025.

Limpieza ESTRUCTURAL de los cuatro CSV PANEL 2021–2025 entregados.
- Lee los datos dentro de ZIP, sin extraer ni modificar originales.
- Conserva filas con AÑO_25 = 2025.
- Selecciona variables pertinentes, transforma nombres terminados en _25.
- Mantiene las llaves/UBIGEO como TEXTO para no perder ceros.
- Vacíos/espacios -> celda vacía; NO imputa ni cambia códigos categóricos.
- Audita duplicados, llaves incompletas y edades de 6–17 años.
- Guarda CSV de salida e informe, sin entrenar ni hacer merges todavía.

Ejecutar desde la raíz del proyecto:
    python src/05_limpiar_enaho_2025.py

No es extracción de la muestra ENAHO anual completa: usa el PANEL suministrado.
"""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import os

RAIZ = Path(__file__).resolve().parent.parent
INEI = RAIZ / "data" / "raw" / "inei"
INTERIM = RAIZ / "data" / "interim"
OUTPUTS = RAIZ / "outputs"
INTERIM.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Columnas prioritarias para el examen, identificadas en auditoria_02.txt.
# Se incluyen variables para futuros perfiles digitales/educativos, pero
# sus CÓDIGOS todavía NO se interpretan como 'sí/no' sin el diccionario.
MODULOS = {
    "100": {
        "zip": "1041-Modulo1474.zip",
        "csv": "enaho01-2021-2025-100-panel.csv",
        "salida": "enaho_100_2025_limpio.csv",
        "llave": ["CONGLOME", "VIVIENDA", "HOGAR"],
        "variables": [
            "AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "UBIGEO",
            "DOMINIO", "ESTRATO", "P1121", "P1144", "FACTOR07",
        ],
    },
    "200": {
        "zip": "1041-Modulo1479.zip",
        "csv": "enaho01-2021-2025-200-panel.csv",
        "salida": "enaho_200_2025_limpio.csv",
        "llave": ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"],
        "variables": [
            "AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
            "UBIGEO", "P203", "P207", "P208A", "FACPOB07", "FACTOR07",
        ],
    },
    "300": {
        "zip": "1041-Modulo1475.zip",
        "csv": "enaho01a-2021-2025-300-panel.csv",
        "salida": "enaho_300_2025_limpio.csv",
        "llave": ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"],
        "variables": [
            "AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
            "UBIGEO", "P208A", "P300A", "P306", "P307", "P308A",
            "P308B", "P314A",
            *[f"P314B_{i}" for i in range(1, 8)],
            *[f"P316_{i}" for i in range(1, 13)],
            "FACTORA07", "FACTOR07",
        ],
    },
    "SUMARIA": {
        "zip": "1041-Modulo1478.zip",
        "csv": "sumaria-2021-2025-panel.csv",
        "salida": "enaho_sumaria_2025_limpio.csv",
        "llave": ["CONGLOME", "VIVIENDA", "HOGAR"],
        "variables": [
            "AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "UBIGEO",
            "TOTMIEHO", "MIEPERHO", "POBREZA", "INGHOG2D",
            "GASHOG2D", "FACTOR07",
        ],
    },
}

# Llaves indispensables. Las otras variables son opcionales y se reportan.
OBLIGATORIAS = {"AÑO", "CONGLOME", "VIVIENDA", "HOGAR"}


def detectar_codificacion(zip_actual, archivo_interno):
    """Elegir UTF-8 con BOM si funciona, o Latin-1 si no."""
    with zip_actual.open(archivo_interno) as f:
        muestra = f.read(65536)
    try:
        muestra.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin1"


def limpio(valor):
    """Limpiar espacios sin convertir identificadores a números."""
    return valor.strip() if valor is not None else ""


def procesar(modulo, config):
    ruta_zip = INEI / config["zip"]
    if not ruta_zip.is_file():
        raise FileNotFoundError(f"No existe: {ruta_zip}")

    columnas_salida = config["variables"]
    claves = config["llave"]
    ruta_salida = INTERIM / config["salida"]
    temporal = ruta_salida.with_suffix(".csv.parcial")

    total = 0
    filas_2025 = 0
    llaves_incompletas = 0
    llaves_duplicadas = 0
    claves_ya_vistas = set()
    no_vacios = Counter()
    edades_6_17 = 0
    edades_invalidas = 0

    print(f"\n========== ENAHO {modulo} ==========")
    print(f"Entrada: {ruta_zip.name}")

    try:
        with ZipFile(ruta_zip) as z:
            coincidencias = [
                nombre for nombre in z.namelist()
                if Path(nombre).name.lower() == config["csv"].lower()
            ]
            if not coincidencias:
                raise FileNotFoundError(
                    f"No encontré {config['csv']} dentro de {ruta_zip.name}"
                )
            interno = coincidencias[0]
            codificacion = detectar_codificacion(z, interno)

            with z.open(interno) as entrada_binaria:
                with io.TextIOWrapper(
                    entrada_binaria, encoding=codificacion, newline=""
                ) as entrada:
                    lector = csv.reader(entrada)
                    encabezado = next(lector)
                    columnas_originales = [
                        c.strip().lstrip("\ufeff").upper() for c in encabezado
                    ]
                    indice_original = {
                        nombre: i for i, nombre in enumerate(columnas_originales)
                    }
                    # Se usa exclusivamente _25, NO las columnas del panel
                    # CONGLOME y VIVIENDA sin sufijo.
                    posiciones = {
                        c: indice_original.get(f"{c}_25") for c in columnas_salida
                    }
                    faltantes = [c for c, i in posiciones.items() if i is None]
                    graves = sorted(OBLIGATORIAS.intersection(faltantes))
                    graves += [c for c in claves if c in faltantes and c not in graves]
                    if graves:
                        raise ValueError(
                            f"Faltan llaves críticas en {config['csv']}: {graves}"
                        )

                    posicion_anio = posiciones["AÑO"]
                    with open(
                        temporal, "w", newline="", encoding="utf-8-sig"
                    ) as salida:
                        escritor = csv.writer(salida)
                        escritor.writerow(columnas_salida)

                        for fila in lector:
                            total += 1
                            if posicion_anio >= len(fila):
                                continue
                            if limpio(fila[posicion_anio]) != "2025":
                                continue
                            filas_2025 += 1

                            registro = {
                                col: (limpio(fila[pos]) if pos is not None
                                      and pos < len(fila) else "")
                                for col, pos in posiciones.items()
                            }
                            escritor.writerow([registro[c] for c in columnas_salida])

                            for c, v in registro.items():
                                if v != "":
                                    no_vacios[c] += 1

                            llave = tuple(registro[k] for k in claves)
                            if any(v == "" for v in llave):
                                llaves_incompletas += 1
                            elif llave in claves_ya_vistas:
                                llaves_duplicadas += 1
                            else:
                                claves_ya_vistas.add(llave)

                            if modulo in ("200", "300"):
                                edad = registro.get("P208A", "")
                                if edad:
                                    try:
                                        numero = float(edad.replace(",", "."))
                                        if 6 <= numero <= 17 and numero.is_integer():
                                            edades_6_17 += 1
                                        elif numero < 0 or numero > 120:
                                            edades_invalidas += 1
                                    except ValueError:
                                        edades_invalidas += 1

                            if filas_2025 % 30000 == 0:
                                print(f"  Registros 2025 procesados: {filas_2025:,}")

        # Sólo sustituir el CSV final cuando el procesamiento se completó.
        os.replace(temporal, ruta_salida)
    except Exception:
        if temporal.exists():
            temporal.unlink()
        raise

    informe = [
        f"ENAHO {modulo}",
        f"CSV de panel: {config['csv']}",
        f"Codificación: {codificacion}",
        f"Filas originales leídas: {total:,}",
        f"Filas seleccionadas AÑO_25=2025: {filas_2025:,}",
        f"Columnas exportadas: {len(columnas_salida)}",
        f"Columnas opcionales ausentes: {faltantes}",
        f"Llave evaluada: {claves}",
        f"Filas con llave incompleta: {llaves_incompletas:,}",
        f"Llaves repetidas (NO se eliminan): {llaves_duplicadas:,}",
    ]
    if modulo in ("200", "300"):
        informe.extend([
            f"Filas con edades de 6 a 17 años: {edades_6_17:,}",
            f"Filas con edad fuera de 0–120 o no interpretable: {edades_invalidas:,}",
            "NOTA: las filas aún NO se filtran por edad; eso se hará tras el cruce.",
        ])
    informe.append("Valores presentes por variable:")
    for c in columnas_salida:
        porcentaje = 100 * no_vacios[c] / filas_2025 if filas_2025 else 0
        informe.append(f"  {c}: {no_vacios[c]:,}/{filas_2025:,} ({porcentaje:.1f}%)")
    informe.append(f"Salida: {ruta_salida.relative_to(RAIZ)}")

    print(f"Filas de 2025 guardadas: {filas_2025:,}")
    print(f"Llaves incompletas: {llaves_incompletas:,}")
    print(f"Llaves duplicadas: {llaves_duplicadas:,}")
    print(f"CSV creado: {ruta_salida.name}")
    return "\n".join(informe)


def main():
    informes = [
        "FASE 1 - LIMPIEZA ESTRUCTURAL ENAHO 2025",
        "Fuente: microdatos de PANEL 2021-2025 proporcionados para el examen.",
        "Acciones: seleccionar AÑO_25=2025, recortar espacios, renombrar _25.",
        "NO se aplican imputaciones, escalado, recodificación de categorías,",
        "eliminación de duplicados ni merges en esta etapa.",
        "",
    ]
    errores = 0
    for modulo, config in MODULOS.items():
        try:
            informes.append(procesar(modulo, config))
        except Exception as exc:
            errores += 1
            mensaje = f"ERROR procesando ENAHO {modulo}: {type(exc).__name__}: {exc}"
            print(mensaje)
            informes.append(mensaje)
        informes.append("\n" + "=" * 60 + "\n")

    destino = OUTPUTS / "reporte_limpieza_enaho_2025.txt"
    destino.write_text("\n".join(informes), encoding="utf-8")
    print("\n========== FIN DE LA PRÁCTICA ==========")
    print(f"Reporte: {destino}")
    print(f"Módulos con error: {errores}")
    print("No se han modificado los ZIP originales.")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
