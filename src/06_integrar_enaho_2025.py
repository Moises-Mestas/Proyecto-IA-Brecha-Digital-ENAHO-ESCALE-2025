"""Examen IA ENAHO-ESCALE 2025 | Fase 1: cruce relacional de ENAHO.

Lee exclusivamente los 4 CSV limpios generados en data/interim por el
programa 05. No modifica las bases originales y no realiza imputaciones.

Dependencia: pandas (python -m pip install pandas)
Ejecutar desde cualquier ruta: python src/06_integrar_enaho_2025.py
"""

from pathlib import Path
import sys
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INTERIM = BASE / "data" / "interim"
OUTPUTS = BASE / "outputs"

ARCHIVOS = {
    "200": INTERIM / "enaho_200_2025_limpio.csv",
    "300": INTERIM / "enaho_300_2025_limpio.csv",
    "100": INTERIM / "enaho_100_2025_limpio.csv",
    "SUMARIA": INTERIM / "enaho_sumaria_2025_limpio.csv",
}

CLAVE_HOGAR = ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR"]
CLAVE_PERSONA = CLAVE_HOGAR + ["CODPERSO"]

reporte = [
    "FASE 1 - INTEGRACIÓN RELACIONAL ENAHO 2025",
    "Fuente: cuatro módulos limpios de los paneles entregados por el docente.",
    "Regla: conservar las llaves como texto; no imputar ni alterar códigos.",
    "Uniones: LEFT, con validación de cardinalidad y auditoría de coincidencias.",
]


def informar(mensaje=""):
    linea = str(mensaje)
    reporte.append(linea)
    print(linea)


def leer_modulo(etiqueta, clave):
    ruta = ARCHIVOS[etiqueta]
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encuentra {ruta}. Ejecuta antes 05_limpiar_enaho_2025.py."
        )

    df = pd.read_csv(
        ruta,
        dtype=str,                 # Conservar '010101' y '01' como texto
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )
    df.columns = df.columns.str.strip().str.upper()

    if df.columns.duplicated().any():
        raise ValueError(f"{etiqueta}: hay nombres de columna repetidos")

    for columna in df.columns:
        df[columna] = df[columna].str.strip().replace("", pd.NA)

    faltantes = [c for c in clave if c not in df.columns]
    if faltantes:
        raise ValueError(f"{etiqueta}: faltan llaves obligatorias {faltantes}")

    if (df[clave].isna().any(axis=1)).any():
        raise ValueError(f"{etiqueta}: existen llaves incompletas; revisar datos")

    repetidos = int(df.duplicated(subset=clave).sum())
    if repetidos:
        raise ValueError(
            f"{etiqueta}: {repetidos} filas tienen llave repetida. "
            "Se detiene el cruce para evitar multiplicar estudiantes."
        )

    if not df["AÑO"].eq("2025").all():
        raise ValueError(f"{etiqueta}: se detectaron filas distintas de 2025")

    informar(f"Módulo {etiqueta}: {len(df):,} filas | {len(df.columns)} columnas "
            f"| duplicados: {repetidos}")
    return df


def prefijar_excepto_llaves(df, prefijo, clave):
    """Evita mezclar UBIGEO, P208A o FACTOR07 de tablas distintas."""
    nombres = {c: f"{prefijo}{c}" for c in df.columns if c not in clave}
    return df.rename(columns=nombres)


def unir(izquierda, derecha, clave, nombre, prefijo, validar):
    derecho = prefijar_excepto_llaves(derecha, prefijo, clave)
    antes = len(izquierda)
    resultado = izquierda.merge(
        derecho,
        on=clave,
        how="left",
        validate=validar,
        indicator=f"UNION_{nombre}",
        sort=False,
    )
    if len(resultado) != antes:
        raise AssertionError(f"Unión {nombre} cambió la cantidad de estudiantes")

    con_datos = int(resultado[f"UNION_{nombre}"].eq("both").sum())
    sin_datos = antes - con_datos
    informar(f"Unión {nombre}: {con_datos:,}/{antes:,} con coincidencia "
            f"({con_datos / antes * 100 if antes else 0:.1f} %) "
            f"| sin coincidencia: {sin_datos:,}")
    return resultado


def verificar_ubigeo(df):
    informar("\n--- Verificación de códigos geográficos ---")
    valido = df["UBIGEO"].str.fullmatch(r"\d{6}", na=False)
    informar(f"UBIGEO del módulo 200 con formato de 6 dígitos: "
            f"{int(valido.sum()):,}/{len(df):,}")

    for origen, indicador in [
        ("EDU", "UNION_300"),
        ("VIV", "UNION_100"),
        ("SUM", "UNION_SUMARIA"),
    ]:
        campo = f"{origen}_UBIGEO"
        if campo not in df.columns:
            continue
        mascara = df[indicador].eq("both") & df[campo].notna() & df["UBIGEO"].notna()
        diferencias = int((df.loc[mascara, campo] != df.loc[mascara, "UBIGEO"]).sum())
        informar(f"Diferencias UBIGEO 200 vs {origen}: {diferencias:,} "
                f"de {int(mascara.sum()):,} pares comparables")
        if diferencias:
            informar("  ADVERTENCIA: revisar antes de integrar MINEDU; no se corrigen automáticamente.")


def calidad_columnas(df):
    informar("\n--- Cobertura en estudiantes de 6 a 17 años ---")
    columnas = [
        "P207", "P208A", "EDU_P300A", "EDU_P306", "EDU_P307",
        "EDU_P308A", "EDU_P308B", "EDU_P314A", "VIV_P1121",
        "VIV_P1144", "SUM_POBREZA", "SUM_INGHOG2D",
        "SUM_GASHOG2D", "FACPOB07", "SUM_FACTOR07",
    ]
    total = len(df)
    for c in columnas:
        if c in df.columns:
            presentes = int(df[c].notna().sum())
            informar(f"{c}: {presentes:,}/{total:,} ({presentes / total * 100 if total else 0:.1f} %)")
        else:
            informar(f"{c}: NO ENCONTRADA")


def principal():
    informar("\n--- Lectura y control de duplicados ---")
    personas = leer_modulo("200", CLAVE_PERSONA)
    educacion = leer_modulo("300", CLAVE_PERSONA)
    vivienda = leer_modulo("100", CLAVE_HOGAR)
    sumaria = leer_modulo("SUMARIA", CLAVE_HOGAR)

    # El factor de hogar 'FACTOR07' está íntegramente vacío en ENAHO 200
    # según el reporte 05. No lo usamos como peso muestral.
    if "FACTOR07" in personas and personas["FACTOR07"].isna().all():
        personas = personas.drop(columns=["FACTOR07"])
        informar("FACTOR07 de ENAHO 200 se omitió por estar totalmente vacío.")
        informar("Se conserva FACPOB07; SUM_FACTOR07 se incorporará desde Sumaria.")

    informar("\n--- Primera unión: persona con educación ---")
    unidos = unir(
        personas, educacion, CLAVE_PERSONA,
        nombre="300", prefijo="EDU_", validar="one_to_one"
    )

    # Seleccionar alumnos en edad escolar usando la edad del módulo 200.
    edad = pd.to_numeric(unidos["P208A"], errors="coerce")
    seleccion = edad.between(6, 17)
    estudiantes = unidos.loc[seleccion].copy()
    estudiantes["EDAD"] = edad.loc[seleccion].astype("Int64")
    informar(f"Personas con edad 6-17 según módulo 200: {len(estudiantes):,}")
    if estudiantes.empty:
        raise ValueError("No se identificaron estudiantes de 6 a 17 años")

    informar("\n--- Agregando contexto de hogar ---")
    estudiantes = unir(
        estudiantes, vivienda, CLAVE_HOGAR,
        nombre="100", prefijo="VIV_", validar="many_to_one"
    )
    estudiantes = unir(
        estudiantes, sumaria, CLAVE_HOGAR,
        nombre="SUMARIA", prefijo="SUM_", validar="many_to_one"
    )

    informar("\n--- Controles finales ---")
    informar(f"Total de estudiantes conservados: {len(estudiantes):,}")
    informar(f"Filas duplicadas por llave de persona: "
            f"{int(estudiantes.duplicated(CLAVE_PERSONA).sum())}")

    # Comparar edad reportada por educación y personas si ambas existen.
    if "EDU_P208A" in estudiantes:
        otra_edad = pd.to_numeric(estudiantes["EDU_P208A"], errors="coerce")
        comparable = otra_edad.notna() & estudiantes["EDAD"].notna()
        distintas = int((otra_edad[comparable] != estudiantes.loc[comparable, "EDAD"]).sum())
        informar(f"Diferencias en edad módulo 200 vs 300: "
                f"{distintas:,} de {int(comparable.sum()):,} comparables")

    verificar_ubigeo(estudiantes)
    calidad_columnas(estudiantes)

    ruta_salida = INTERIM / "enaho_estudiantes_2025_integrado.csv"
    estudiantes.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    informar("\n--- Resultado ---")
    informar(f"CSV integrado de ENAHO: {ruta_salida}")
    informar(f"Filas: {len(estudiantes):,}; columnas: {len(estudiantes.columns)}")
    informar("Este NO es el dataset final del examen: falta agregar MINEDU por UBIGEO.")
    informar("No se imputaron valores, no se entrenaron modelos y no se usaron datos de prueba.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ruta_reporte = OUTPUTS / "reporte_integracion_enaho_2025.txt"
    ruta_reporte.write_text("\n".join(reporte) + "\n", encoding="utf-8")
    print(f"Reporte guardado en: {ruta_reporte}")


if __name__ == "__main__":
    try:
        principal()
    except (FileNotFoundError, ValueError, AssertionError, KeyError, pd.errors.ParserError) as error:
        print(f"\nERROR DE VALIDACIÓN: {error}")
        sys.exit(1)
