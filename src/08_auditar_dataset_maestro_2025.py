"""Examen IA ENAHO–ESCALE 2025 | Fase 1, control previo al modelado.

Revisa el dataset maestro generado por el programa 07 CORREGIDO.
No modifica el CSV maestro, no rellena datos faltantes, no recodifica códigos
INEI ni entrena modelos. Produce reporte y diccionario de disponibilidad.

Requiere: pandas
Ejecutar: python src/08_auditar_dataset_maestro_2025.py
"""

from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / 'data' / 'processed' / 'dataset_enaho_escale_2025.csv'
SALIDAS = RAIZ / 'outputs'
REPORTE = SALIDAS / 'reporte_calidad_dataset_2025.txt'
DICCIONARIO = SALIDAS / 'diccionario_columnas_maestro_2025.csv'

CLAVE = ['AÑO', 'CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO']
VARIABLES_CLAVE = [
    'EDAD', 'P207', 'EDU_P300A', 'EDU_P306', 'EDU_P307', 'EDU_P308A',
    'EDU_P308B', 'EDU_P314A', 'EDU_P314B_1', 'EDU_P316_1',
    'VIV_P1121', 'VIV_P1144', 'SUM_POBREZA', 'SUM_INGHOG2D',
    'SUM_GASHOG2D', 'MIN_RATIO_ALUMNOS_DOCENTE', 'MIN_PROP_PUBLICOS',
]
CODIGOS_PARA_REVISAR = [
    'P207', 'EDU_P300A', 'EDU_P306', 'EDU_P307', 'EDU_P308A',
    'EDU_P308B', 'EDU_P314A', 'EDU_P314B_1', 'EDU_P316_1',
    'VIV_P1121', 'VIV_P1144', 'SUM_POBREZA',
]

lineas = []

def mostrar(mensaje=''):
    mensaje = str(mensaje)
    print(mensaje)
    lineas.append(mensaje)


def origen(columna):
    if columna in CLAVE or columna in ('UBIGEO', 'EDAD'):
        return 'Identificador / persona ENAHO 200'
    if columna.startswith('EDU_'):
        return 'ENAHO 300 - Educación'
    if columna.startswith('VIV_'):
        return 'ENAHO 100 - Vivienda'
    if columna.startswith('SUM_'):
        return 'ENAHO Sumaria - Hogar'
    if columna.startswith('MIN_'):
        return 'MINEDU - Indicador distrital'
    if columna.startswith('UNION_'):
        return 'Control de integración'
    return 'ENAHO 200 / otras variables'


def a_numero(serie):
    return pd.to_numeric(
        serie.astype('string').str.replace(',', '.', regex=False),
        errors='coerce',
    )


def verificar_proporcion(df, nombre):
    if nombre not in df:
        mostrar(f'  {nombre}: AUSENTE')
        return
    valores = a_numero(df[nombre])
    presente = int(valores.notna().sum())
    fuera = int((valores.notna() & ~valores.between(0, 1)).sum())
    mostrar(f'  {nombre}: {presente:,}/{len(df):,} numericos; fuera del rango [0,1]: {fuera:,}')


def main():
    if not ENTRADA.is_file():
        raise FileNotFoundError(f'No existe {ENTRADA}. Ejecuta antes el programa 07 corregido.')

    SALIDAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        ENTRADA, dtype='string', encoding='utf-8-sig',
        keep_default_na=False, low_memory=False,
    )
    df.columns = df.columns.str.strip().str.upper()
    if df.columns.duplicated().any():
        raise ValueError('Hay nombres de columnas duplicados.')
    # Vacíos/espacios se convierten en valores ausentes solo en memoria.
    for col in df.columns:
        df[col] = df[col].str.strip().replace('', pd.NA)

    mostrar('FASE 1 - AUDITORIA DEL DATASET MAESTRO ENAHO–ESCALE 2025')
    mostrar('Datos del examen proporcionados por el docente. SIN entrenamiento ni imputacion.')
    mostrar(f'Filas: {len(df):,} | Columnas: {len(df.columns):,}')
    mostrar(f'Ruta: {ENTRADA}')

    mostrar('\n--- Validacion de identificadores ---')
    faltantes = [x for x in CLAVE if x not in df]
    if faltantes:
        raise ValueError(f'Faltan identificadores: {faltantes}')
    mostrar(f'Llaves incompletas: {int(df[CLAVE].isna().any(axis=1).sum()):,}')
    duplicados = int(df.duplicated(subset=CLAVE).sum())
    mostrar(f'Llaves persona duplicadas: {duplicados:,}')
    mostrar(f'Años distintos: {df["AÑO"].value_counts(dropna=False).to_dict()}')
    if 'UBIGEO' in df:
        formato = df['UBIGEO'].str.fullmatch(r'\d{6}', na=False)
        mostrar(f'UBIGEO de 6 digitos: {int(formato.sum()):,}/{len(df):,}')
    if 'EDAD' in df:
        edades = a_numero(df['EDAD'])
        mostrar(f'Edades validas de 6 a 17: {int(edades.between(6,17).sum()):,}/{len(df):,}')
        mostrar(f'Edades faltantes/no numericas: {int(edades.isna().sum()):,}')

    mostrar('\n--- Coincidencias de modulos ---')
    for col in ['UNION_300', 'UNION_100', 'UNION_SUMARIA']:
        if col in df:
            mostrar(f'{col}: {df[col].fillna("(vacio)").value_counts().to_dict()}')

    mostrar('\n--- Integridad de indicadores MINEDU ---')
    verificar_proporcion(df, 'MIN_PROP_PUBLICOS')
    verificar_proporcion(df, 'MIN_PROP_PRIVADOS')
    if {'MIN_PROP_PUBLICOS', 'MIN_PROP_PRIVADOS'}.issubset(df.columns):
        pub, priv = a_numero(df['MIN_PROP_PUBLICOS']), a_numero(df['MIN_PROP_PRIVADOS'])
        ambos = pub.notna() & priv.notna()
        incorrectos = int(((pub[ambos] + priv[ambos] - 1).abs() > 0.00001).sum())
        mostrar(f'Proporciones publica+privada distintas de 1: {incorrectos:,} de {int(ambos.sum()):,}')
    if 'MIN_RATIO_ALUMNOS_DOCENTE' in df:
        ratio = a_numero(df['MIN_RATIO_ALUMNOS_DOCENTE'])
        mostrar(f'Ratio alumnos/docente disponible: {int(ratio.notna().sum()):,}/{len(df):,}')
        mostrar(f'Ratios negativos: {int((ratio < 0).sum()):,}')
        if ratio.notna().any():
            mostrar(f'Ratio minimo/mediana/maximo: {ratio.min():.2f} / {ratio.median():.2f} / {ratio.max():.2f}')
    if {'MIN_COLEGIOS_PUBLICOS', 'MIN_COLEGIOS_PRIVADOS', 'MIN_GESTION_CONOCIDA'}.issubset(df.columns):
        pub = a_numero(df['MIN_COLEGIOS_PUBLICOS'])
        priv = a_numero(df['MIN_COLEGIOS_PRIVADOS'])
        total = a_numero(df['MIN_GESTION_CONOCIDA'])
        verificar = pub.notna() & priv.notna() & total.notna()
        mal = int(((pub[verificar] + priv[verificar] - total[verificar]).abs() > 0.00001).sum())
        mostrar(f'Recuentos publica+privada distintos del total conocido: {mal:,}')

    mostrar('\n--- Cobertura de variables prioritarias ---')
    for col in VARIABLES_CLAVE:
        if col in df:
            mostrar(f'  {col}: {int(df[col].notna().sum()):,}/{len(df):,} ({df[col].notna().mean():.1%})')
        else:
            mostrar(f'  {col}: NO EXISTE')

    mostrar('\n--- Valores observados en variables categóricas (SIN interpretar codigos) ---')
    for col in CODIGOS_PARA_REVISAR:
        if col not in df:
            continue
        frecuencias = df[col].fillna('(vacio)').value_counts().head(12)
        mostrar(f'{col}: {frecuencias.to_dict()}')

    resumen = pd.DataFrame([{
        'variable': col,
        'origen': origen(col),
        'tipo_lectura': 'texto (sin recodificar)',
        'filas': len(df),
        'no_vacios': int(df[col].notna().sum()),
        'porcentaje_no_vacio': round(df[col].notna().mean() * 100, 2),
        'valores_distintos_no_vacios': int(df[col].nunique(dropna=True)),
    } for col in df.columns])
    resumen.to_csv(DICCIONARIO, index=False, encoding='utf-8-sig')
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print('\n--- Archivos creados ---')
    print(REPORTE)
    print(DICCIONARIO)
    print('Importante: no se definieron etiquetas ni se interpretaron codigos de encuesta.')


if __name__ == '__main__':
    main()
