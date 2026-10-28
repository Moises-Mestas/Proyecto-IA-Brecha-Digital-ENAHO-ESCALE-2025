"""EXAMEN IA ENAHO-ESCALE 2025 | PASO 10: PERFILES Y VARIABLE OBJETIVO.

Usa EXCLUSIVAMENTE el CSV con clusters generado en la practica 09.
No modifica las bases originales ni entrena un clasificador.

EJECUTAR:
  python src/10_analizar_clusters_y_objetivo_2025.py

SALIDAS:
  outputs/reporte_perfiles_y_objetivo_2025.txt
  outputs/clustering/distribucion_categorias_por_cluster_2025.csv
  outputs/clustering/resumen_numerico_clusters_2025.csv
  outputs/clustering/cruce_p306_p314a_2025.csv

IMPORTANTE: Los codigos de ENAHO se informan sin interpretar etiquetas.
Consultar los cuestionarios/diccionarios entregados para decidir la variable y.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / 'data' / 'processed' / 'estudiantes_2025_clusters_exploratorios.csv'
DIR_SALIDA = BASE / 'outputs' / 'clustering'
REPORTE = BASE / 'outputs' / 'reporte_perfiles_y_objetivo_2025.txt'
SALIDA_CATEGORIAS = DIR_SALIDA / 'distribucion_categorias_por_cluster_2025.csv'
SALIDA_NUMERICOS = DIR_SALIDA / 'resumen_numerico_clusters_2025.csv'
SALIDA_CRUCE = DIR_SALIDA / 'cruce_p306_p314a_2025.csv'

LLAVES = ['AÑO', 'CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO']
CLUSTER = 'CLUSTER_EXPLORATORIO'
CATEGORICAS = [
    'P207', 'SUM_POBREZA', 'VIV_P1121', 'VIV_P1144',
    'EDU_P306', 'EDU_P314A', 'EDU_P307', 'EDU_P300A',
    'EDU_P308A', 'EDU_P308B',
]
NUMERICAS = [
    'EDAD', 'SUM_INGHOG2D', 'SUM_GASHOG2D',
    'MIN_RATIO_ALUMNOS_DOCENTE', 'MIN_PROP_PUBLICOS',
]
CANDIDATAS_Y = ['EDU_P306', 'EDU_P314A']


def registrar(lineas, mensaje=''):
    print(mensaje)
    lineas.append(str(mensaje))


def cargar():
    if not ENTRADA.exists():
        raise FileNotFoundError(
            'No se encontro el archivo de la practica 09:\n'
            f'{ENTRADA}\nEjecuta antes 09_clustering_estudiantes_2025.py'
        )
    df = pd.read_csv(
        ENTRADA, dtype='string', encoding='utf-8-sig',
        keep_default_na=False, low_memory=False,
    )
    df.columns = df.columns.str.strip().str.upper()
    for columna in df.columns:
        df[columna] = df[columna].str.strip().replace('', pd.NA)

    indispensables = LLAVES + [CLUSTER, 'EDAD'] + CANDIDATAS_Y
    faltan = [c for c in indispensables if c not in df.columns]
    if faltan:
        raise ValueError(f'Columnas requeridas ausentes: {faltan}')
    if df[LLAVES].isna().any(axis=1).any() or df.duplicated(LLAVES).any():
        raise ValueError('Llaves duplicadas o incompletas: verificar practica 09.')
    if df[CLUSTER].isna().any():
        raise ValueError('Hay estudiantes sin cluster asignado.')
    if not df['AÑO'].eq('2025').all():
        raise ValueError('La base contiene anios distintos de 2025.')
    edades = pd.to_numeric(df['EDAD'], errors='coerce')
    if not edades.between(6, 17).all():
        raise ValueError('Hay edades no validas (se esperaban 6 a 17).')
    return df


def frecuencia_categorias(df):
    resultados = []
    for variable in CATEGORICAS:
        if variable not in df.columns:
            continue
        for cluster, grupo in df.groupby(CLUSTER, sort=True):
            serie = grupo[variable].fillna('SIN_DATO')
            for codigo, cantidad in serie.value_counts(dropna=False).items():
                resultados.append({
                    'cluster': str(cluster),
                    'variable': variable,
                    'codigo_sin_interpretar': str(codigo),
                    'cantidad': int(cantidad),
                    'porcentaje_dentro_cluster': round(100 * cantidad / len(grupo), 2),
                    'total_cluster': len(grupo),
                })
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(SALIDA_CATEGORIAS, index=False, encoding='utf-8-sig')
    return tabla


def resumen_numerico(df):
    filas = []
    for variable in NUMERICAS:
        if variable not in df.columns:
            continue
        for cluster, grupo in df.groupby(CLUSTER, sort=True):
            numerico = pd.to_numeric(
                grupo[variable].str.replace(',', '.', regex=False),
                errors='coerce',
            ).dropna()
            filas.append({
                'cluster': str(cluster), 'variable': variable,
                'n_con_valor': int(len(numerico)),
                'q25': float(numerico.quantile(0.25)) if len(numerico) else None,
                'mediana': float(numerico.median()) if len(numerico) else None,
                'q75': float(numerico.quantile(0.75)) if len(numerico) else None,
            })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA_NUMERICOS, index=False, encoding='utf-8-sig')
    return tabla


def objetivo_provisional(df, lineas):
    registrar(lineas, '\n--- Candidatas a variable objetivo (CODIGOS SIN SIGNIFICADO ASIGNADO) ---')
    for variable in CANDIDATAS_Y:
        datos = df[variable].fillna('SIN_DATO')
        registrar(lineas, f'\n{variable} | valores y frecuencias:')
        for valor, n in datos.value_counts().items():
            registrar(lineas, f'  Codigo {valor}: {n:,} ({100*n/len(df):.2f}%)')
        codigos_validos = datos[datos != 'SIN_DATO']
        registrar(lineas, f'  Registros con respuesta: {len(codigos_validos):,}')
        registrar(lineas, f'  Codigos distintos con respuesta: {codigos_validos.nunique()}')
        if codigos_validos.nunique() != 2:
            registrar(lineas, '  AVISO: no es automaticamente una variable binaria.')
    # Solo conteos de combinaciones de codigos, no inferencias ni etiquetas.
    tab = pd.crosstab(
        df['EDU_P306'].fillna('SIN_DATO'),
        df['EDU_P314A'].fillna('SIN_DATO'),
        dropna=False,
    )
    tab.index.name = 'P306_CODIGO'
    tab.columns.name = 'P314A_CODIGO'
    tab.to_csv(SALIDA_CRUCE, encoding='utf-8-sig')
    registrar(lineas, '\nCruce de codigos P306 vs P314A (conteos, sin interpretar):')
    registrar(lineas, tab.to_string())


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    df = cargar()
    lineas = []
    registrar(lineas, 'EXAMEN IA ENAHO-ESCALE 2025 | PASO 10')
    registrar(lineas, 'Profundizacion de perfiles y revision de posibles objetivos de clasificacion')
    registrar(lineas, f'Estudiantes: {len(df):,} | columnas: {len(df.columns)}')
    registrar(lineas, '\n--- Grupos del clustering exploratorio ---')
    registrar(lineas, df[CLUSTER].value_counts().sort_index().to_string())

    cats = frecuencia_categorias(df)
    nums = resumen_numerico(df)
    registrar(lineas, '\n--- Perfil numerico por cluster (medianas) ---')
    medianas = nums.pivot(index='variable', columns='cluster', values='mediana')
    registrar(lineas, medianas.round(2).to_string())

    registrar(lineas, '\n--- Distribuciones completas de dos variables candidatas ---')
    for v in CANDIDATAS_Y:
        sub = cats[cats['variable'] == v]
        if not sub.empty:
            vista = sub.pivot(index='codigo_sin_interpretar', columns='cluster',
                              values='porcentaje_dentro_cluster').fillna(0)
            registrar(lineas, f'\n{v}, porcentaje DENTRO DE CADA CLUSTER:')
            registrar(lineas, vista.round(2).to_string())

    objetivo_provisional(df, lineas)
    registrar(lineas, '\n--- Recomendaciones metodologicas ---')
    registrar(lineas, '1) NO asignar significados a codigos 1/2 hasta leer los cuestionarios ENAHO.')
    registrar(lineas, '2) Los perfiles de clustering son descriptivos; no prueban causalidad.')
    registrar(lineas, '3) Para clasificacion: no usar CLUSTER_EXPLORATORIO como predictor.')
    registrar(lineas, '4) No usar como X variables que definen Y o son consecuencia de Y.')
    registrar(lineas, '5) Validar entrenamiento/prueba por CONGLOME; ajustar transformaciones solo en train.')
    registrar(lineas, '6) No declarar resultados representativos de la poblacion sin pesos y diseno muestral.')
    registrar(lineas, '\nArchivos generados:')
    for p in [SALIDA_CATEGORIAS, SALIDA_NUMERICOS, SALIDA_CRUCE]:
        registrar(lineas, f'  {p}')
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print(f'\nReporte guardado: {REPORTE}')


if __name__ == '__main__':
    main()
