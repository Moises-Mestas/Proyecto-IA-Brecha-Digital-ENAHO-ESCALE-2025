"""EXAMEN IA ENAHO–ESCALE 2025 | FASE 2: CLUSTERING DESCRIPTIVO.

Trabaja EXCLUSIVAMENTE con el dataset maestro ya integrado por el programa 07.
Compara K-Means y Mezclas Gaussianas (GMM) para k=2..6, usando las mismas
variables y transformaciones. No interpreta códigos de ENAHO sin diccionario.

EJECUCIÓN:
  python -m pip install pandas numpy scikit-learn matplotlib
  python src/09_clustering_estudiantes_2025.py

NOTAS METODOLÓGICAS:
  - Datos suministrados: panel ENAHO 2021–2025, corte 2025 + ESCALE 2025.
  - Clustering descriptivo exploratorio (NO clasificador, NO prueba de causalidad).
  - No se utiliza factor de expansión: evitar afirmaciones de representatividad.
  - No usar los clusters ajustados con toda la muestra como variable de entrada
    para la clasificación supervisada: produciría fuga de información.
  - Consultar cuestionario/diccionarios para interpretar categorías 1/2/3.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42
BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / 'data' / 'processed' / 'dataset_enaho_escale_2025.csv'
OUT = BASE / 'outputs'
OUT_CL = OUT / 'clustering'
OUT_FIG = OUT / 'figures'
OUT_DATA = BASE / 'data' / 'processed' / 'estudiantes_2025_clusters_exploratorios.csv'
REPORTE = OUT / 'reporte_clustering_2025.txt'
COMPARACION = OUT_CL / 'comparacion_modelos_2025.csv'
PERFILES = OUT_CL / 'perfiles_clusters_2025.csv'
FIGURA = OUT_FIG / 'clustering_pca_2025.png'

# Variables numéricas con significado cuantitativo conocido en esta etapa.
NUMERICAS = [
    'EDAD', 'SUM_INGHOG2D', 'SUM_GASHOG2D',
    'MIN_RATIO_ALUMNOS_DOCENTE', 'MIN_PROP_PUBLICOS',
]

# Códigos de encuesta: variables CATEGÓRICAS, no magnitudes 1 < 2 < 3.
# El reporte mostrará sus códigos sin asignarles etiquetas semánticas.
CATEGORICAS = [
    'P207', 'SUM_POBREZA', 'VIV_P1121', 'VIV_P1144',
    'EDU_P306', 'EDU_P314A',
]
LLAVE = ['AÑO', 'CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO']


def decir(lineas, mensaje=''):
    mensaje = str(mensaje)
    print(mensaje)
    lineas.append(mensaje)


def leer_maestro():
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f'No se encontró {ENTRADA}. Ejecuta antes 07 corregido y 08.'
        )
    df = pd.read_csv(
        ENTRADA, dtype='string', encoding='utf-8-sig',
        keep_default_na=False, low_memory=False,
    )
    df.columns = df.columns.str.strip().str.upper()
    for col in df.columns:
        df[col] = df[col].str.strip().replace('', pd.NA)
    requeridas = LLAVE + ['UBIGEO'] + NUMERICAS + CATEGORICAS
    ausentes = [c for c in requeridas if c not in df.columns]
    if ausentes:
        raise ValueError(f'Faltan columnas necesarias: {ausentes}')
    if df[LLAVE].isna().any(axis=1).any() or df.duplicated(LLAVE).any():
        raise ValueError('Hay identificadores incompletos o duplicados.')
    if not df['AÑO'].eq('2025').all():
        raise ValueError('Se detectaron registros distintos de 2025.')
    edades = pd.to_numeric(df['EDAD'], errors='coerce')
    if not edades.between(6, 17).all():
        raise ValueError('Hay edades fuera del rango 6–17.')
    return df


def armar_variables(df, lineas):
    x = pd.DataFrame(index=df.index)
    for col in NUMERICAS:
        numero = pd.to_numeric(
            df[col].str.replace(',', '.', regex=False), errors='coerce'
        )
        # Escala de logaritmo hiperbólico: comprime valores económicos extremos;
        # no pierde el signo de posibles valores negativos.
        if col in {'SUM_INGHOG2D', 'SUM_GASHOG2D'}:
            numero = np.arcsinh(numero)
        x[col] = numero.astype('float64')
        decir(lineas, f'  Numérica {col}: {int(numero.notna().sum()):,}/{len(df):,} válidos')

    for col in CATEGORICAS:
        valor = df[col].fillna('SIN_DATO').astype(str)
        x[col] = valor
        decir(lineas, f'  Categórica {col}: {int((valor == "SIN_DATO").sum()):,} sin dato')

    # Evitar introducir variables constantes o códigos personales/geográficos.
    sin_info = [c for c in NUMERICAS if x[c].notna().sum() == 0]
    if sin_info:
        raise ValueError(f'Variables numéricas sin datos: {sin_info}')
    return x


def transformar(x):
    numericas = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('estandarizar', StandardScaler()),
    ])
    categoricas = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    proceso = ColumnTransformer([
        ('numericas', numericas, NUMERICAS),
        ('categoricas', categoricas, CATEGORICAS),
    ], remainder='drop', sparse_threshold=0)
    matriz = proceso.fit_transform(x).astype('float64', copy=False)
    if not np.isfinite(matriz).all():
        raise ValueError('La matriz contiene NaN o infinito tras transformaciones.')
    return matriz


def evaluar(matriz, etiquetas):
    n_clusters = len(np.unique(etiquetas))
    if n_clusters < 2:
        return None
    cuenta = pd.Series(etiquetas).value_counts()
    # El coste cuadrático de silhouette se limita con muestra de 1800 personas.
    silhouette = silhouette_score(
        matriz, etiquetas, sample_size=min(1800, len(matriz)),
        random_state=SEMILLA,
    )
    return {
        'silhouette_muestra_1800': float(silhouette),
        'davies_bouldin': float(davies_bouldin_score(matriz, etiquetas)),
        'calinski_harabasz': float(calinski_harabasz_score(matriz, etiquetas)),
        'cluster_minimo_n': int(cuenta.min()),
        'cluster_minimo_pct': round(100 * cuenta.min() / len(etiquetas), 2),
    }


def perfilar(df, etiquetas, metodo, k):
    tabla = df.copy()
    tabla['CLUSTER'] = etiquetas
    perfil = []
    for cluster, grupo in tabla.groupby('CLUSTER', sort=True):
        fila = {
            'metodo': metodo,
            'k': k,
            'cluster': int(cluster),
            'cantidad': len(grupo),
            'porcentaje': round(100 * len(grupo) / len(tabla), 2),
        }
        for col in NUMERICAS:
            valores = pd.to_numeric(
                grupo[col].str.replace(',', '.', regex=False), errors='coerce'
            )
            fila[f'{col}_MEDIANA'] = float(valores.median()) if valores.notna().any() else np.nan
        for col in CATEGORICAS:
            modo = grupo[col].fillna('SIN_DATO').mode(dropna=False)
            fila[f'{col}_CODIGO_MAS_FRECUENTE'] = str(modo.iloc[0]) if len(modo) else ''
            fila[f'{col}_SIN_DATO_PCT'] = round(100 * grupo[col].isna().mean(), 2)
        perfil.append(fila)
    return pd.DataFrame(perfil)


def principal():
    for carpeta in [OUT, OUT_CL, OUT_FIG, OUT_DATA.parent]:
        carpeta.mkdir(parents=True, exist_ok=True)

    lineas = []
    decir(lineas, 'FASE 2 - CLUSTERING EXPLORATORIO ENAHO–ESCALE 2025')
    decir(lineas, 'Origen: únicamente CSV maestro del examen; NO se modifican originales.')
    df = leer_maestro()
    decir(lineas, f'Dataset: {len(df):,} estudiantes x {len(df.columns)} columnas')
    decir(lineas, 'No se utilizan UBIGEO ni llaves como predictores del clustering.')
    decir(lineas, '\n--- Variables seleccionadas ---')
    x = armar_variables(df, lineas)
    matriz = transformar(x)
    decir(lineas, f'Matriz después de One-Hot y escalado: {matriz.shape}')

    decir(lineas, '\n--- Comparación de modelos: k=2..6 ---')
    resultados = []
    etiquetas_guardadas = {}
    for metodo in ['KMEANS', 'GMM']:
        for k in range(2, 7):
            if metodo == 'KMEANS':
                modelo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10)
                etiquetas = modelo.fit_predict(matriz)
            else:
                modelo = GaussianMixture(
                    n_components=k, covariance_type='diag',
                    random_state=SEMILLA, n_init=2,
                    max_iter=200, reg_covar=1e-5,
                )
                with warnings.catch_warnings(record=True) as avisos:
                    warnings.simplefilter('always')
                    etiquetas = modelo.fit_predict(matriz)
                for aviso in avisos:
                    decir(lineas, f'  AVISO GMM k={k}: {aviso.message}')

            metricas = evaluar(matriz, etiquetas)
            if metricas is None:
                decir(lineas, f'{metodo} k={k}: produjo un único cluster. Se omite.')
                continue
            resultado = {'algoritmo': metodo, 'k': k, **metricas}
            resultados.append(resultado)
            etiquetas_guardadas[(metodo, k)] = etiquetas
            decir(lineas,
                  f'{metodo:7} k={k} | silueta={metricas["silhouette_muestra_1800"]:.4f}'
                  f' | DB={metricas["davies_bouldin"]:.3f}'
                  f' | CH={metricas["calinski_harabasz"]:.1f}'
                  f' | cluster menor={metricas["cluster_minimo_pct"]:.2f}%')

    if not resultados:
        raise RuntimeError('No se pudo obtener al menos dos grupos con ningún modelo.')
    comparacion = pd.DataFrame(resultados)
    comparacion.sort_values(
        ['silhouette_muestra_1800', 'davies_bouldin'],
        ascending=[False, True], inplace=True,
    )
    comparacion.to_csv(COMPARACION, index=False, encoding='utf-8-sig')

    # Elegir por mayor silhouette, descartando soluciones con grupos <2% si hay.
    admisibles = comparacion.loc[comparacion['cluster_minimo_pct'] >= 2]
    candidato = admisibles.iloc[0] if not admisibles.empty else comparacion.iloc[0]
    mejor_alg, mejor_k = str(candidato['algoritmo']), int(candidato['k'])
    mejor_labels = etiquetas_guardadas[(mejor_alg, mejor_k)]
    decir(lineas, f'\nMejor solución exploratoria: {mejor_alg} k={mejor_k} '
                  f'(silueta={candidato["silhouette_muestra_1800"]:.4f})')
    if admisibles.empty:
        decir(lineas, 'ADVERTENCIA: toda solución contiene al menos un grupo menor al 2%.')

    df['CLUSTER_EXPLORATORIO'] = mejor_labels
    df.to_csv(OUT_DATA, index=False, encoding='utf-8-sig')
    perfil = perfilar(df.drop(columns=['CLUSTER_EXPLORATORIO']), mejor_labels,
                      mejor_alg, mejor_k)
    perfil.to_csv(PERFILES, index=False, encoding='utf-8-sig')

    # Visualización PCA. El PCA solo proyecta para observar, no se usa para
    # generar los clusters ni calcular las métricas anteriores.
    puntos = PCA(n_components=2, random_state=SEMILLA).fit_transform(matriz)
    rng = np.random.default_rng(SEMILLA)
    indices = rng.choice(len(df), size=min(len(df), 3000), replace=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in np.unique(mejor_labels):
        sel = indices[mejor_labels[indices] == cluster]
        ax.scatter(puntos[sel, 0], puntos[sel, 1], s=7, alpha=.55,
                   label=f'Grupo {cluster} (n={int((mejor_labels==cluster).sum())})')
    ax.set_title(f'Clustering exploratorio: {mejor_alg} (k={mejor_k})')
    ax.set_xlabel('Componente principal 1')
    ax.set_ylabel('Componente principal 2')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=160)
    plt.close(fig)

    decir(lineas, '\n--- Tamaño de los grupos del modelo seleccionado ---')
    for _, f in perfil.iterrows():
        decir(lineas, f'  Cluster {int(f["cluster"])}: {int(f["cantidad"]):,}'
                      f' estudiantes ({f["porcentaje"]:.2f}%)')
    decir(lineas, '\n--- Archivos creados ---')
    for ruta in [COMPARACION, PERFILES, FIGURA, OUT_DATA]:
        decir(lineas, f'  {ruta}')
    decir(lineas, '\nADVERTENCIAS PARA EL INFORME:')
    decir(lineas, '- Los códigos de ENAHO NO se interpretan sin diccionario.')
    decir(lineas, '- Imputación mediana y escalado se usan solo para clustering exploratorio.')
    decir(lineas, '- No usar CLUSTER_EXPLORATORIO como predictor en clasificación supervisada.')
    decir(lineas, '- No inferir causalidad ni representatividad poblacional sin diseño muestral.')
    decir(lineas, '- Se usan datos del panel 2021–2025 aportado, corte 2025.')
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    decir([], f'\nReporte guardado en: {REPORTE}')


if __name__ == '__main__':
    principal()
