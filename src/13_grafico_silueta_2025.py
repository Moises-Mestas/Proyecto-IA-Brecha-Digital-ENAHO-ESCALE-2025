"""Complemento a la practica 09: grafico de silueta del clustering ya generado.

Ejecucion (desde raiz del proyecto):
    python src/13_grafico_silueta_2025.py

Reutiliza exactamente las variables y transformaciones de la practica 09.
Analiza una muestra reproducible de 1,200 estudiantes para controlar memoria.
No modifica el dataset ni vuelve a asignar clusters.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

BASE = Path(__file__).resolve().parent.parent
RUTA_09 = BASE / 'src' / '09_clustering_estudiantes_2025.py'
CLUSTERS = BASE / 'data' / 'processed' / 'estudiantes_2025_clusters_exploratorios.csv'
FIG = BASE / 'outputs' / 'figures' / 'clustering_silueta_2025.png'
RESUMEN = BASE / 'outputs' / 'clustering' / 'resumen_silueta_2025.txt'
SEMILLA = 42
MUESTRA = 1200


def main():
    if not RUTA_09.is_file() or not CLUSTERS.is_file():
        raise FileNotFoundError('Se requieren la practica 09 y su CSV de clusters.')

    spec = importlib.util.spec_from_file_location('clustering_practica_09', RUTA_09)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    # La practica 09 guardo el mismo orden de las filas originales en este CSV.
    df = pd.read_csv(CLUSTERS, dtype='string', encoding='utf-8-sig',
                     keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    if 'CLUSTER_EXPLORATORIO' not in df:
        raise ValueError('No existe la columna CLUSTER_EXPLORATORIO.')
    if len(df) < 3:
        raise ValueError('Muy pocos estudiantes para calcular la silueta.')
    if df['CLUSTER_EXPLORATORIO'].eq('').any():
        raise ValueError('Hay estudiantes sin cluster asignado.')
    for c in df.columns:
        df[c] = df[c].str.strip().replace('', pd.NA)
    etiquetas = df['CLUSTER_EXPLORATORIO'].astype(int).to_numpy()
    if len(np.unique(etiquetas)) < 2:
        raise ValueError('Se requieren al menos dos clusters.')

    # Reproduce el tratamiento exacto usado en el clustering exploratorio.
    x = modulo.armar_variables(df, [])
    matriz = modulo.transformar(x)

    rng = np.random.default_rng(SEMILLA)
    indices = np.sort(rng.choice(len(df), size=min(MUESTRA, len(df)), replace=False))
    x_sub = matriz[indices]
    y_sub = etiquetas[indices]
    n_por_grupo = pd.Series(y_sub).value_counts()
    if n_por_grupo.min() < 2 or len(n_por_grupo) < 2:
        raise ValueError('La muestra no permite calcular silueta por cluster.')

    valores = silhouette_samples(x_sub, y_sub, metric='euclidean')
    promedio = float(np.mean(valores))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cursor = 10
    for grupo in sorted(np.unique(y_sub)):
        valores_grupo = np.sort(valores[y_sub == grupo])
        fin = cursor + len(valores_grupo)
        ax.fill_betweenx(np.arange(cursor, fin), 0, valores_grupo,
                         alpha=0.72, label=f'Cluster {grupo} (n={len(valores_grupo)})')
        ax.text(-0.09, cursor + len(valores_grupo) / 2, str(grupo), fontsize=10)
        cursor = fin + 10
    ax.axvline(promedio, linestyle='--', linewidth=1.5,
               label=f'Media en muestra = {promedio:.3f}')
    ax.set_xlim(min(-0.2, float(np.min(valores)) - 0.05), 1.0)
    ax.set_xlabel('Coeficiente de silueta individual')
    ax.set_ylabel('Estudiantes agrupados por cluster')
    ax.set_yticks([])
    ax.set_title('Perfil de silueta del clustering K-Means k=2 (muestra)')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=170, bbox_inches='tight')
    plt.close(fig)
    texto = (
        'GRAFICO DE SILUETA - EVIDENCIA COMPLEMENTARIA DEL EXAMEN\n'
        f'Filas originales: {len(df)}\n'
        f'Muestra sin reemplazo, semilla {SEMILLA}: {len(indices)}\n'
        f'Coeficiente promedio calculado sobre la muestra: {promedio:.6f}\n'
        'AVISO: este valor puede diferir de 0.2179 del reporte 09\n'
        'porque se recalcula con otra muestra y sus distancias internas.\n'
        'La figura es descriptiva; no es prueba de causalidad.\n'
        f'Figura: {FIG}\n'
    )
    RESUMEN.write_text(texto, encoding='utf-8')
    print(texto)


if __name__ == '__main__':
    main()
