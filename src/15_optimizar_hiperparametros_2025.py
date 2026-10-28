"""PASO 15 | Búsqueda reproducible de hiperparámetros SIN tocar el test.

Ejecutar desde la raíz del proyecto:
    python src/15_optimizar_hiperparametros_2025.py

Se reutilizan cargar() y preprocesador() del Paso 11 para que los
predictores, la limpieza, la semilla y los grupos sean EXACTAMENTE iguales.
Solo se usa el TRAIN de la partición agrupada anterior. No se evalúa el TEST,
no se sobrescribe el modelo anterior y no se sustituyen las evidencias SHAP.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent.parent
ORIGINAL = BASE / 'src' / '11_clasificacion_2025.py'
CARPETA = BASE / 'outputs' / 'optimizacion'
MODELO_SALIDA = BASE / 'outputs' / 'models' / 'clasificador_2025_optimizado_solo_train.joblib'
REPORTE = BASE / 'outputs' / 'reporte_optimizacion_2025.txt'


def cargar_programa_anterior():
    if not ORIGINAL.is_file():
        raise FileNotFoundError(f'No se encontro el programa 11: {ORIGINAL}')
    spec = importlib.util.spec_from_file_location('clasificacion_paso11', ORIGINAL)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main():
    parser = argparse.ArgumentParser(description='Optimización agrupada SOLO con train.')
    parser.add_argument('--iteraciones', type=int, default=12,
                        help='Configuraciones aleatorias a evaluar (defecto: 12).')
    args = parser.parse_args()
    if not 1 <= args.iteraciones <= 100:
        raise ValueError('El numero de iteraciones debe estar entre 1 y 100.')

    paso11 = cargar_programa_anterior()
    _, X, y, grupos, _ = paso11.cargar()
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                random_state=paso11.RANDOM_STATE)
    train_idx, test_idx = next(divisor.split(X, y, groups=grupos))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    grupos_train = grupos.iloc[train_idx]
    grupos_test = grupos.iloc[test_idx]

    if set(grupos_train) & set(grupos_test):
        raise AssertionError('Fuga de conglomerados entre train y test.')
    if y_train.nunique() != 2:
        raise ValueError('Train no contiene ambas clases.')
    if (len(train_idx), len(test_idx)) != (17690, 5928):
        raise RuntimeError('La partición difiere de la práctica 11. Revisar el CSV y el script original antes de comparar resultados.')

    CARPETA.mkdir(parents=True, exist_ok=True)
    MODELO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    REPORTE.parent.mkdir(parents=True, exist_ok=True)

    # Idéntico algoritmo y parámetros iniciales utilizados en el programa 11.
    original = paso11.modelos()['Hist_Gradient_Boosting']
    base = Pipeline([
        ('preprocesar', paso11.preprocesador()),
        ('modelo', original),
    ])
    cv = GroupKFold(n_splits=3)
    metricas = {
        'ap': 'average_precision',
        'roc_auc': 'roc_auc',
        'f1_macro': 'f1_macro',
        'balanced_accuracy': 'balanced_accuracy',
    }
    print('Comparando la configuración original mediante GroupKFold (solo entrenamiento)...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        anterior = cross_validate(
            base, X_train, y_train, groups=grupos_train,
            cv=cv, scoring=metricas, n_jobs=1, error_score='raise'
        )
    ap_original = float(np.mean(anterior['test_ap']))

    # Los parámetros se buscan DENTRO del Pipeline, sin contaminar los folds.
    estimador = Pipeline([
        ('preprocesar', paso11.preprocesador()),
        ('modelo', HistGradientBoostingClassifier(random_state=paso11.RANDOM_STATE)),
    ])
    espacio = {
        'modelo__learning_rate': [0.03, 0.05, 0.07, 0.1],
        'modelo__max_iter': [80, 110, 150, 200],
        'modelo__max_leaf_nodes': [7, 15, 31],
        'modelo__min_samples_leaf': [20, 40, 60],
        'modelo__l2_regularization': [0.0, 0.1, 1.0, 3.0],
    }
    busqueda = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=espacio,
        n_iter=args.iteraciones,
        scoring=metricas,
        refit='ap',
        cv=cv,
        random_state=paso11.RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        error_score='raise',
        return_train_score=False,
    )
    print('Buscando hiperparámetros con GroupKFold, sin utilizar el conjunto de prueba...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        busqueda.fit(X_train, y_train, groups=grupos_train)

    tabla = pd.DataFrame(busqueda.cv_results_)
    cols = [
        'rank_test_ap', 'mean_test_ap', 'std_test_ap',
        'mean_test_roc_auc', 'mean_test_f1_macro',
        'mean_test_balanced_accuracy', 'params',
    ]
    tabla = tabla[cols].sort_values('rank_test_ap')
    ruta_tabla = CARPETA / 'busqueda_hiperparametros_2025.csv'
    tabla.to_csv(ruta_tabla, index=False, encoding='utf-8-sig')

    mejor_ap = float(busqueda.best_score_)
    diferencia = mejor_ap - ap_original
    resumen = [
        'EXAMEN ENAHO–ESCALE 2025 | PASO 15 AJUSTE DE HIPERPARAMETROS',
        'Objetivo: P314A=2, NO USO INTERNET MES ANTERIOR.',
        'Se usa exactamente el mismo preprocesamiento y la misma particion del Paso 11.',
        f'Train: {len(train_idx):,} | Test reservado y NO utilizado: {len(test_idx):,}',
        f'Conglomerados de train: {grupos_train.nunique():,} | test: {grupos_test.nunique():,} | compartidos: 0',
        'Validacion: GroupKFold, 3 folds; todas las transformaciones dentro de Pipeline.',
        f'Configuraciones probadas: {args.iteraciones}; semilla: {paso11.RANDOM_STATE}',
        'Metrica de seleccion: average_precision (AP) en validacion cruzada agrupada.',
        f'AP de configuracion original en folds train: {ap_original:.6f}',
        f'Mejor AP de busqueda en folds train: {mejor_ap:.6f}',
        f'Diferencia AP (optimizada - original) en CV train: {diferencia:+.6f}',
        'Mejores parametros:',
        json.dumps(busqueda.best_params_, indent=2, ensure_ascii=False),
        '',
        f'Tabla de busqueda: {ruta_tabla}',
        f'Modelo nuevo ajustado SOLO con TRAIN: {MODELO_SALIDA}',
        '',
        'PRECAUCIONES:',
        '1. Este script NO evalua el conjunto TEST ni cambia los resultados publicados del Paso 11.',
        '2. El AP optimizado es una estimacion de CV usada para seleccionar parametros; puede ser optimista.',
        '3. Si la configuracion nueva va a reemplazar al modelo final, hace falta una evaluacion independiente.',
        '4. No sobrescribe clasificador_2025.joblib ni los graficos SHAP anteriores.',
        '5. Los resultados son del panel provisto (corte 2025); no representan automaticamente a todo Peru.',
    ]
    REPORTE.write_text('\n'.join(resumen) + '\n', encoding='utf-8')
    joblib.dump(busqueda.best_estimator_, MODELO_SALIDA)
    print('\n'.join(resumen), flush=True)


if __name__ == '__main__':
    main()
