"""EXAMEN IA ENAHO–ESCALE 2025 | PRACTICA 11 - CLASIFICACION.

Ejecutar desde la raiz del proyecto:
    python src/11_clasificacion_2025.py

Usa SOLO data/processed/dataset_enaho_escale_2025.csv, preparado
con los archivos proporcionados por el docente.

IMPORTANTE: OBJETIVO = EDU_P314A, positivo = CODIGO 2.
Se trata de una clasificacion TECNICA DEL CODIGO DE ENCUESTA, SIN
interpretar como 'usa/no usa internet' hasta verificar el cuestionario.

Previene fuga de datos: excluye todas las variables EDU_* y los clusters;
agrupa por CONGLOME; aprende transformaciones SOLO en train; el test se
usa UNA VEZ para evaluar al ganador elegido por CV agrupada en train.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / 'data' / 'processed' / 'dataset_enaho_escale_2025.csv'
SALIDAS = BASE / 'outputs' / 'clasificacion'
FIGURAS = BASE / 'outputs' / 'figures'
MODELOS = BASE / 'outputs' / 'models'
REPORTE = BASE / 'outputs' / 'reporte_clasificacion_2025.txt'

OBJETIVO = 'EDU_P314A'
CODIGO_POSITIVO = '2'    # Significado PENDIENTE de verificar en cuestionario
CODIGOS_VALIDOS = {'1', '2'}

# Predictores sociodemograficos, de vivienda y oferta escolar por distrito.
# No se incluyen EDU_*, CLUSTER_EXPLORATORIO, identificadores ni factores.
NUMERICAS = [
    'EDAD', 'SUM_INGHOG2D', 'SUM_GASHOG2D',
    'MIN_RATIO_ALUMNOS_DOCENTE', 'MIN_PROP_PUBLICOS',
]
CATEGORICAS = ['P207', 'SUM_POBREZA', 'VIV_P1121', 'VIV_P1144']


def log(lineas, texto=''):
    print(texto, flush=True)
    lineas.append(str(texto))


def cargar():
    if not ENTRADA.is_file():
        raise FileNotFoundError(f'Falta {ENTRADA}. Debes ejecutar antes la fase de integracion.')
    df = pd.read_csv(ENTRADA, dtype=str, encoding='utf-8-sig',
                     keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    necesarias = [OBJETIVO, 'CONGLOME', 'AÑO'] + NUMERICAS + CATEGORICAS
    faltantes = sorted(set(necesarias) - set(df.columns))
    if faltantes:
        raise ValueError(f'Faltan columnas: {faltantes}')
    for c in necesarias:
        df[c] = df[c].str.strip()
    if not df['AÑO'].eq('2025').all():
        raise ValueError('Se esperaban solo registros de 2025.')
    if df['CONGLOME'].eq('').any():
        raise ValueError('Existen conglomerados vacios. No es valido dividir asi.')
    if df['CONGLOME'].nunique() < 6:
        raise ValueError('Muy pocos conglomerados diferentes para validacion agrupada.')

    conteos_todos = df[OBJETIVO].replace('', 'SIN_DATO').value_counts().to_dict()
    df = df.loc[df[OBJETIVO].isin(CODIGOS_VALIDOS)].copy()
    if df.empty or df[OBJETIVO].nunique() != 2:
        raise ValueError('No se encuentran ambas clases de la encuesta.')
    y = df[OBJETIVO].eq(CODIGO_POSITIVO).astype(int)
    if y.value_counts().min() < 10:
        raise ValueError('Muy pocas observaciones de una clase.')
    X = df[NUMERICAS + CATEGORICAS].copy()
    for c in NUMERICAS:
        X[c] = pd.to_numeric(
            X[c].str.replace(',', '.', regex=False).replace('', np.nan),
            errors='coerce'
        )
    for c in CATEGORICAS:
        X[c] = X[c].replace('', np.nan).astype(object)
    grupos = df['CONGLOME'].astype(str)
    return df, X, y, grupos, conteos_todos


def preprocesador():
    # La imputacion/OneHot/escalado se ajustan dentro de cada fold de train.
    num = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler()),
    ])
    cat = Pipeline([
        ('imputar', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numericas', num, NUMERICAS),
        ('categoricas', cat, CATEGORICAS),
    ], sparse_threshold=0)


def modelos():
    return {
        'Regresion_Logistica_BASE': LogisticRegression(
            max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=5,
            class_weight='balanced_subsample', random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'Hist_Gradient_Boosting': HistGradientBoostingClassifier(
            max_iter=110, max_leaf_nodes=15, learning_rate=0.07,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        'MLP_Red_Neuronal': MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01,
            max_iter=200, early_stopping=True,
            n_iter_no_change=12, random_state=RANDOM_STATE,
        ),
    }


def figuras(ytest, probas, predicciones):
    cm = confusion_matrix(ytest, predicciones, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=.046)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1], labels=['Cod. 1', 'Cod. 2'])
    ax.set_yticks([0, 1], labels=['Cod. 1', 'Cod. 2'])
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Respuesta real')
    ax.set_title('Matriz de confusion (test por conglomerado)')
    fig.tight_layout()
    fig.savefig(FIGURAS / 'clasificacion_matriz_confusion_2025.png', dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(ytest, probas)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(ytest, probas):.3f}')
    ax.plot([0,1], [0,1], '--', color='gray', label='Azar')
    ax.set(xlabel='Tasa de falsos positivos', ylabel='Sensibilidad',
           title='Curva ROC — test agrupado')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / 'clasificacion_roc_2025.png', dpi=160)
    plt.close(fig)

    precisiones, recalls, _ = precision_recall_curve(ytest, probas)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(recalls, precisiones,
            label=f'PR-AUC(AP) = {average_precision_score(ytest, probas):.3f}')
    ax.axhline(float(ytest.mean()), linestyle='--', color='gray',
               label='Prevalencia clase 2')
    ax.set(xlabel='Recall clase 2', ylabel='Precision clase 2',
           title='Curva precision-recall — test agrupado')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / 'clasificacion_precision_recall_2025.png', dpi=160)
    plt.close(fig)

    obs, est = calibration_curve(ytest, probas, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(est, obs, 'o-', label='Modelo')
    ax.plot([0,1], [0,1], '--', color='gray', label='Calibracion perfecta')
    ax.set(xlabel='Probabilidad estimada (codigo 2)',
           ylabel='Frecuencia observada',
           title='Calibracion de probabilidades')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / 'clasificacion_calibracion_2025.png', dpi=160)
    plt.close(fig)
    return cm


def main():
    for p in (SALIDAS, FIGURAS, MODELOS, REPORTE.parent):
        p.mkdir(parents=True, exist_ok=True)
    informe = []
    df, X, y, grupos, todos = cargar()
    log(informe, 'EXAMEN IA ENAHO–ESCALE 2025 | PRACTICA 11 CLASIFICACION')
    log(informe, 'OBJETIVO TECNICO: respuesta codigo 2 de EDU_P314A vs codigo 1.')
    log(informe, 'El significado semantico de 1/2 requiere verificar el cuestionario entregado.')
    log(informe, f'Dataset maestro: {len(pd.read_csv(ENTRADA, usecols=[OBJETIVO])):,} filas')
    log(informe, f'Distribucion original de {OBJETIVO}: {todos}')
    log(informe, f'Registros con respuesta valida (1/2): {len(df):,}')
    log(informe, f'Clase 0 (codigo 1): {(y==0).sum():,}')
    log(informe, f'Clase 1 (codigo 2): {(y==1).sum():,}')
    log(informe, f'Predictores: {NUMERICAS + CATEGORICAS}')
    log(informe, 'Excluidos como predictores: EDU_*, clusters, codigos de persona, UBIGEO, factores muestrales.')

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    i_train, i_test = next(divisor.split(X, y, groups=grupos))
    Xtr, Xte = X.iloc[i_train], X.iloc[i_test]
    ytr, yte = y.iloc[i_train], y.iloc[i_test]
    gtr, gte = grupos.iloc[i_train], grupos.iloc[i_test]
    if len(set(gtr) & set(gte)):
        raise AssertionError('FUGA DE DATOS: conglomerados compartidos entre train y test')
    if ytr.nunique() != 2 or yte.nunique() != 2:
        raise ValueError('La division agrupada dejo a un conjunto sin una clase.')
    log(informe, f'Entrenamiento: {len(i_train):,} | Prueba: {len(i_test):,}')
    log(informe, f'Conglomerados train: {gtr.nunique()} | test: {gte.nunique()} | interseccion: 0')
    log(informe, f'Clase codigo 2 en train: {ytr.mean():.3%} | test: {yte.mean():.3%}')
    log(informe, '\nVALIDACION CRUZADA AGRUPADA (solo train, 3 folds)')

    cv = GroupKFold(n_splits=3)
    scores = {'roc_auc': 'roc_auc', 'pr_auc': 'average_precision',
              'f1_macro': 'f1_macro', 'balanced_acc': 'balanced_accuracy'}
    filas = []
    configs = modelos()
    for nombre, algoritmo in configs.items():
        log(informe, f'Calculando CV: {nombre} ...')
        pipe = Pipeline([('preprocesar', preprocesador()), ('modelo', algoritmo)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            resultado = cross_validate(
                pipe, Xtr, ytr, groups=gtr, cv=cv,
                scoring=scores, n_jobs=1, error_score='raise'
            )
        fila = {'modelo': nombre}
        for metrica in scores:
            valores = resultado[f'test_{metrica}']
            fila[f'{metrica}_media'] = float(np.mean(valores))
            fila[f'{metrica}_desv'] = float(np.std(valores))
        filas.append(fila)
        log(informe, f"  ROC-AUC={fila['roc_auc_media']:.4f} | PR-AUC(AP)={fila['pr_auc_media']:.4f}"
                     f" | F1-macro={fila['f1_macro_media']:.4f} | BalancedAcc={fila['balanced_acc_media']:.4f}")

    tabla = pd.DataFrame(filas).sort_values(
        ['pr_auc_media', 'roc_auc_media'], ascending=False
    )
    archivo_cv = SALIDAS / 'comparacion_modelos_cv_2025.csv'
    tabla.to_csv(archivo_cv, index=False, encoding='utf-8-sig')
    ganador = tabla.iloc[0]['modelo']
    log(informe, f'\nSeleccion por PR-AUC promedio en CV train: {ganador}')
    mejor = Pipeline([('preprocesar', preprocesador()), ('modelo', configs[ganador])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        mejor.fit(Xtr, ytr)
    probas = mejor.predict_proba(Xte)[:, 1]
    predicciones = mejor.predict(Xte)
    test = {
        'modelo': ganador,
        'roc_auc': float(roc_auc_score(yte, probas)),
        'pr_auc_ap': float(average_precision_score(yte, probas)),
        'f1_macro': float(f1_score(yte, predicciones, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(yte, predicciones)),
        'balanced_accuracy': float(balanced_accuracy_score(yte, predicciones)),
        'precision_clase2': float(precision_score(yte, predicciones, zero_division=0)),
        'recall_clase2': float(recall_score(yte, predicciones, zero_division=0)),
    }
    cm = figuras(yte, probas, predicciones)
    log(informe, '\nTEST FINAL (evaluacion unica del mejor modelo):')
    for k,v in test.items():
        log(informe, f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    log(informe, f'Confusion (filas reales [cod.1,cod.2] vs pred [cod.1,cod.2]): {cm.tolist()}')
    pd.DataFrame([test]).to_csv(SALIDAS / 'metricas_test_2025.csv', index=False,
                                 encoding='utf-8-sig')
    # Evitamos publicar claves/UBIGEO individuales en exportes.
    pd.DataFrame({'y_real_codigo2': yte.to_numpy(),
                  'prob_codigo2': probas,
                  'pred_codigo2': predicciones}).to_csv(
                      SALIDAS / 'predicciones_test_sin_identificadores.csv',
                      index=False, encoding='utf-8-sig'
                  )
    ruta_modelo = MODELOS / 'clasificador_2025.joblib'
    joblib.dump(mejor, ruta_modelo)
    log(informe, '\nARCHIVOS CREADOS:')
    for p in [archivo_cv, SALIDAS / 'metricas_test_2025.csv',
              SALIDAS / 'predicciones_test_sin_identificadores.csv',
              ruta_modelo]:
        log(informe, f'  {p}')
    log(informe, '  Graficas: outputs/figures/clasificacion_*.png')
    log(informe, 'PRECAUCION: no usar estas metricas como evidencia de representatividad nacional.')
    log(informe, 'PRECAUCION: verificar etiquetas de la encuesta antes de interpretar la clase 2.')
    REPORTE.write_text('\n'.join(informe) + '\n', encoding='utf-8')
    print(f'\nReporte guardado en: {REPORTE}', flush=True)

if __name__ == '__main__':
    main()
