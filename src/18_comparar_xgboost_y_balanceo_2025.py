"""EXAMEN IA ENAHO–ESCALE 2025 | PASO 18.

Objetivo principal: evaluar XGBoost (tercer algoritmo avanzado de la lista
explicita del docente) y el efecto del balanceo de clases SOLO con validacion
cruzada agrupada dentro de TRAIN. No abre ni evalua el TEST.

Ejecutar en la raiz del proyecto (con venv activado):
    python -m pip install xgboost pandas numpy scikit-learn
    python src/18_comparar_xgboost_y_balanceo_2025.py

Opcional, SOLO para una nueva pregunta de investigacion:
    python src/18_comparar_xgboost_y_balanceo_2025.py --objetivo B

A: P314A=2 entre todas las personas de 6-17 con respuesta valida.
B: P314A=2 SOLO entre personas matriculadas P306=1 (experimento distinto).

Este script solo crea ficheros nuevos en outputs/auditoria_clasificacion;
NO sobrescribe dataset, informe, modelos o SHAP anteriores.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, balanced_accuracy_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(__file__).resolve().parent.parent
PASO_11 = BASE / 'src' / '11_clasificacion_2025.py'
OUT = BASE / 'outputs' / 'auditoria_clasificacion'
SEED = 42


def importar_paso11():
    if not PASO_11.is_file():
        raise FileNotFoundError(f'No encuentro el script del Paso 11: {PASO_11}')
    spec = importlib.util.spec_from_file_location('paso11_para_18', PASO_11)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def confirmar_split_no_fuga(X, y, grupos, paso11, objetivo):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=paso11.RANDOM_STATE)
    itr, ite = next(splitter.split(X, y, groups=grupos))
    Xtr, ytr, gtr = X.iloc[itr], y.iloc[itr], grupos.iloc[itr]
    gte = grupos.iloc[ite]
    if set(gtr) & set(gte):
        raise AssertionError('Fuga: conglomerado repetido entre entrenamiento y prueba.')
    if objetivo == 'A' and (len(itr), len(ite)) != (17690, 5928):
        raise RuntimeError('La particion A no coincide con el Paso 11 (17690/5928); '
                           'revisar que dataset y src/11_clasificacion_2025.py no hayan cambiado.')
    if ytr.nunique() != 2:
        raise RuntimeError('TRAIN debe contener las 2 clases.')
    return Xtr, ytr, gtr, len(ite), gte.nunique()


def construir_modelos(paso11):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit('Falta la biblioteca xgboost. Ejecuta: python -m pip install xgboost') from exc

    xgb_parametros = dict(
        n_estimators=160, max_depth=3, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.90,
        min_child_weight=5, reg_lambda=2.0,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_jobs=2, random_state=SEED,
    )
    # Se usa exactamente el HistGradientBoosting que ya fue evaluado en Paso 11
    # y el mismo algoritmo con pesos calculados en cada fold de entrenamiento.
    return [
        ('HistGB_original_sin_pesos', paso11.modelos()['Hist_Gradient_Boosting'], False),
        ('HistGB_con_pesos_clase', paso11.modelos()['Hist_Gradient_Boosting'], True),
        ('XGBoost_sin_pesos', XGBClassifier(**xgb_parametros), False),
        ('XGBoost_con_pesos_clase', XGBClassifier(**xgb_parametros), True),
    ]


def numero_json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    raise TypeError(str(type(v)))


def main():
    parser = argparse.ArgumentParser(description='CV agrupada de HistGB y XGBoost con/sin balanceo de clase.')
    parser.add_argument('--objetivo', choices=['A', 'B'], default='A',
                        help='A=cohorte original; B=solo matriculados, experimento nuevo.')
    args = parser.parse_args()
    objetivo = args.objetivo

    paso11 = importar_paso11()
    df, X, y, grupos, _ = paso11.cargar()  # conserva el universo original filtrado a P314A=1/2
    if objetivo == 'B':
        if 'EDU_P306' not in df:
            raise RuntimeError('Falta EDU_P306 para construir el objetivo B.')
        m = df['EDU_P306'].astype(str).str.strip().eq('1')
        df, X, y, grupos = df.loc[m], X.loc[m], y.loc[m], grupos.loc[m]
    Xtr, ytr, gtr, n_test_reservado, conglomerados_test = confirmar_split_no_fuga(
        X, y, grupos, paso11, objetivo
    )

    modelos = construir_modelos(paso11)
    cv = GroupKFold(n_splits=3)
    folds = list(cv.split(Xtr, ytr, groups=gtr))
    informe = [
        'ENAHO–ESCALE 2025 | PASO 18 | XGBOOST Y BALANCEO EN CV DE TRAIN',
        f'Objetivo: {objetivo} | positivo: P314A=2 (NO USO de internet el mes anterior).',
        'A = todas las respuestas validas (objetivo previo). B = solo matriculados (experimento nuevo).',
        f'Cohorte objetivo: {len(X):,} | TRAIN: {len(Xtr):,} | TEST INTACTO: {n_test_reservado:,}',
        f'Conglomerados TRAIN: {gtr.nunique():,} | TEST: {conglomerados_test:,} | interseccion: 0',
        f'Positivos en TRAIN: {int(ytr.sum()):,} ({ytr.mean():.2%}).',
        'GroupKFold: 3 folds. Misma semilla (42) y preprocesamiento del Paso 11.',
        'Imputacion, escalado y One-Hot se ajustan SOLO sobre entrenamiento de cada fold.',
        'Los pesos de clase (si se usan) se calculan SOLO con y del entrenamiento de cada fold.',
        'SUM_FACTOR07 NO se usa: aun no se ha confirmado su idoneidad para esta subpoblacion.',
        'NO se entrena ni se evalua sobre el conjunto TEST. NO se sobrescriben modelos ni SHAP.',
        '',
    ]
    detalles = []
    promedios = []
    for nombre, estimador, balanceado in modelos:
        print(f'Validando {nombre}...', flush=True)
        for fold, (i_entrena, i_valida) in enumerate(folds, start=1):
            g_entrena = gtr.iloc[i_entrena]
            g_valida = gtr.iloc[i_valida]
            if set(g_entrena) & set(g_valida):
                raise AssertionError(f'Fuga entre los grupos de fold {fold}')
            y_entrena = ytr.iloc[i_entrena]
            y_valida = ytr.iloc[i_valida]
            if y_entrena.nunique() != 2 or y_valida.nunique() != 2:
                raise RuntimeError(f'Fold {fold} no tiene ambas clases.')
            pipe = Pipeline([
                ('preprocesar', paso11.preprocesador()),
                ('modelo', clone(estimador)),
            ])
            pesos = None
            if balanceado:
                pesos = compute_sample_weight(class_weight='balanced', y=y_entrena)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                if pesos is not None:
                    pipe.fit(Xtr.iloc[i_entrena], y_entrena, modelo__sample_weight=pesos)
                else:
                    pipe.fit(Xtr.iloc[i_entrena], y_entrena)
            prob = pipe.predict_proba(Xtr.iloc[i_valida])[:, 1]
            pred = (prob >= 0.5).astype(int)
            resultado = {
                'objetivo': objetivo, 'modelo': nombre, 'fold': fold,
                'train_n': len(i_entrena), 'validacion_n': len(i_valida),
                'positivos_train': int(y_entrena.sum()),
                'positivos_validacion': int(y_valida.sum()),
                'ap': float(average_precision_score(y_valida, prob)),
                'roc_auc': float(roc_auc_score(y_valida, prob)),
                'f1_macro': float(f1_score(y_valida, pred, average='macro', zero_division=0)),
                'balanced_accuracy': float(balanced_accuracy_score(y_valida, pred)),
                'precision_clase2': float(precision_score(y_valida, pred, zero_division=0)),
                'recall_clase2': float(recall_score(y_valida, pred, zero_division=0)),
                'pesos_clase': 'SI' if balanceado else 'NO',
            }
            detalles.append(resultado)
            print(f"  Fold {fold}: AP={resultado['ap']:.4f}, ROC-AUC={resultado['roc_auc']:.4f}, "
                  f"F1-macro={resultado['f1_macro']:.4f}, recall clase2={resultado['recall_clase2']:.4f}",
                  flush=True)
        resultado_modelo = [r for r in detalles if r['modelo'] == nombre]
        fila = {'objetivo': objetivo, 'modelo': nombre,
                'pesos_clase': 'SI' if balanceado else 'NO'}
        for metrica in ('ap', 'roc_auc', 'f1_macro', 'balanced_accuracy', 'precision_clase2', 'recall_clase2'):
            valores = np.asarray([r[metrica] for r in resultado_modelo], dtype=float)
            fila[metrica+'_media'] = float(valores.mean())
            fila[metrica+'_desv'] = float(valores.std(ddof=0))
        promedios.append(fila)
    OUT.mkdir(parents=True, exist_ok=True)
    archivo_fold = OUT / f'paso18_resultados_por_fold_objetivo_{objetivo}.csv'
    archivo_comp = OUT / f'paso18_comparacion_cv_objetivo_{objetivo}.csv'
    archivo_rep = BASE / 'outputs' / f'reporte_paso18_xgboost_balanceo_objetivo_{objetivo}.txt'
    pd.DataFrame(detalles).to_csv(archivo_fold, encoding='utf-8-sig', index=False)
    tabla = pd.DataFrame(promedios).sort_values(['ap_media', 'roc_auc_media'], ascending=False)
    tabla.to_csv(archivo_comp, encoding='utf-8-sig', index=False)
    informe.append('--- Resultados medios de CV SOLO en TRAIN ---')
    for _, r in tabla.iterrows():
        informe.append(
            f"{r['modelo']}: AP={r['ap_media']:.6f} (+/- {r['ap_desv']:.6f}); "
            f"ROC-AUC={r['roc_auc_media']:.4f}; F1-macro={r['f1_macro_media']:.4f}; "
            f"Recall clase2={r['recall_clase2_media']:.4f}; "
            f"Precision clase2={r['precision_clase2_media']:.4f}"
        )
    informe.extend([
        '',
        f"Mejor AP en esta comparacion CV: {tabla.iloc[0]['modelo']} "
        f"(AP={tabla.iloc[0]['ap_media']:.6f}).",
        'IMPORTANTE: un AP mejor en CV NO demuestra mejor rendimiento en TEST.',
        'No seleccionar ni sustituir el modelo final solo porque este experimento obtuvo un AP mayor.',
        'La variacion entre folds y la precision-recall deben considerarse antes de concluir.',
        'Objetivo B NO es identico al del Paso 11; sus resultados NO se comparan directamente con el TEST A.',
        '',
        f'Archivo por fold: {archivo_fold.relative_to(BASE)}',
        f'Archivo resumen: {archivo_comp.relative_to(BASE)}',
        'No se cambian archivos de pasos 05-17 ni modelos o graficas anteriores.',
    ])
    archivo_rep.write_text('\n'.join(informe)+'\n', encoding='utf-8')
    print('\n'.join(informe), flush=True)
    print(f'\nREPORTE CREADO: {archivo_rep}', flush=True)


if __name__ == '__main__':
    main()
