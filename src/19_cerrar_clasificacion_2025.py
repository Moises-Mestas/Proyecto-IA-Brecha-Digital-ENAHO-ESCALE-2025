"""ENAHO-ESCALE 2025 | PASO 19: clasificacion de no uso de internet entre asistentes.

Este nuevo experimento responde mejor al contexto educativo del examen:
  Universo: 6-17 anios, matriculado P306=1, ASISTE P307=1.
  Objetivo: P314A=2 NO uso internet el mes anterior (vs 1=SI).
Es un indicador medible de no uso entre asistentes, NO mide competencias digitales.
No altera el modelo, SHAP, dataset ni informe del experimento A.

Ejecutar desde raiz del proyecto:
    python src/19_cerrar_clasificacion_2025.py
Opcional:
    python src/19_cerrar_clasificacion_2025.py --iteraciones 20
    python src/19_cerrar_clasificacion_2025.py --sin-shap

Requiere: pip install pandas numpy scikit-learn matplotlib xgboost shap joblib
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             balanced_accuracy_score, recall_score, precision_score,
                             accuracy_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

SEED=42
BASE=Path(__file__).resolve().parent.parent
SRC=BASE/'src'/'11_clasificacion_2025.py'
OUT=BASE/'outputs'/'cierre_2025'
FIG=BASE/'outputs'/'figures'
MODELS=BASE/'outputs'/'models'
REPORTS=BASE/'reports'


def cargar_paso11():
    if not SRC.is_file():
        raise FileNotFoundError(f'Falta el script original del Paso 11: {SRC}')
    spec=importlib.util.spec_from_file_location('paso11_para_cierre19', SRC)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def formato_metrica(y, prob):
    pred=np.asarray(prob)>=0.5
    return {'ap':float(average_precision_score(y,prob)),
            'roc_auc':float(roc_auc_score(y,prob)),
            'f1_macro':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),
            'recall_no_uso':float(recall_score(y,pred,zero_division=0)),
            'precision_no_uso':float(precision_score(y,pred,zero_division=0))}


def modelos_base():
    from xgboost import XGBClassifier
    xgb=dict(n_estimators=175,max_depth=3,learning_rate=.06,
             subsample=.85,colsample_bytree=.90,min_child_weight=5,
             reg_lambda=2.,objective='binary:logistic',eval_metric='logloss',
             tree_method='hist',n_jobs=2,random_state=SEED)
    return {
        'Regresion_Logistica_BASE': (LogisticRegression(max_iter=1200,random_state=SEED), 'clase'),
        'Random_Forest_pesos_clase': (RandomForestClassifier(n_estimators=160,max_depth=14,
                                   min_samples_leaf=5,n_jobs=2,random_state=SEED), 'clase'),
        'MLP_Red_Neuronal': (MLPClassifier(hidden_layer_sizes=(32,16),alpha=.01,max_iter=200,
                             early_stopping=True,n_iter_no_change=12,random_state=SEED), 'sin'),
        'HistGB_sin_pesos':(HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=15,
                             learning_rate=.07,l2_regularization=1.,random_state=SEED),'sin'),
        'HistGB_con_pesos_clase':(HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=15,
                             learning_rate=.07,l2_regularization=1.,random_state=SEED),'clase'),
        'XGBoost_sin_pesos':(XGBClassifier(**xgb),'sin'),
        'XGBoost_con_pesos_clase':(XGBClassifier(**xgb),'clase'),
        'XGBoost_factor_sensibilidad':(XGBClassifier(**xgb),'factor_clase')
    }


def pesar(y_train, factors_train, mode):
    if mode=='sin': return None
    yv=np.asarray(y_train).astype(int)
    w=compute_sample_weight('balanced',yv).astype(float)
    if mode=='clase': return w
    if mode=='factor_clase':
        # El FACTOR07 es objeto de una prueba de sensibilidad: no implica representatividad.
        fw=np.asarray(factors_train,dtype=float)
        if not np.isfinite(fw).all() or np.any(fw<=0):
            raise ValueError('SUM_FACTOR07 debe ser finito y positivo para la sensibilidad.')
        fw=fw/np.mean(fw)  # Factor normalizado dentro del TRAIN de cada fold
        return w*fw
    raise ValueError(f'Modo de pesos desconocido: {mode}')


def validar_cv(nombre,algo,modo,Xtr,ytr,gtr,ftr,paso11,folds):
    filas=[]
    for f,(i1,i2) in enumerate(folds,1):
        if set(gtr.iloc[i1]) & set(gtr.iloc[i2]):
            raise AssertionError('Fuga de conglomerados en folds.')
        yy1,yy2=ytr.iloc[i1],ytr.iloc[i2]
        if yy1.nunique()!=2 or yy2.nunique()!=2:
            raise ValueError('Un fold no tiene las dos clases.')
        pipe=Pipeline([('preprocesar',paso11.preprocesador()),('modelo',clone(algo))])
        sw=pesar(yy1,ftr.iloc[i1],modo)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=UserWarning)
            if sw is None: pipe.fit(Xtr.iloc[i1],yy1)
            else: pipe.fit(Xtr.iloc[i1],yy1,modelo__sample_weight=sw)
        met=formato_metrica(yy2,pipe.predict_proba(Xtr.iloc[i2])[:,1])
        filas.append({'modelo':nombre,'fold':f,'modo_peso':modo,
                      'n_train':len(i1),'n_validacion':len(i2),**met})
        print(f'  {nombre} fold {f}: AP={met["ap"]:.4f} ROC={met["roc_auc"]:.4f} recall={met["recall_no_uso"]:.3f}',flush=True)
    return filas


def crear_figuras(yte, probs):
    y=np.asarray(yte)
    pred=(probs>=0.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(6.1,4.7))
    im=ax.imshow(cm,cmap='Blues')
    fig.colorbar(im,ax=ax,fraction=.046)
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=13,
                                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1],['Uso','No uso']);ax.set_yticks([0,1],['Uso','No uso'])
    ax.set(xlabel='Prediccion',ylabel='Respuesta real',title='No uso internet entre asistentes - test agrupado')
    fig.tight_layout();fig.savefig(FIG/'cierre19_matriz_confusion.png',dpi=155);plt.close(fig)

    fpr,tpr,_=roc_curve(y,probs)
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    ax.plot(fpr,tpr,label=f'ROC-AUC={roc_auc_score(y,probs):.3f}')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Tasa falsos positivos',ylabel='Recall',title='ROC - asistentes - test')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'cierre19_roc.png',dpi=155);plt.close(fig)

    p,r,_=precision_recall_curve(y,probs)
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    ax.plot(r,p,label=f'AP={average_precision_score(y,probs):.3f}')
    ax.axhline(y.mean(),color='gray',linestyle='--',label='Prevalencia')
    ax.set(xlabel='Recall de no uso',ylabel='Precision de no uso',title='PR - asistentes - test')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'cierre19_precision_recall.png',dpi=155);plt.close(fig)

    prob_cal,estim_cal=calibration_curve(y,probs,n_bins=10,strategy='quantile')
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    ax.plot(estim_cal,prob_cal,'o-',label='Modelo')
    ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
    ax.set(xlabel='Probabilidad predicha',ylabel='Proporcion observada',title='Calibracion - asistentes - test')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'cierre19_calibracion.png',dpi=155);plt.close(fig)
    return cm


def explicar_shap(pipeline,Xte,yte,probs,paso11,n=240):
    import shap
    procesador=pipeline.named_steps['preprocesar']
    modelo=pipeline.named_steps['modelo']
    y=np.asarray(yte)
    buenos_pos=np.flatnonzero((y==1)&(probs>=0.5))
    buenos_neg=np.flatnonzero((y==0)&(probs<0.5))
    if not len(buenos_pos) or not len(buenos_neg):
        raise RuntimeError('No hay dos casos extremos correctamente clasificados para SHAP.')
    izq=int(buenos_pos[np.argmax(probs[buenos_pos])])
    der=int(buenos_neg[np.argmin(probs[buenos_neg])])
    rng=np.random.default_rng(SEED)
    indices=np.unique(np.r_[rng.choice(len(Xte),size=min(n,len(Xte)),replace=False),[izq,der]])
    xi=Xte.iloc[indices]
    z=np.asarray(procesador.transform(xi),dtype=float)
    nombres_raw=procesador.get_feature_names_out()
    nombres=[]
    for name in nombres_raw:
        if name.startswith('numericas__'): nombres.append(name.replace('numericas__','',1))
        elif name.startswith('categoricas__'): nombres.append(name.replace('categoricas__','',1))
        else:nombres.append(name)
    try:
        expl=shap.TreeExplainer(modelo,model_output='raw')
        sv=expl(z)
    except (TypeError,ValueError,AssertionError) as ex:
        # MLP u otro modelo no arboreo: explicador de permutaciones, sin tocar test/entrenamiento
        ztrain=np.asarray(procesador.transform(Xte.iloc[:min(80,len(Xte))]),dtype=float)
        expl=shap.Explainer(modelo.predict_proba,shap.maskers.Independent(ztrain),algorithm='permutation')
        sv=expl(z,max_evals=2*z.shape[1]+1)
    if sv.values.ndim==3: sv=sv[:,:,1]
    sv.feature_names=nombres
    sv.data=z
    reconstruccion=np.asarray(sv.base_values).reshape(-1)+sv.values.sum(1)
    exp_prob=np.asarray(modelo.predict_proba(z))[:,1]
    from scipy.special import expit
    err_prob=float(np.max(np.abs(reconstruccion-exp_prob)))
    err_logit=float(np.max(np.abs(expit(reconstruccion)-exp_prob)))
    if min(err_prob,err_logit)>.025:
        raise AssertionError(f'Error de aditividad SHAP demasiado alto: prob={err_prob}; logit={err_logit}')
    escala='probabilidad' if err_prob<err_logit else 'log-odds'
    orden=np.argsort(np.abs(sv.values).mean(axis=0))[::-1]
    pd.DataFrame({'variable':nombres,'importancia_media_absoluta':np.abs(sv.values).mean(axis=0)})\
       .sort_values('importancia_media_absoluta',ascending=False)\
       .to_csv(OUT/'cierre19_importancia_shap.csv',index=False,encoding='utf-8-sig')

    plt.close('all');shap.plots.beeswarm(sv,max_display=10,show=False)
    plt.title('SHAP - no uso internet entre asistentes')
    plt.savefig(FIG/'cierre19_shap_beeswarm.png',dpi=160,bbox_inches='tight');plt.close('all')
    numeric_nombres=paso11.NUMERICAS
    idx_nums=sorted([nombres.index(c) for c in numeric_nombres if c in nombres],
                    key=lambda i:np.abs(sv.values[:,i]).mean(),reverse=True)
    for i,idx in enumerate(idx_nums[:2],1):
        plt.close('all');shap.plots.scatter(sv[:,idx],show=False)
        plt.title('Dependencia SHAP: '+nombres[idx]);plt.savefig(FIG/f'cierre19_shap_dependencia_{i}.png',dpi=160,bbox_inches='tight');plt.close('all')
    for tag,ind in [('no_uso',izq),('uso',der)]:
        j=int(np.where(indices==ind)[0][0])
        plt.close('all');shap.plots.waterfall(sv[j],max_display=10,show=False)
        plt.title('Caso anonimo: '+tag)
        plt.savefig(FIG/f'cierre19_shap_waterfall_{tag}.png',dpi=160,bbox_inches='tight');plt.close('all')
    return escala,len(indices),[(nombres[i],float(np.abs(sv.values[:,i]).mean())) for i in orden[:10]],err_prob,err_logit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iteraciones',type=int,default=12)
    parser.add_argument('--sin-shap',action='store_true')
    args=parser.parse_args()
    if not (1<=args.iteraciones<=60): raise ValueError('Iteraciones deben estar entre 1 y 60.')
    for d in (OUT,FIG,MODELS,REPORTS):d.mkdir(exist_ok=True,parents=True)
    paso11=cargar_paso11()
    df,X,y,grupos,_=paso11.cargar()  # solo respuesta P314A valida, sin EDU en predictores
    if 'EDU_P307' not in df.columns:
        raise ValueError('Falta EDU_P307: la base maestro debe incluir la pregunta de asistencia escolar.')
    m=(df['EDU_P306'].astype(str).str.strip().eq('1') &
       df['EDU_P307'].astype(str).str.strip().eq('1'))
    df,X,y,grupos=df.loc[m].copy(),X.loc[m].copy(),y.loc[m].copy(),grupos.loc[m].copy()
    if len(df)<500 or y.nunique()!=2:raise ValueError('Cohorte de asistentes insuficiente para clasificacion.')
    if 'SUM_FACTOR07' not in df.columns:raise ValueError('Falta SUM_FACTOR07 para la prueba de sensibilidad.')
    factors=pd.to_numeric(df['SUM_FACTOR07'].astype(str).str.replace(',','.',regex=False),errors='coerce')
    if not (np.isfinite(factors).all() and (factors>0).all()):
        raise ValueError('Hay factores muestrales invalidos para el experimento de sensibilidad.')
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED)
    itr,ite=next(splitter.split(X,y,groups=grupos))
    Xtr,Xte=X.iloc[itr],X.iloc[ite]
    ytr,yte=y.iloc[itr],y.iloc[ite]
    gtr,gte=grupos.iloc[itr],grupos.iloc[ite]
    ftr=factors.iloc[itr]
    if set(gtr)&set(gte):raise AssertionError('Hay conglomerados compartidos train-test.')
    if ytr.nunique()!=2 or yte.nunique()!=2:raise ValueError('Una particion carece de clase.')
    print(f'COHORTE C: {len(df)} estudiantes matriculados que asisten. Train={len(itr)} Test={len(ite)}.',flush=True)
    print(f'Conglomerados train={gtr.nunique()}, test={gte.nunique()}, compartidos=0',flush=True)
    folds=list(GroupKFold(n_splits=3).split(Xtr,ytr,groups=gtr))
    detalles=[]
    configs=modelos_base()
    for nombre,(alg,modo) in configs.items():
        detalles+=validar_cv(nombre,alg,modo,Xtr,ytr,gtr,ftr,paso11,folds)
    # Optimizar el XGBoost SIN pesos mediante GroupKFold; el test no interviene.
    from xgboost import XGBClassifier
    print(f'Busqueda de {args.iteraciones} combinaciones de XGBoost sobre TRAIN...',flush=True)
    busqueda=RandomizedSearchCV(Pipeline([('preprocesar',paso11.preprocesador()),
                          ('modelo',XGBClassifier(objective='binary:logistic',eval_metric='logloss',
                                tree_method='hist',random_state=SEED,n_jobs=2))]),
          {'modelo__n_estimators':[100,160,230,300],
           'modelo__max_depth':[2,3,4,5],
           'modelo__learning_rate':[.025,.05,.07,.10],
           'modelo__min_child_weight':[2,5,10],
           'modelo__subsample':[.75,.9,1.],
           'modelo__colsample_bytree':[.8,1.],
           'modelo__reg_lambda':[1.,3.,6.]},
          n_iter=args.iteraciones,scoring='average_precision',cv=GroupKFold(3),
          random_state=SEED,n_jobs=1,error_score='raise',refit=True)
    busqueda.fit(Xtr,ytr,groups=gtr)
    pd.DataFrame(busqueda.cv_results_)[['rank_test_score','mean_test_score','std_test_score','params']]\
      .sort_values('rank_test_score').to_csv(OUT/'cierre19_busqueda_xgboost.csv',index=False,encoding='utf-8-sig')
    params=busqueda.best_estimator_.named_steps['modelo'].get_params()
    tuned=XGBClassifier(**params)
    # Reestimar score de configuracion escogida: es CV de seleccion, no evaluacion externa.
    detalle_tuned=validar_cv('XGBoost_optimizado_sin_pesos',tuned,'sin',Xtr,ytr,gtr,ftr,paso11,folds)
    detalles+=detalle_tuned
    # Tambien medir equilibrio con los hiperparametros nuevos, en cada fold.
    detalles+=validar_cv('XGBoost_optimizado_con_pesos',tuned,'clase',Xtr,ytr,gtr,ftr,paso11,folds)

    tab=pd.DataFrame(detalles)
    tab.to_csv(OUT/'cierre19_resultados_por_fold.csv',index=False,encoding='utf-8-sig')
    metricas=['ap','roc_auc','f1_macro','balanced_accuracy','accuracy','recall_no_uso','precision_no_uso']
    comps=tab.groupby('modelo',sort=False)[metricas].agg(['mean','std'])
    comps.columns=['_'.join(c) for c in comps.columns.to_flat_index()]
    comps=comps.reset_index().sort_values('ap_mean',ascending=False)
    comps.to_csv(OUT/'cierre19_comparacion_cv.csv',index=False,encoding='utf-8-sig')
    ganador=comps.iloc[0]['modelo']
    if ganador=='XGBoost_optimizado_sin_pesos':estimador,modo=tuned,'sin'
    elif ganador=='XGBoost_optimizado_con_pesos':estimador,modo=tuned,'clase'
    else:estimador,modo=configs[ganador]
    print(f'GANADOR CV {ganador} AP={comps.iloc[0]["ap_mean"]:.5f}',flush=True)
    # Test se evalua exactamente una vez para el ganador seleccionado en TRAIN.
    pipe=Pipeline([('preprocesar',paso11.preprocesador()),('modelo',clone(estimador))])
    sw=pesar(ytr,ftr,modo)
    if sw is None:pipe.fit(Xtr,ytr)
    else:pipe.fit(Xtr,ytr,modelo__sample_weight=sw)
    probs=pipe.predict_proba(Xte)[:,1]
    test_scores=formato_metrica(yte,probs)
    cm=crear_figuras(yte,probs)
    joblib.dump(pipe,MODELS/'clasificador_2025_asistentes_paso19.joblib')
    pd.DataFrame([{'modelo':ganador,'cohorte':'MATRICULADOS_Y_ASISTENTES',
                   'n_train':len(ytr),'n_test':len(yte),**test_scores}]).to_csv(
                    OUT/'cierre19_metricas_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'y_real':np.asarray(yte,dtype=int), 'prob_no_uso':probs,
                  'pred_no_uso':(probs>=.5).astype(int)}).to_csv(
                   OUT/'cierre19_predicciones_test_sin_ids.csv',index=False,encoding='utf-8-sig')
    shap_res=None
    if not args.sin_shap:
        try:
            print('Generando 5 graficas SHAP del ganador...',flush=True)
            shap_res=explicar_shap(pipe,Xte,yte,probs,paso11)
        except Exception as exc:
            (OUT/'cierre19_error_shap.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(f'AVISO: SHAP fallo, el modelo y las metricas SI se guardaron: {exc}',flush=True)
    lines=['EXAMEN ENAHO-ESCALE 2025 | PASO 19 | MODELO EDUCACION + INTERNET',
      'NUEVO EXPERIMENTO C: P306=1 (matriculados), P307=1 (asisten), P314A=2 (NO uso internet).',
      'Mide NO USO RECIENTE ENTRE ASISTENTES, no competencia digital ni rendimiento.',
      'Se utilizan SOLO los archivos locales entregados por el docente.',
      f'Universo C: {len(df):,} | positivos no uso {int(y.sum()):,} ({y.mean():.2%}).',
      f'Train {len(itr):,} | Test {len(ite):,} | conglomerados compartidos 0.',
      f'Grupos train={gtr.nunique():,}, test={gte.nunique():,}.',
      'GroupKFold 3 folds dentro de train; preprocesamiento ajustado en cada fold.',
      f'Busquedas XGB de hiperparametros: {args.iteraciones}; mejor AP interno {busqueda.best_score_:.6f}.',
      'IMPORTANTE: la comparacion CV de hiperparametros ya explorados es optimista; test independiente sirve de evaluacion.',
      'Clase: pesos calculados en cada entrenamiento/fold; FACTOR07 solo como sensibilidad, no inferencia nacional.',
      '\nRESULTADOS CV AGRUPADA - TRAIN:']
    for _,r in comps.iterrows():
        lines.append(f"  {r['modelo']}: AP={r['ap_mean']:.4f} ROC={r['roc_auc_mean']:.4f} F1m={r['f1_macro_mean']:.4f} Recall={r['recall_no_uso_mean']:.4f} Precision={r['precision_no_uso_mean']:.4f}")
    lines.extend([f'\nGANADOR CV: {ganador}',
                  'TEST FINAL (evaluado solamente para el ganador):',
                  *[f'  {k}: {v:.6f}' for k,v in test_scores.items()],
                  f'Matriz confusion [real0,real1 x pred0,pred1]: {cm.tolist()}',
                  f'Parametros XGB optimizados: {json.dumps(busqueda.best_params_,ensure_ascii=False)}',
                  'SHAP: '+('SI; escala '+shap_res[0]+f', n {shap_res[1]}' if shap_res else 'NO o pendiente por error; revisar cierre19_error_shap.txt'),
                  'No se modifica ningun archivo ni modelo de pasos anteriores.',
                  'Precaucion: panel proporcionado corte 2025; no estimacion representativa sin diseno complejo.'])
    (BASE/'outputs'/'reporte_cierre19_clasificacion_asistentes_2025.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORTS/'19_Anexo_Clasificacion_Asistentes.md').write_text(
      '# Anexo: no uso reciente de internet entre estudiantes que asisten\n\n'
      '**Definicion:** ENAHO P306=1 (matriculado), P307=1 (asiste), P314A=2 (no usó internet el mes anterior). '
      'La variable no mide habilidades digitales ni notas escolares. No confundir con exclusión digital efectiva completa.\n\n'
      f'**Cohorte:** {len(df):,} personas, train {len(itr):,}, test {len(ite):,}; sin conglomerados compartidos.\n\n'
      '## Resultados de validación cruzada (solo train)\n\n'
      '| Modelo | AP | ROC-AUC | F1 macro | Recall no uso | Precisión no uso |\n|---|---:|---:|---:|---:|---:|\n'+
      ''.join(f"| {r['modelo']} | {r['ap_mean']:.4f} | {r['roc_auc_mean']:.4f} | {r['f1_macro_mean']:.4f} | {r['recall_no_uso_mean']:.4f} | {r['precision_no_uso_mean']:.4f} |\n" for _,r in comps.iterrows())+
      f'\n## Modelo seleccionado por CV: {ganador}\n\n'
      '**Test final:** '+', '.join(f'{k}={v:.4f}' for k,v in test_scores.items())+'\n\n'
      f'**Matriz de confusión** (filas real uso/no uso; columnas pred uso/no uso): `{cm.tolist()}`.\n\n'
      '**Precauciones:** el experimento C es distinto del A; NO comparar su test directamente con las cifras del modelo original. '
      'SUM_FACTOR07 se analizó solo como sensibilidad en entrenamiento, sin afirmar inferencia nacional. '
      'SHAP explica predicciones, no relaciones causales.\n\n'
      '**Figuras:** `outputs/figures/cierre19_*.png`. **Reportes:** `outputs/cierre_2025/`.\n',
      encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print('\nREPORTE: outputs/reporte_cierre19_clasificacion_asistentes_2025.txt',flush=True)

if __name__=='__main__':main()
