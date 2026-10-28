"""ENAHO-ESCALE 2025 | PASO 20: alternativa B de la rubrica, indice CONTINUO.

Indice de vulnerabilidad educativo-digital (definido por el investigador):
  50% no uso reciente de internet: P314A=2
  30% no asistencia: P306=2, o (P306=1 y P307=2)
  20% presion educativa distrital: alumnos/docentes /(alumnos/docentes + 15)
Se incluyen solo registros validos para esos indicadores. La constante 15 es una
escala previa, sin estimarla con el test. En general 0<=indice<=1.
NO es una medicion oficial, ni resultado academico, ni medida de competencias.
Los pesos (0.50/0.30/0.20) son decisiones operacionales documentadas,
y se evalua su sensibilidad. Se PROHIBE incluir componentes directos como X.

Se compara Ridge baseline + RandomForest, HistGB, XGBoost y MLP de regresion.
GroupShuffleSplit y GroupKFold por CONGLOME. Ajustes solo dentro del train.
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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

SEED=42
BASE=Path(__file__).resolve().parent.parent
SRC=BASE/'src'/'11_clasificacion_2025.py'
OUT=BASE/'outputs'/'indice_brecha_2025'
FIG=BASE/'outputs'/'figures'
MODELS=BASE/'outputs'/'models'
REPORTS=BASE/'reports'


def paso11_import():
    if not SRC.is_file():raise FileNotFoundError(f'Falta {SRC}')
    s=importlib.util.spec_from_file_location('paso11_indice20',SRC)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def model_defs():
    from xgboost import XGBRegressor
    xgbr=XGBRegressor(n_estimators=170,max_depth=3,learning_rate=.06,subsample=.85,
           colsample_bytree=.9,min_child_weight=5,reg_lambda=2.,objective='reg:squarederror',
           tree_method='hist',random_state=SEED,n_jobs=2)
    return {
        'Media_BASE':DummyRegressor(strategy='mean'),
        'Ridge_BASE':Ridge(alpha=1.),
        'Random_Forest_Reg':RandomForestRegressor(n_estimators=170,max_depth=12,min_samples_leaf=4,n_jobs=2,random_state=SEED),
        'HistGB_Reg':HistGradientBoostingRegressor(max_iter=130,learning_rate=.07,max_leaf_nodes=17,l2_regularization=1.,random_state=SEED),
        'XGBoost_Reg':xgbr,
        'MLP_Reg':MLPRegressor(hidden_layer_sizes=(32,16),max_iter=230,alpha=.015,early_stopping=True,n_iter_no_change=12,random_state=SEED),
    }


def metrics(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),
                r2=float(r2_score(y,p)))


def main():
    par=argparse.ArgumentParser()
    par.add_argument('--iteraciones',type=int,default=12)
    par.add_argument('--sin-shap',action='store_true')
    args=par.parse_args()
    if not 1<=args.iteraciones<=50:raise ValueError('Iteraciones validas de 1 a 50')
    for d in (OUT,FIG,MODELS,REPORTS):d.mkdir(parents=True,exist_ok=True)
    paso11=paso11_import()
    df=pd.read_csv(paso11.ENTRADA,dtype=str,encoding='utf-8-sig',keep_default_na=False,low_memory=False)
    df.columns=df.columns.str.strip().str.upper()
    req=['EDU_P314A','EDU_P306','EDU_P307','MIN_RATIO_ALUMNOS_DOCENTE','CONGLOME','AÑO','SUM_FACTOR07']
    # Excluir componentes directos de X: EDU*, P1144 si se incorporase al indice (aqui no).
    predictor_num=['EDAD','SUM_INGHOG2D','SUM_GASHOG2D','MIN_PROP_PUBLICOS']
    predictor_cat=['P207','SUM_POBREZA','VIV_P1121','VIV_P1144']
    falt=[c for c in req+predictor_num+predictor_cat if c not in df]
    if falt:raise ValueError('Columnas faltantes: '+str(falt))
    if not df['AÑO'].str.strip().eq('2025').all():raise ValueError('El panel no esta filtrado a 2025.')
    for c in ['EDU_P314A','EDU_P306','EDU_P307','CONGLOME']:
        df[c]=df[c].astype(str).str.strip()
    ratio=pd.to_numeric(df['MIN_RATIO_ALUMNOS_DOCENTE'].str.replace(',','.',regex=False),errors='coerce')
    fac=pd.to_numeric(df['SUM_FACTOR07'].str.replace(',','.',regex=False),errors='coerce')
    ok=df['EDU_P314A'].isin(['1','2']) & df['EDU_P306'].isin(['1','2']) & ratio.ge(0) & np.isfinite(ratio)
    ok &= df['EDU_P306'].eq('2') | (df['EDU_P306'].eq('1')&df['EDU_P307'].isin(['1','2']))
    # Son componentes verificados del indice. No imputar las respuestas de cuestionario.
    df=df.loc[ok].copy();ratio=ratio.loc[ok];fac=fac.loc[ok]
    if len(df)<500:raise ValueError('Muy pocas filas con componentes completos del indice.')
    falta_escuela=(df['EDU_P306'].eq('2') | df['EDU_P307'].eq('2')).astype(float)
    sin_internet=df['EDU_P314A'].eq('2').astype(float)
    presion=ratio/(ratio+15.)
    y=.50*sin_internet + .30*falta_escuela + .20*presion
    if not np.isfinite(y).all() or y.min()<0 or y.max()>1:
        raise AssertionError('Indice fuera de rango.')
    # Otras dos especificaciones de pesos: prueba de sensibilidad del constructo.
    variantes=pd.DataFrame({'indice_principal':y,
               'sensibilidad_digital_alta':.65*sin_internet+.20*falta_escuela+.15*presion,
               'sensibilidad_educacion_alta':.35*sin_internet+.45*falta_escuela+.20*presion})
    variantes.corr().to_csv(OUT/'cierre20_correlacion_variantes_indice.csv',encoding='utf-8-sig')
    X=df[predictor_num+predictor_cat].copy()
    for c in predictor_num:
        X[c]=pd.to_numeric(X[c].str.replace(',','.',regex=False).replace('',np.nan),errors='coerce')
    for c in predictor_cat:X[c]=X[c].replace('',np.nan).astype(object)
    grupos=df['CONGLOME']
    if grupos.eq('').any():raise ValueError('CONGLOME vacio')
    # preprocesador basado en paso11, pero SIN ratio: esta dentro del objetivo.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    num=Pipeline([('imputar',SimpleImputer(strategy='median')),('escalar',StandardScaler())])
    cat=Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),
                  ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    def preparar():
        return ColumnTransformer([('numericas',clone(num),predictor_num),('categoricas',clone(cat),predictor_cat)],
                                 sparse_threshold=0)
    itr,ite=next(GroupShuffleSplit(test_size=.25,n_splits=1,random_state=SEED).split(X,y,groups=grupos))
    Xtr,Xte=X.iloc[itr],X.iloc[ite];ytr,yte=y.iloc[itr],y.iloc[ite]
    gtr,gte=grupos.iloc[itr],grupos.iloc[ite]
    if set(gtr)&set(gte):raise AssertionError('Fuga de conglomerados')
    folds=list(GroupKFold(n_splits=3).split(Xtr,ytr,groups=gtr))
    print(f'INDICE B. Registros completos={len(df)}, train={len(itr)}, test={len(ite)}, grupos compartidos=0.',flush=True)
    filas=[];modelos=model_defs()
    for nombre,reg in modelos.items():
        for fold,(a,b) in enumerate(folds,1):
            pipe=Pipeline([('preprocesar',preparar()),('modelo',clone(reg))])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipe.fit(Xtr.iloc[a],ytr.iloc[a])
            res=metrics(ytr.iloc[b],pipe.predict(Xtr.iloc[b]))
            filas.append({'modelo':nombre,'fold':fold,**res})
            print(f'{nombre} fold {fold}: RMSE={res["rmse"]:.4f} MAE={res["mae"]:.4f} R2={res["r2"]:.4f}',flush=True)
    # Buscar 12 combinaciones de XGBoost (solo dentro de train), sin tocar test.
    from xgboost import XGBRegressor
    search=RandomizedSearchCV(Pipeline([('preprocesar',preparar()),
            ('modelo',XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=SEED,n_jobs=2))]),
            {'modelo__n_estimators':[90,160,250,330],
             'modelo__max_depth':[2,3,4,5],
             'modelo__learning_rate':[.025,.05,.08,.11],
             'modelo__min_child_weight':[2,5,10],
             'modelo__subsample':[.75,.9,1.],
             'modelo__reg_lambda':[1.,3.,6.]},
            scoring='neg_root_mean_squared_error',n_iter=args.iteraciones,random_state=SEED,
            cv=GroupKFold(3),n_jobs=1,refit=True,error_score='raise')
    print(f'Buscando {args.iteraciones} combinaciones XGBRegressor en TRAIN...',flush=True)
    search.fit(Xtr,ytr,groups=gtr)
    pd.DataFrame(search.cv_results_)[['rank_test_score','mean_test_score','std_test_score','params']]\
        .sort_values('rank_test_score').to_csv(OUT/'cierre20_busqueda_regresion.csv',index=False,encoding='utf-8-sig')
    bestreg=search.best_estimator_.named_steps['modelo']
    for fold,(a,b) in enumerate(folds,1):
        p=Pipeline([('preprocesar',preparar()),('modelo',clone(bestreg))]);p.fit(Xtr.iloc[a],ytr.iloc[a])
        filas.append({'modelo':'XGBoost_Reg_optimizado','fold':fold,**metrics(ytr.iloc[b],p.predict(Xtr.iloc[b]))})
    detalle=pd.DataFrame(filas)
    detalle.to_csv(OUT/'cierre20_folds_regresion.csv',index=False,encoding='utf-8-sig')
    resumen=detalle.groupby('modelo',sort=False)[['mae','rmse','r2']].agg(['mean','std'])
    resumen.columns=['_'.join(c) for c in resumen.columns.to_flat_index()]
    resumen=resumen.reset_index().sort_values('rmse_mean')
    resumen.to_csv(OUT/'cierre20_cv_regresion.csv',index=False,encoding='utf-8-sig')
    ganador=resumen.iloc[0]['modelo']
    reg=bestreg if ganador=='XGBoost_Reg_optimizado' else modelos[ganador]
    final=Pipeline([('preprocesar',preparar()),('modelo',clone(reg))])
    final.fit(Xtr,ytr)
    pred=final.predict(Xte)
    test=metrics(yte,pred)
    # A esta altura test se consulta solo para modelo ganador elegido mediante CV.
    joblib.dump(final,MODELS/'regresion_indice_educativo_digital_paso20.joblib')
    pd.DataFrame([{'modelo':ganador,**test,'n_test':len(yte)}]).to_csv(OUT/'cierre20_metricas_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'real':np.asarray(yte),'predicho':pred}).to_csv(OUT/'cierre20_test_sin_ids.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(6,4.6))
    ax.scatter(yte,pred,alpha=.22,s=13,rasterized=True)
    ax.plot([0,1],[0,1],linestyle='--',color='gray')
    ax.set(xlabel='Indice observado (construido)',ylabel='Indice predicho',title='Regresion: brecha educativo-digital - test agrupado',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(FIG/'cierre20_regresion_real_vs_predicho.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax.hist(yte,bins=30,alpha=.7,label='Indice real')
    ax.hist(pred,bins=30,alpha=.5,label='Predicho')
    ax.set(xlabel='Indice construido',ylabel='Frecuencia',title='Distribucion del indice en test')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'cierre20_distribucion_indice.png',dpi=160);plt.close(fig)
    shap_info='Sin calcular (se uso --sin-shap)'
    if not args.sin_shap:
        try:
            import shap
            pre=final.named_steps['preprocesar'];mdl=final.named_steps['modelo']
            rng=np.random.default_rng(SEED)
            indices=np.sort(rng.choice(len(Xte),size=min(220,len(Xte)),replace=False))
            z=np.asarray(pre.transform(Xte.iloc[indices]),dtype=float)
            nombres=list(pre.get_feature_names_out())
            nombres=[s.replace('numericas__','').replace('categoricas__','') for s in nombres]
            try:
                expl=shap.TreeExplainer(mdl)
                sv=expl(z)
            except (ValueError,TypeError,AssertionError):
                bg=np.asarray(pre.transform(Xte.iloc[:min(90,len(Xte))]),dtype=float)
                expl=shap.Explainer(mdl.predict,shap.maskers.Independent(bg),algorithm='permutation')
                sv=expl(z,max_evals=2*z.shape[1]+1)
            sv.feature_names=nombres;sv.data=z
            if sv.values.ndim!=2:raise ValueError('SHAP de regresion: arreglo no escalar')
            e=float(np.max(np.abs(np.asarray(sv.base_values).reshape(-1)+sv.values.sum(axis=1)-mdl.predict(z))))
            if e>.03:raise AssertionError(f'SHAP no aditivo: {e}')
            imp=np.abs(sv.values).mean(0)
            pd.DataFrame({'variable':nombres,'abs_shap_medio':imp}).sort_values('abs_shap_medio',ascending=False).to_csv(
                          OUT/'cierre20_importancia_shap.csv',index=False,encoding='utf-8-sig')
            plt.close('all');shap.plots.beeswarm(sv,max_display=10,show=False)
            plt.title('SHAP - indice educativo-digital (regresion)')
            plt.savefig(FIG/'cierre20_shap_beeswarm.png',dpi=160,bbox_inches='tight');plt.close('all')
            # Dependencia de las dos variables numericas con mayor efecto.
            ids_num=sorted([nombres.index(k) for k in predictor_num],key=lambda i:imp[i],reverse=True)[:2]
            for j,k in enumerate(ids_num,1):
                plt.close('all');shap.plots.scatter(sv[:,k],show=False)
                plt.title('SHAP dependencia: '+nombres[k]);plt.savefig(FIG/f'cierre20_shap_dependencia_{j}.png',dpi=160,bbox_inches='tight');plt.close('all')
            hi=int(np.argmax(yte.iloc[indices].to_numpy()))
            lo=int(np.argmin(yte.iloc[indices].to_numpy()))
            for k,ind in [('alto',hi),('bajo',lo)]:
                plt.close('all');shap.plots.waterfall(sv[ind],max_display=10,show=False)
                plt.title('Caso anonimo indice '+k);plt.savefig(FIG/f'cierre20_shap_waterfall_{k}.png',dpi=160,bbox_inches='tight');plt.close('all')
            shap_info=f'Beeswarm, 2 dependencias y 2 waterfalls, error max aditivo {e:.6f}, n={len(indices)}'
        except Exception as ex:
            (OUT/'cierre20_error_shap.txt').write_text(traceback.format_exc(),encoding='utf-8')
            shap_info=f'ERROR: {ex} (revisar cierre20_error_shap.txt)'
    lines=['ENAHO-ESCALE | PASO 20 | OPCION B (INDICE CONTINUO, PROPUESTO)',
      'Este indice es una construccion explicita del equipo, no un indicador oficial.',
      'Definicion: 0.50 * NO uso reciente de internet + 0.30 * NO asistencia educativa +',
      '0.20 * (alumnos por docente)/(alumnos por docente + 15).',
      'NO asistencia = NO matriculado O matriculado pero NO asiste.',
      'La escala 15 es prefijada, no ajustada en datos.',
      'Las 3 variables de origen del objetivo se excluyen de X para evitar fuga directa.',
      f'Universo n={len(df):,}; train={len(itr):,}; test={len(ite):,}; sin conglomerados compartidos.',
      f'Indice min/mediana/max={y.min():.4f}/{y.median():.4f}/{y.max():.4f}',
      f'Sensibilidad corr digital alta={variantes.indice_principal.corr(variantes.sensibilidad_digital_alta):.4f}',
      f'Sensibilidad corr educacion alta={variantes.indice_principal.corr(variantes.sensibilidad_educacion_alta):.4f}',
      'CV 3 folds SOLO train, transformaciones dentro de cada fold. Se evaluo TEST una sola vez.',
      f'XGBoost busqueda n={args.iteraciones}, mejor RMSE CV={-search.best_score_:.5f}.',
      'MODELOS EN CV (orden RMSE menor mejor):']
    for _,r in resumen.iterrows():
        lines.append(f"  {r['modelo']}: RMSE={r['rmse_mean']:.5f}, MAE={r['mae_mean']:.5f}, R2={r['r2_mean']:.5f}")
    lines.extend([f'GANADOR CV: {ganador}',
       f'TEST: RMSE={test["rmse"]:.5f}, MAE={test["mae"]:.5f}, R2={test["r2"]:.5f}',
       'SHAP: '+shap_info,
       'No se usa factor expansion para inferencia nacional.',
       'Indice es experimental, componentes y pesos son decisiones de investigacion.',
       'No es calificacion, competencia digital ni muestra de representatividad nacional.'])
    (BASE/'outputs'/'reporte_cierre20_indice_brecha_2025.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORTS/'20_Anexo_Indice_Brecha_Educativo_Digital.md').write_text(
       '# Alternativa B - Indice continuo educativo-digital (propuesto)\n\n'
       '**Definicion reproducible:** I = 0.50 × I(no usó internet el mes anterior, P314A=2) + '
       '0.30 × I(no asiste: P306=2 o P307=2 si matriculado) + '
       '0.20 × ratio_alumnos_docente/(ratio_alumnos_docente + 15).\n\n'
       '**Aviso:** constructo exploratorio no oficial, sin validación externa; no mide calificaciones ni competencias digitales. '
       'La escuela del distrito no implica institución a la que asiste cada estudiante.\n\n'
       f'Se analizaron **{len(df):,}** registros completos; train={len(itr):,}, test={len(ite):,}. '
       'GroupKFold 3 particiones y conjunto de test separado por conglomerado. '
       '**Se excluyeron de X las variables que construyen directamente Y.**\n\n'
       '## Modelos: CV de TRAIN\n\n| Modelo | RMSE | MAE | R² |\n|---|---:|---:|---:|\n'+
       ''.join(f"| {r['modelo']} | {r['rmse_mean']:.4f} | {r['mae_mean']:.4f} | {r['r2_mean']:.4f} |\n" for _,r in resumen.iterrows())+
       f'\n**Seleccionado:** {ganador}. **Test:** RMSE={test["rmse"]:.4f}, MAE={test["mae"]:.4f}, R²={test["r2"]:.4f}.\n\n'
       'Sensibilidad del indice al peso de componentes: correlaciones registradas en `outputs/indice_brecha_2025/`. '
       'No se infiere causalidad ni se declara representatividad poblacional.\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print('\nREPORTE: outputs/reporte_cierre20_indice_brecha_2025.txt',flush=True)

if __name__=='__main__':main()
