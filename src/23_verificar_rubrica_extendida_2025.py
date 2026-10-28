"""Auditoria automatica final de evidencias (NO asigna nota ni declara validez externa)."""
from pathlib import Path
import math
import pandas as pd
from pypdf import PdfReader

BASE=Path(__file__).resolve().parent.parent
O=BASE/'outputs'
FIG=O/'figures'
R=BASE/'reports'

REQUIRED=[
  'data/processed/dataset_enaho_escale_2025.csv',
  'outputs/reporte_conciliacion_minedu_2025.txt',
  'outputs/reporte_clustering_2025.txt',
  'outputs/reporte_clasificacion_2025.txt',
  'outputs/reporte_shap_2025.txt',
  'outputs/reporte_optimizacion_2025.txt',
  'outputs/reporte_paso18_xgboost_balanceo_objetivo_A.txt',
  'outputs/reporte_cierre19_clasificacion_asistentes_2025.txt',
  'outputs/reporte_cierre20_indice_brecha_2025.txt',
  'outputs/cierre_2025/cierre19_comparacion_cv.csv',
  'outputs/cierre_2025/cierre19_metricas_test.csv',
  'outputs/indice_brecha_2025/cierre20_cv_regresion.csv',
  'outputs/indice_brecha_2025/cierre20_metricas_test.csv',
  'outputs/models/clasificador_2025_asistentes_paso19.joblib',
  'outputs/models/regresion_indice_educativo_digital_paso20.joblib',
  'reports/Informe_Final_IEEE_ENAHO_ESCALE_2025_COMPLETO.pdf',
  'reports/19_Anexo_Clasificacion_Asistentes.md',
  'reports/20_Anexo_Indice_Brecha_Educativo_Digital.md',
  'outputs/figures/cierre19_matriz_confusion.png',
  'outputs/figures/cierre19_roc.png',
  'outputs/figures/cierre19_precision_recall.png',
  'outputs/figures/cierre19_calibracion.png',
  'outputs/figures/cierre19_shap_beeswarm.png',
  'outputs/figures/cierre19_shap_dependencia_1.png',
  'outputs/figures/cierre19_shap_dependencia_2.png',
  'outputs/figures/cierre19_shap_waterfall_no_uso.png',
  'outputs/figures/cierre19_shap_waterfall_uso.png',
  'outputs/figures/cierre20_regresion_real_vs_predicho.png',
  'outputs/figures/cierre20_distribucion_indice.png',
  'outputs/figures/cierre20_shap_beeswarm.png',
  'outputs/figures/cierre20_shap_dependencia_1.png',
  'outputs/figures/cierre20_shap_dependencia_2.png',
  'outputs/figures/cierre20_shap_waterfall_alto.png',
  'outputs/figures/cierre20_shap_waterfall_bajo.png',
]

def main():
    records=[]
    def check(cond,text):records.append((bool(cond),text));print(('OK: ' if cond else 'FALTA/REVISAR: ')+text,flush=True)
    for p in REQUIRED:check((BASE/p).is_file() and (BASE/p).stat().st_size>0,p)
    if not all(ok for ok,_ in records):
        # Los chequeos restantes dependen de ciertos productos.
        (O/'reporte_revision_rubrica_extendida_2025.txt').write_text(
            '\n'.join(('OK ' if ok else 'FALTA ')+label for ok,label in records)+'\n',encoding='utf-8')
        raise SystemExit('Hay archivos faltantes. Revisar reporte de verificacion; no declarar entrega completa.')
    pdf=PdfReader(str(R/'Informe_Final_IEEE_ENAHO_ESCALE_2025_COMPLETO.pdf'))
    check(len(pdf.pages)==6, f'Informe IEEE de 6 paginas (actual={len(pdf.pages)})')
    cv19=pd.read_csv(O/'cierre_2025'/'cierre19_comparacion_cv.csv')
    test19=pd.read_csv(O/'cierre_2025'/'cierre19_metricas_test.csv')
    cv20=pd.read_csv(O/'indice_brecha_2025'/'cierre20_cv_regresion.csv')
    test20=pd.read_csv(O/'indice_brecha_2025'/'cierre20_metricas_test.csv')
    noms=set(cv19['modelo'].astype(str))
    check(any('Regresion_Logistica' in n for n in noms),'Baseline clasificacion (Regresion Logistica)')
    check(any('Random_Forest' in n for n in noms),'Modelo avanzado RandomForest')
    check(any('MLP' in n for n in noms),'Modelo avanzado MLP')
    check(any('XGBoost' in n for n in noms),'Modelo avanzado XGBoost')
    check('XGBoost_con_pesos_clase' in noms and 'XGBoost_sin_pesos' in noms,'Balanceo clase comparado en CV')
    check('XGBoost_factor_sensibilidad' in noms,'Factor de expansion evaluado como sensibilidad en CV')
    check(len(cv19)>=7 and len(cv20)>=5,'Comparacion amplia de clasificadores y regresores')
    check(test19.shape[0]==1 and test20.shape[0]==1,'Un solo modelo final por experimento')
    for label,cols,table in [('clasificacion',['ap','roc_auc','f1_macro','recall_no_uso'],test19),
                              ('regresion',['mae','rmse','r2'],test20)]:
        check(all(c in table for c in cols) and all(math.isfinite(float(table[c].iloc[0])) for c in cols),
              f'Metricas finales validas de {label}')
    conciliacion=(O/'reporte_conciliacion_minedu_2025.txt').read_text(encoding='utf-8',errors='replace')
    check('VEREDICTO: COINCIDE' in conciliacion,'Conciliacion MINEDU coincide con el CSV maestro')
    resumen=[
       'EXAMEN IA ENAHO-ESCALE 2025 | AUDITORIA FINAL AMPLIADA',
       'Este chequeo automatizado verifica evidencias y coherencia estructural, NO adjudica 20/20.',
       f'Requisitos comprobados: {sum(ok for ok,_ in records)}/{len(records)}',
       *[('OK: ' if ok else 'FALTA/REVISAR: ')+t for ok,t in records],
       '',
       'VERIFICACION CIENTIFICA ADICIONAL:',
       '- El indice educativo-digital es una variable definida por el equipo, no un indicador oficial.',
       '- No uso reciente de internet NO equivale a no tener competencias digitales.',
       '- P307 se interpreta como asistencia segun cuestionario INEI; cotejar el ejemplar 2025.',
       '- La ponderacion con SUM_FACTOR07 fue un analisis de sensibilidad, NO inferencia nacional.',
       '- SHAP y clustering describen asociaciones del modelo; NO demuestran causalidad.',
       '- Revisar visualmente las 6 paginas del PDF y preparar defensa oral.',
    ]
    (O/'reporte_revision_rubrica_extendida_2025.txt').write_text('\n'.join(resumen)+'\n',encoding='utf-8')
    if any(not ok for ok,_ in records):raise SystemExit('Faltan evidencias; revisar reporte_revision_rubrica_extendida_2025.txt')
    print(f'REVISION ESTRUCTURAL APROBADA: {len(records)}/{len(records)}',flush=True)
    print('Reporte: outputs/reporte_revision_rubrica_extendida_2025.txt',flush=True)

if __name__=='__main__':main()
