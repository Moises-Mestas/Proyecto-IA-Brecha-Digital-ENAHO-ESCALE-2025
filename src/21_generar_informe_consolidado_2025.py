"""Genera anexo IEEE de 1 pagina y lo adjunta al PDF IEEE de 5 paginas.

Combina el informe original y los resultados REALES de pasos 18, 19 y 20.
NO modifica el PDF original. PDF final esperado: 6 paginas.
Entradas: reportes y CSV de los pasos 19 y 20 (no genera metricas).
Requiere reportlab y pypdf.
"""
from pathlib import Path
import tempfile
import re
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader

BASE=Path(__file__).resolve().parent.parent
REPORTS=BASE/'reports'
OUT=BASE/'outputs'
FOLDER=OUT/'cierre_2025'
INDICE=OUT/'indice_brecha_2025'
PDF_OUT=REPORTS/'Informe_Final_IEEE_ENAHO_ESCALE_2025_COMPLETO.pdf'
RUTA_A=REPORTS/'Informe_Final_IEEE_ENAHO_ESCALE_2025.pdf'
if not RUTA_A.is_file():RUTA_A=BASE/'Informe_Final_IEEE_ENAHO_ESCALE_2025.pdf'


def formato(x,n=3):
    return f'{float(x):.{n}f}'


def parrafo(c,raw,x,y,w,size=8,leading=10,color='#242424',bold=False):
    st=ParagraphStyle('texto',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,
                      leading=leading,textColor=HexColor(color),alignment=TA_LEFT,spaceAfter=0)
    p=Paragraph(raw,st)
    pw,ph=p.wrap(w,900)
    p.drawOn(c,x,y-ph)
    return y-ph-6


def titulo(c,txt,x,y,w):
    return parrafo(c,txt,x,y,w,size=9,leading=11,color='#123A59',bold=True)


def tabla(c,headers,rows,x,y,w,relative=None):
    data=[headers]+rows
    widths=[w*a for a in relative] if relative else [w/len(headers)]*len(headers)
    tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([
       ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('BACKGROUND',(0,0),(-1,0),HexColor('#E6EDF4')),
       ('TEXTCOLOR',(0,0),(-1,-1),HexColor('#263644')),
       ('FONTSIZE',(0,0),(-1,-1),6.5),('LEADING',(0,0),(-1,-1),8.2),
       ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),3),
       ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
       ('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#C0CDD8')),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#FFFFFF'),HexColor('#F5F7FA')])
    ]))
    _,h=tab.wrap(w,800)
    tab.drawOn(c,x,y-h)
    return y-h-9


def imagen(c,path,x,y,w,max_h):
    if not path.is_file():return y
    im=ImageReader(str(path));iw,ih=im.getSize()
    scale=min(w/iw,max_h/ih)
    dw,dh=iw*scale,ih*scale
    c.drawImage(im,x+(w-dw)/2,y-dh,dw,dh,mask='auto',preserveAspectRatio=True)
    return y-dh-7


def crear_pagina_anexo(archivo,cv19,mt19,cv20,mt20):
    from reportlab.lib.pagesizes import A4
    W,H=A4
    c=canvas.Canvas(str(archivo),pagesize=A4)
    c.setFillColor(HexColor('#142E43'))
    c.rect(0,H-84,W,84,fill=1,stroke=0)
    c.setFont('Helvetica-Bold',14);c.setFillColorRGB(1,1,1)
    c.drawString(35,H-39,'ANEXO | COMPROBACIONES COMPLEMENTARIAS')
    c.setFont('Helvetica',8);c.drawString(35,H-56,'ENAHO - ESCALE 2025  |  Revision de modelos y operacionalizaciones')
    c.setFont('Helvetica',7);c.drawRightString(W-34,H-75,'6 / 6')
    x1=35;x2=307;gap=18;width=(W-70-gap)/2
    top=H-105
    yl=titulo(c,'1. Ampliacion de la clasificacion',x1,top,width)
    yl=parrafo(c,'Se agregaron <b>XGBoost</b> y experimentos con pesos de clase al objetivo original A '
           '(no uso reciente de internet). Se compararon mediante <b>GroupKFold de 3 particiones sobre TRAIN</b>; '
           'estas cifras no son del test.',x1,yl,width,size=7.7,leading=10)
    inf18=OUT/'reporte_paso18_xgboost_balanceo_objetivo_A.txt'
    if inf18.is_file():
        # Resumen de CV original, validado previamente en reporte paso18.
        lista=[('XGB sin pesos','0.6649','0.4806'),('XGB con pesos','0.6647','0.7542'),
               ('HistGB con pesos','0.6611','0.7485'),('HistGB original','0.6593','0.4720')]
        yl=tabla(c,['Experimento A','AP (CV)','Recall'],lista,x1,yl,width,[.58,.22,.20])
    else:
        yl=parrafo(c,'Comparacion original A: ver <b>outputs/reporte_paso18_xgboost_balanceo_objetivo_A.txt</b>.',x1,yl,width,size=7)
    yl=titulo(c,'2. Clasificacion C: estudiantes asistentes',x1,yl-1,width)
    yl=parrafo(c,'Nueva cohorte: matriculado <b>P306=1</b>, asiste <b>P307=1</b>; positivo '
       '<b>P314A=2</b> (no uso de internet el mes anterior). Esto sigue sin medir habilidades digitales.',x1,yl,width,size=7.6,leading=10)
    mt=mt19.iloc[0]
    yl=parrafo(c, f"<b>Modelo elegido por CV:</b> {mt['modelo']}. <b>Test agrupado:</b> AP {formato(mt['ap'])}; "
      f"ROC-AUC {formato(mt['roc_auc'])}; F1 macro {formato(mt['f1_macro'])}; "
      f"Recall {formato(mt['recall_no_uso'])}. N prueba: {int(mt['n_test']):,}.",
      x1,yl,width,size=7.6,leading=10)
    sel19=cv19.head(4)
    yl=tabla(c,['Modelo (CV train)','AP','Recall'],[[str(r['modelo'])[:26],formato(r['ap_mean']),formato(r['recall_no_uso_mean'])]
      for _,r in sel19.iterrows()],x1,yl,width,[.61,.18,.21])
    yl=imagen(c,OUT/'figures'/'cierre19_matriz_confusion.png',x1,yl,width,172)
    yr=titulo(c,'3. Opcion B: indice continuo propuesto',x2,top,width)
    yr=parrafo(c,'Se construyo un indice propio (escala 0-1): <b>0.50</b> por no uso de internet; '
       '<b>0.30</b> por no asistencia a estudios y <b>0.20</b> por presion educativa distrital '
       '(alumnos/docente dividido por alumnos/docente + 15). Su definicion no es un estandar INEI.',
       x2,yr,width,size=7.6,leading=10)
    mt=mt20.iloc[0]
    yr=parrafo(c,f"<b>Modelo elegido por CV:</b> {mt['modelo']}. En test: RMSE {formato(mt['rmse'])}; "
        f"MAE {formato(mt['mae'])}; R2 {formato(mt['r2'])}. N prueba: {int(mt['n_test']):,}.",
        x2,yr,width,size=7.6,leading=10)
    sel20=cv20.head(4)
    yr=tabla(c,['Modelo (CV train)','RMSE','R2'],[[str(r['modelo'])[:26],formato(r['rmse_mean']),formato(r['r2_mean'])]
      for _,r in sel20.iterrows()],x2,yr,width,[.63,.19,.18])
    yr=imagen(c,OUT/'figures'/'cierre20_regresion_real_vs_predicho.png',x2,yr,width,178)
    yr=titulo(c,'4. Control metodologico',x2,yr-2,width)
    yr=parrafo(c,'<b>Sin fugas:</b> particion por conglomerado; transformaciones ajustadas en '
         'cada entrenamiento; indicadores que definen el indice excluidos de sus predictores. '
         '<b>Desbalance:</b> pesos calculados dentro del fold. '
         '<b>Factor07:</b> sensibilidad en train; sin inferencia nacional porque no se incorporo '
         'el diseno muestral complejo. SHAP solo explica predicciones, no causalidad.',
         x2,yr,width,size=7.2,leading=9.6)
    lower=min(yl,yr)
    if lower<38:
        raise RuntimeError(f'El anexo IEEE no cabe en pagina unica: y={lower:.1f}. Reducir texto/tablas antes de exportar.')
    c.setStrokeColor(HexColor('#8FA6B8'));c.line(35,25,W-35,25)
    c.setFont('Helvetica',7);c.setFillColor(HexColor('#3E5565'))
    c.drawString(35,15,'Fuente: microdatos del examen, corte 2025 | analisis reproducible por grupos')
    c.save()


def main():
    if not RUTA_A.is_file():raise FileNotFoundError(f'No encuentro el PDF IEEE original: {RUTA_A}. Guardalo en reports/.')
    cv19=pd.read_csv(FOLDER/'cierre19_comparacion_cv.csv')
    mt19=pd.read_csv(FOLDER/'cierre19_metricas_test.csv')
    cv20=pd.read_csv(INDICE/'cierre20_cv_regresion.csv')
    mt20=pd.read_csv(INDICE/'cierre20_metricas_test.csv')
    ini=PdfReader(str(RUTA_A))
    if len(ini.pages)!=5:
        raise RuntimeError(f'El informe original debe tener exactamente 5 paginas, ahora tiene {len(ini.pages)}. No lo sobrescribire.')
    with tempfile.TemporaryDirectory(prefix='anexo_ieee_') as temp:
        hoja=Path(temp)/'anexo.pdf'
        crear_pagina_anexo(hoja,cv19,mt19,cv20,mt20)
        anexo=PdfReader(str(hoja))
        if len(anexo.pages)!=1:raise AssertionError('El anexo debe caber en 1 pagina.')
        escritor=PdfWriter()
        for p in ini.pages:escritor.add_page(p)
        escritor.add_page(anexo.pages[0])
        with open(PDF_OUT,'wb') as f:escritor.write(f)
    pages=len(PdfReader(str(PDF_OUT)).pages)
    if pages!=6:raise AssertionError(f'PDF generado incorrecto: {pages} paginas')
    resumen=(REPORTS/'21_Resumen_de_Entrega_Final.md')
    resumen.write_text(
      '# Entrega final ENAHO - ESCALE 2025\n\n'
      'El archivo **Informe_Final_IEEE_ENAHO_ESCALE_2025_COMPLETO.pdf** contiene 6 paginas: '
      'articulo original de cinco paginas y una nueva pagina con los experimentos de clasificacion C, '
      'regresion del indice B y comparacion XGBoost/pesos del objetivo original A.\n\n'
      '**Nota:** los objetivos A, B y C son diferentes; sus metricas no deben compararse '
      'directamente entre si. El indice B es una operacionalizacion del equipo, no un indicador oficial.\n\n'
      '## Archivos para sustentar\n\n'
      '- `outputs/reporte_cierre19_clasificacion_asistentes_2025.txt`\n'
      '- `outputs/reporte_cierre20_indice_brecha_2025.txt`\n'
      '- `outputs/cierre_2025/cierre19_comparacion_cv.csv`\n'
      '- `outputs/indice_brecha_2025/cierre20_cv_regresion.csv`\n'
      '- `outputs/figures/cierre19_shap_*.png`\n'
      '- `outputs/figures/cierre20_shap_*.png`\n'
      '- `reports/19_Anexo_Clasificacion_Asistentes.md`\n'
      '- `reports/20_Anexo_Indice_Brecha_Educativo_Digital.md`\n\n'
      '## Verificar el PDF\n\n'
      'Abrir en un lector PDF y comprobar que son 6 paginas, que las tablas se leen y que '
      'la ultima pagina muestra las metricas reales de los pasos 19 y 20.\n',encoding='utf-8')
    print(f'PDF final de {pages} paginas: {PDF_OUT}',flush=True)
    print(f'Guia de entrega: {resumen}',flush=True)

if __name__=='__main__':main()
