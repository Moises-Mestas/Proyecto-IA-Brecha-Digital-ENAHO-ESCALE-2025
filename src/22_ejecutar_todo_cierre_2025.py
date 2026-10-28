"""Ejecucion unica en VS Code, SIN sobrescribir resultados de etapas anteriores.

Desde raiz de proyecto:
    python src/22_ejecutar_todo_cierre_2025.py
    python src/22_ejecutar_todo_cierre_2025.py --iteraciones 20

Aborta si falla cualquier paso. Registra salida en outputs/cierre_completo_2025.log.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import importlib.util

BASE=Path(__file__).resolve().parent.parent
SRC=BASE/'src'
LOG=BASE/'outputs'/'cierre_completo_2025.log'
STEPS=['19_cerrar_clasificacion_2025.py',
       '20_indice_brecha_educativa_digital_2025.py',
       '21_generar_informe_consolidado_2025.py',
       '23_verificar_rubrica_extendida_2025.py']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--iteraciones',type=int,default=12)
    a=p.parse_args()
    if not 1<=a.iteraciones<=50:raise ValueError('Iteraciones fuera de rango 1-50')
    pdf_candidates=[BASE/'reports'/'Informe_Final_IEEE_ENAHO_ESCALE_2025.pdf',
                    BASE/'Informe_Final_IEEE_ENAHO_ESCALE_2025.pdf']
    if not any(p.is_file() for p in pdf_candidates):
        raise SystemExit('ANTES DE ENTRENAR: coloca el PDF IEEE original de 5 paginas dentro de reports/.')
    if not (BASE/'data'/'processed'/'dataset_enaho_escale_2025.csv').is_file():
        raise SystemExit('ANTES DE ENTRENAR: falta data/processed/dataset_enaho_escale_2025.csv')
    if not (SRC/'11_clasificacion_2025.py').is_file():
        raise SystemExit('ANTES DE ENTRENAR: falta src/11_clasificacion_2025.py')
    deps=['numpy','pandas','sklearn','matplotlib','xgboost','shap','joblib','scipy','reportlab','pypdf']
    missing=[s for s in deps if importlib.util.find_spec(s) is None]
    if missing:
        raise SystemExit('Instala dependencias faltantes: python -m pip install '+ ' '.join(missing))
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('w',encoding='utf-8') as f:
        for name in STEPS:
            script=SRC/name
            if not script.is_file():raise FileNotFoundError(f'No esta el archivo {script}; extrae el ZIP en la raiz.')
            cmd=[sys.executable,str(script)]
            if name.startswith(('19_','20_')):cmd+=['--iteraciones',str(a.iteraciones)]
            label=f'\n=== EJECUTANDO {name} ===\n'
            print(label,flush=True);f.write(label);f.flush()
            with subprocess.Popen(cmd,cwd=str(BASE),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace',bufsize=1) as proc:
                for line in proc.stdout:
                    print(line,end='',flush=True)
                    f.write(line);f.flush()
                result=proc.wait()
            if result!=0:
                print(f'ERROR en {name}; codigo={result}. Ver outputs/cierre_completo_2025.log',flush=True)
                sys.exit(result)
    print('\nEXITO: 4 pasos completados. PDF completo: reports/Informe_Final_IEEE_ENAHO_ESCALE_2025_COMPLETO.pdf',flush=True)

if __name__=='__main__':main()
