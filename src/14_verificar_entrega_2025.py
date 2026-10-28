"""Paso 14. Verificacion reproducible de las evidencias del examen.

Desde la raiz del proyecto:
    python src/14_verificar_entrega_2025.py

NO modifica los CSV, ZIP, modelos ni graficas. Crea un reporte y manifiesto.
Requiere pandas, ya utilizado en las practicas anteriores.
"""
from pathlib import Path
from hashlib import sha256
import csv
from datetime import datetime
import struct
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
REPORTE = BASE / 'outputs' / 'reporte_revision_entrega_2025.txt'
MANIFIESTO = BASE / 'outputs' / 'manifiesto_entrega_2025.csv'

REQUERIDOS = {
    'Fuentes INEI': [
        'data/raw/inei/1041-Modulo1474.zip',
        'data/raw/inei/1041-Modulo1475.zip',
        'data/raw/inei/1041-Modulo1478.zip',
        'data/raw/inei/1041-Modulo1479.zip',
    ],
    'Fuentes MINEDU': [
        'data/raw/minedu/00_Padron.zip',
        'data/raw/minedu/00_Padlocal.zip',
    ],
    'Dataset y diccionario': [
        'data/processed/dataset_enaho_escale_2025.csv',
        'outputs/diccionario_columnas_maestro_2025.csv',
    ],
    'Clustering': [
        'outputs/reporte_clustering_2025.txt',
        'outputs/clustering/comparacion_modelos_2025.csv',
        'outputs/clustering/perfiles_clusters_2025.csv',
        'outputs/figures/clustering_pca_2025.png',
        'outputs/figures/clustering_silueta_2025.png',
    ],
    'Clasificacion': [
        'outputs/reporte_clasificacion_2025.txt',
        'outputs/clasificacion/comparacion_modelos_cv_2025.csv',
        'outputs/clasificacion/metricas_test_2025.csv',
        'outputs/figures/clasificacion_roc_2025.png',
        'outputs/figures/clasificacion_precision_recall_2025.png',
        'outputs/figures/clasificacion_matriz_confusion_2025.png',
        'outputs/figures/clasificacion_calibracion_2025.png',
        'outputs/models/clasificador_2025.joblib',
    ],
    'SHAP': [
        'outputs/reporte_shap_2025.txt',
        'outputs/shap/importancia_shap_top_variables_2025.csv',
        'outputs/figures/shap_beeswarm_top10_2025.png',
        'outputs/figures/shap_dependencia_1_2025.png',
        'outputs/figures/shap_dependencia_2_2025.png',
        'outputs/figures/shap_waterfall_no_uso_real_2025.png',
        'outputs/figures/shap_waterfall_uso_real_2025.png',
    ],
    'Articulo y documentacion': [
        'reports/14_Articulo_ENAHO_ESCALE_2025.md',
        'outputs/figures/diagrama_integracion_enaho_escale_2025.png',
        'src/09_clustering_estudiantes_2025.py',
        'src/11_clasificacion_2025.py',
        'src/12_explicabilidad_shap_2025.py',
    ],
}


def huella(ruta):
    digest = sha256()
    with ruta.open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def dimensiones_png(ruta):
    with ruta.open('rb') as f:
        cabecera = f.read(24)
    if len(cabecera) < 24 or cabecera[:8] != b'\x89PNG\r\n\x1a\n':
        return 'NO ES PNG VALIDO'
    ancho, alto = struct.unpack('>II', cabecera[16:24])
    return f'{ancho}x{alto}'


def analizar_maestro(lineas):
    ruta = BASE / 'data/processed/dataset_enaho_escale_2025.csv'
    if not ruta.is_file():
        lineas.append('NO SE PUEDE AUDITAR EL DATASET MAESTRO: archivo ausente.')
        return
    try:
        df = pd.read_csv(ruta, dtype=str, encoding='utf-8-sig',
                         keep_default_na=False, low_memory=False)
        df.columns = df.columns.str.strip().str.upper()
        lineas.append(f'Filas: {len(df):,} | Columnas: {len(df.columns)}')
        llave = ['AÑO','CONGLOME','VIVIENDA','HOGAR','CODPERSO']
        faltantes = [c for c in llave + ['UBIGEO','EDAD','EDU_P314A'] if c not in df]
        if faltantes:
            lineas.append(f'ERROR: columnas esenciales faltantes: {faltantes}')
            return
        df[llave] = df[llave].apply(lambda col: col.str.strip())
        duplicados = int(df.duplicated(llave).sum())
        incompletas = int(df[llave].eq('').any(axis=1).sum())
        ubigeos = int(df['UBIGEO'].str.strip().str.fullmatch(r'\d{6}').fillna(False).sum())
        edades = pd.to_numeric(df['EDAD'].str.strip(), errors='coerce')
        edad_valida = int(edades.between(6,17).sum())
        anios = sorted(df['AÑO'].unique().tolist())
        codigos = df['EDU_P314A'].str.strip().replace('', 'SIN_DATO').value_counts().to_dict()
        lineas += [
            f'Llaves duplicadas: {duplicados}; incompletas: {incompletas}',
            f'UBIGEO de 6 digitos: {ubigeos}/{len(df)}',
            f'Edades 6–17: {edad_valida}/{len(df)}',
            f'Años detectados: {anios}',
            f'Distribucion de P314A: {codigos}',
        ]
        if duplicados or incompletas or edad_valida != len(df) or ubigeos != len(df):
            lineas.append('ADVERTENCIA: alguna validación estructural no pasó.')
        else:
            lineas.append('OK: controles estructurales de la tabla maestra.')
    except Exception as exc:
        lineas.append(f'ERROR al leer CSV maestro: {type(exc).__name__}: {exc}')


def main():
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    lineas = ['EXAMEN IA ENAHO–ESCALE 2025 | AUDITORIA DE ENTREGA',
              'Solo archivos del docente y productos de los pasos anteriores.',
              'El chequeo NO sustituye la verificacion metodologica del docente.', '']
    manifiesto = []
    total = 0
    encontrados = 0
    for seccion, rutas in REQUERIDOS.items():
        lineas.append(f'--- {seccion} ---')
        for nombre in rutas:
            total += 1
            ruta = BASE / nombre
            ok = ruta.is_file() and ruta.stat().st_size > 0
            encontrados += int(ok)
            tamano = ruta.stat().st_size if ok else 0
            detalles = ''
            if ok and ruta.suffix.lower() == '.png':
                detalles = dimensiones_png(ruta)
            # Huellas de archivos de tamaño moderado; no rehacer hashing de los ZIP grandes.
            checksum = huella(ruta) if ok and tamano < 30_000_000 else ''
            lineas.append(f'{"OK" if ok else "FALTA"}: {nombre}' + (f' [{detalles}]' if detalles else ''))
            manifiesto.append({'seccion':seccion, 'archivo':nombre,
                               'estado':'OK' if ok else 'FALTA',
                               'tamano_bytes':tamano, 'sha256':checksum})
        lineas.append('')
    lineas.append(f'ARCHIVOS DE EVIDENCIA: {encontrados}/{total} presentes')
    lineas.append('\n--- Controles del dataset maestro ---')
    analizar_maestro(lineas)
    lineas.extend([
        '', '--- Observaciones metodologicas pendientes (verificacion manual) ---',
        '1. OBJETIVO: P314A=2 mide NO USO RECIENTE, no equivale a exclusion digital efectiva ni rendimiento escolar.',
        '2. MUESTRA: panel ENAHO 2021–2025, corte 2025, provisto en el examen; no equiparar al anual completo.',
        '3. DISEÑO: no se ha hecho inferencia poblacional con factores muestrales ni diseño complejo.',
        '4. HIPERPARAMETROS: falta documentar una busqueda sistematica si se desea cubrir ese criterio de la rubrica.',
        '5. ARTICULO: exportar y comprobar formato IEEE/MDPI con extension 4–6 paginas.',
        '6. CLUSTERING: silueta ~0.211 para muestra grafica de 1200; 0.2179 para otra muestra de 1800.',
        '7. INTERPRETACION: contrastar las variables categoricas con el diccionario del docente.',
        '', 'No incorporar IDs personales en graficos ni en anexos publicos.',
    ])
    REPORTE.write_text('\n'.join(lineas)+'\n', encoding='utf-8')
    with MANIFIESTO.open('w', newline='', encoding='utf-8-sig') as salida:
        writer = csv.DictWriter(salida, fieldnames=['seccion','archivo','estado','tamano_bytes','sha256'])
        writer.writeheader()
        writer.writerows(manifiesto)
    print('\n'.join(lineas))
    print(f'\nREPORTE: {REPORTE}')
    print(f'MANIFIESTO: {MANIFIESTO}')

if __name__ == '__main__':
    main()
