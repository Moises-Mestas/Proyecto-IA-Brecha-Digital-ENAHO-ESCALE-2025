"""Auditoría NO destructiva: verifica MIN_PROP_PUBLICOS del CSV maestro.

Ejecutar desde raíz del proyecto: python src/16_comprobar_consistencia_minedu_2025.py
Requiere: pandas, dbfread.
No cambia datasets ni modelos. Solo crea dos archivos en outputs/.
"""
from pathlib import Path
from zipfile import ZipFile
import shutil
import tempfile
import re
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ZIP_PADRON = BASE / "data/raw/minedu/00_Padron.zip"
CSV_MAESTRO = BASE / "data/processed/dataset_enaho_escale_2025.csv"
REPORTE = BASE / "outputs/reporte_conciliacion_minedu_2025.txt"
TABLA = BASE / "outputs/conciliacion_minedu_distritos_2025.csv"


def texto(v):
    if v is None:
        return ""
    s = unicodedata.normalize("NFKD", str(v))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).upper().strip()


def ubigeo(v):
    v = "" if v is None else str(v).strip()
    if re.fullmatch(r"\d{1,6}", v):
        return v.zfill(6)
    return v


def clasificacion(gestion):
    gestion = texto(gestion)
    if gestion.startswith("PUBLICA"):
        return 1.0
    if gestion == "PRIVADA":
        return 0.0
    return float('nan')


def main():
    try:
        from dbfread import DBF
    except ImportError as e:
        raise SystemExit("Instala dbfread: python -m pip install dbfread") from e
    if not ZIP_PADRON.is_file() or not CSV_MAESTRO.is_file():
        raise SystemExit("No encuentro el Padron ZIP o el dataset maestro. Ejecuta desde la carpeta del proyecto.")
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(ZIP_PADRON) as z:
        encontrados = [n for n in z.namelist() if Path(n).name.lower() == 'padron.dbf']
        if not encontrados:
            raise SystemExit("No encuentro Padron.dbf en 00_Padron.zip")
        with tempfile.TemporaryDirectory(prefix='revision_minedu_') as t:
            ruta = Path(t) / 'Padron.dbf'
            with z.open(encontrados[0]) as entrada, ruta.open('wb') as salida:
                shutil.copyfileobj(entrada, salida)
            tabla = DBF(str(ruta), encoding='latin1', ignore_missing_memofile=True, load=False)
            filas = []
            for reg in tabla:
                if texto(reg.get('D_ESTADO', '')).startswith('INACTIV'):
                    continue
                distrito = ubigeo(reg.get('CODGEO'))
                if not re.fullmatch(r'\d{6}', distrito):
                    continue
                filas.append((distrito, texto(reg.get('D_GESTION')), clasificacion(reg.get('D_GESTION'))))
    padron = pd.DataFrame(filas, columns=['UBIGEO', 'GESTION', 'ES_PUBLICO'])
    cuenta_publicas = int((padron.ES_PUBLICO == 1).sum())
    cuenta_privadas = int((padron.ES_PUBLICO == 0).sum())
    mal_clasificadas = int(padron.GESTION.str.startswith('PUBLICA DE GESTION PRIVADA').sum())
    esperado = padron.groupby('UBIGEO', as_index=False)['ES_PUBLICO'].mean().rename(
        columns={'ES_PUBLICO': 'PROP_PUBLICOS_ESPERADA'})

    df = pd.read_csv(CSV_MAESTRO, dtype=str, encoding='utf-8-sig',
                     keep_default_na=False, low_memory=False, usecols=['UBIGEO', 'MIN_PROP_PUBLICOS'])
    df['UBIGEO'] = df['UBIGEO'].map(ubigeo)
    df['MIN_PROP_PUBLICOS'] = pd.to_numeric(
        df['MIN_PROP_PUBLICOS'].str.strip().str.replace(',', '.', regex=False), errors='coerce')
    n_inconsistentes = int((df.groupby('UBIGEO')['MIN_PROP_PUBLICOS'].nunique(dropna=True) > 1).sum())
    observadas = df.groupby('UBIGEO', as_index=False)['MIN_PROP_PUBLICOS'].mean()
    unidas = observadas.merge(esperado, on='UBIGEO', how='left', validate='one_to_one')
    unidas['DIFERENCIA_ABS'] = (unidas['MIN_PROP_PUBLICOS'] - unidas['PROP_PUBLICOS_ESPERADA']).abs()
    unidas['DESACUERDO'] = unidas['PROP_PUBLICOS_ESPERADA'].isna() | unidas['MIN_PROP_PUBLICOS'].isna() | (unidas['DIFERENCIA_ABS'] > 1e-8)
    discrepancias = int(unidas['DESACUERDO'].sum())
    estudiantes_afectados = int(df['UBIGEO'].isin(unidas.loc[unidas.DESACUERDO, 'UBIGEO']).sum())
    unidas.sort_values(['DESACUERDO', 'DIFERENCIA_ABS'], ascending=False).to_csv(TABLA, index=False, encoding='utf-8-sig')

    lineas = [
        'AUDITORIA INDEPENDIENTE DE MIN_PROP_PUBLICOS — ENAHO/ESCALE 2025',
        'Solo se leen el ZIP Padron y el CSV maestro. No modifica ningun dato.',
        f'Filas PADRON (con UBIGEO valido, sin inactivos explicitos): {len(padron):,}',
        f'Gestion correctamente clasificada: PUBLICA={cuenta_publicas:,}; PRIVADA={cuenta_privadas:,}',
        f'Servicios "Publica de gestion privada" incluidos dentro de PUBLICA: {mal_clasificadas:,}',
        f'Filas del CSV maestro: {len(df):,}; distritos distintos: {observadas.shape[0]:,}',
        f'Distritos con multiples valores MIN_PROP_PUBLICOS dentro del maestro: {n_inconsistentes:,}',
        f'Distritos con proporciones distintas de las esperadas: {discrepancias:,}',
        f'Estudiantes en distritos con discrepancia: {estudiantes_afectados:,}',
        'VEREDICTO: ' + ('NO COINCIDE; revisar/reconstruir el dataset maestro y los resultados.'
                       if discrepancias or n_inconsistentes else
                       'COINCIDE; es posible que el reporte de integracion previo estuviera desactualizado.'),
        f'Detalle distrital: {TABLA.relative_to(BASE)}',
    ]
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print('\n'.join(lineas))


if __name__ == '__main__':
    main()
