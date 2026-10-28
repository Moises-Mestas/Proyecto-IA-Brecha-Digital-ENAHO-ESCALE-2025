"""Examen IA ENAHO-ESCALE 2025 | Fase 1, paso 07.

Construye indicadores educativos distritales a partir de Padron.dbf y Padlocal.dbf
(archivos recibidos del docente) y los integra al ENAHO ya preparado.

Reglas: no modificar datos crudos, no inventar valores, no imputar ni escalar;
validar el cruce muchos-a-uno por UBIGEO y conservar los 24 195 alumnos.

Dependencias: pip install pandas dbfread
Ejecución: python src/07_integrar_minedu_escale_2025.py
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
RAW_MINEDU = BASE / 'data' / 'raw' / 'minedu'
RUTA_ENAHO = BASE / 'data' / 'interim' / 'enaho_estudiantes_2025_integrado.csv'
DIR_INTERIM = BASE / 'data' / 'interim'
DIR_PROCESSED = BASE / 'data' / 'processed'
DIR_OUTPUTS = BASE / 'outputs'

SALIDA_DISTRITO = DIR_INTERIM / 'minedu_indicadores_distrito_2025.csv'
SALIDA_AREA = DIR_INTERIM / 'minedu_indicadores_distrito_area_2025.csv'
SALIDA_FINAL = DIR_PROCESSED / 'dataset_enaho_escale_2025.csv'
SALIDA_REPORTE = DIR_OUTPUTS / 'reporte_integracion_minedu_2025.txt'

LOG = []


def registrar(mensaje=''):
    mensaje = str(mensaje)
    print(mensaje)
    LOG.append(mensaje)


def normalizar_texto(valor):
    if pd.isna(valor):
        return ''
    txt = unicodedata.normalize('NFKD', str(valor))
    return ''.join(c for c in txt if not unicodedata.combining(c)).upper().strip()


def limpiar_ubigeo(valor):
    if pd.isna(valor):
        return ''
    texto = str(valor).strip()
    # Solo estandarizar códigos realmente numéricos: no inventar otros códigos.
    if re.fullmatch(r'\d{1,6}', texto):
        return texto.zfill(6)
    return texto


def numero(serie):
    # Los decimales de DBF suelen convertirse automáticamente a int/float/Decimal.
    normal = serie.astype('string').str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(normal, errors='coerce')


def leer_dbf_desde_zip(nombre_zip, nombre_dbf, campos):
    """Extraer DBF temporalmente; conservar originales y leer solo campos útiles."""
    try:
        from dbfread import DBF
    except ImportError as exc:
        raise RuntimeError(
            'Falta instalar dbfread. En VS Code ejecuta: python -m pip install pandas dbfread'
        ) from exc

    ruta_zip = RAW_MINEDU / nombre_zip
    if not ruta_zip.is_file():
        raise FileNotFoundError(f'No se encontró {ruta_zip}')

    with ZipFile(ruta_zip, 'r') as comprimido:
        candidatos = [
            x for x in comprimido.namelist()
            if Path(x).name.casefold() == nombre_dbf.casefold()
        ]
        if not candidatos:
            raise FileNotFoundError(f'{nombre_dbf} no existe dentro de {nombre_zip}')
        with tempfile.TemporaryDirectory(prefix='examen_escale_') as temporal:
            destino = Path(temporal) / nombre_dbf
            with comprimido.open(candidatos[0]) as origen, destino.open('wb') as dst:
                shutil.copyfileobj(origen, dst)
            tabla = DBF(
                str(destino),
                encoding='latin1',
                char_decode_errors='replace',
                ignore_missing_memofile=True,
                load=False,
            )
            disponibles = set(tabla.field_names)
            faltantes = [campo for campo in campos if campo not in disponibles]
            if faltantes:
                raise ValueError(f'{nombre_dbf}: columnas necesarias ausentes: {faltantes}')
            registros = (
                {campo: registro.get(campo) for campo in campos}
                for registro in tabla
            )
            df = pd.DataFrame.from_records(registros, columns=campos)
    registrar(f'{nombre_dbf}: {len(df):,} filas, {len(df.columns)} columnas seleccionadas')
    return df


def construir_indicadores(padron, padlocal):
    """Tablas agregadas por distrito y por distrito+área escolar."""
    p = padron.copy()
    l = padlocal.copy()

    # Solo excluir estados que dicen explícitamente INACTIVO/A.
    # Si el estado viene codificado o es ambiguo, no adivinar su significado.
    for etiqueta, frame in (('Padrón', p), ('Padlocal', l)):
        estados = frame['D_ESTADO'].map(normalizar_texto)
        inactivos = estados.str.startswith('INACTIV')
        registrar(f'{etiqueta}: {int(inactivos.sum()):,} registros explícitamente inactivos excluidos')
        frame.drop(index=frame.index[inactivos], inplace=True)

    for frame in (p, l):
        frame['CODGEO'] = frame['CODGEO'].map(limpiar_ubigeo)
        frame.drop(frame.index[~frame['CODGEO'].str.fullmatch(r'\d{6}')], inplace=True)

    for frame in (p, l):
        frame['CODLOCAL'] = frame['CODLOCAL'].fillna('').astype(str).str.strip()
    p['COD_MOD'] = p['COD_MOD'].fillna('').astype(str).str.strip()

    p['TALUM'] = numero(p['TALUM'])
    p['TDOC'] = numero(p['TDOC'])
    # Valores negativos no representan conteos legítimos.
    p.loc[p['TALUM'] < 0, 'TALUM'] = float('nan')
    p.loc[p['TDOC'] < 0, 'TDOC'] = float('nan')

    # Clasificación a partir del TEXTO explícito, sin suponer el código GESTION.
    gestion = p['D_GESTION'].map(normalizar_texto)
    p['GESTION_CLASE'] = 'DESCONOCIDA'
    # IMPORTANTE: "Publica de gestion privada" sigue siendo PUBLICA.
    # No clasificar por contener "PRIVADA" porque esa frase aparece en
    # servicios publicos administrados por entidades privadas.
    p.loc[gestion.str.startswith('PUBLICA'), 'GESTION_CLASE'] = 'PUBLICA'
    p.loc[gestion.eq('PRIVADA'), 'GESTION_CLASE'] = 'PRIVADA'
    p['ES_PUBLICO'] = p['GESTION_CLASE'].map({'PUBLICA': 1.0, 'PRIVADA': 0.0})
    # Control automatico que detecta la confusión del programa anterior.
    publica_gestion_privada = gestion.str.startswith('PUBLICA DE GESTION PRIVADA')
    if not p.loc[publica_gestion_privada, 'GESTION_CLASE'].eq('PUBLICA').all():
        raise AssertionError('Error: publica de gestion privada debe clasificarse como publica.')

    # Identificar área usando la etiqueta textual DAREACENSO, no códigos supuestos.
    area = p['DAREACENSO'].map(normalizar_texto)
    p['AREA_ESCOLAR'] = 'NO_IDENTIFICADA'
    p.loc[area.str.contains('URBAN', regex=False), 'AREA_ESCOLAR'] = 'URBANA'
    p.loc[area.str.contains('RURAL', regex=False), 'AREA_ESCOLAR'] = 'RURAL'

    # Coords. Padrón contiene NLAT_IE y NLONG_IE (Padlocal no las contiene).
    p['LAT'] = numero(p['NLAT_IE'])
    p['LON'] = numero(p['NLONG_IE'])
    coordenadas_validas = (
        p['LAT'].between(-19, 1, inclusive='both')
        & p['LON'].between(-82, -68, inclusive='both')
    )

    # Distribuciones de texto, útiles para ver si la clasificación fue suficiente.
    registrar('Distribución D_GESTION (hasta 8 valores):')
    for clave, cantidad in p['D_GESTION'].fillna('(vacío)').value_counts().head(8).items():
        registrar(f'  {clave!s}: {cantidad:,}')
    registrar('Clasificación de gestión: ' + str(p['GESTION_CLASE'].value_counts().to_dict()))
    registrar('Nota: la proporción pública/privada se calcula sobre servicios educativos del Padrón (registros), no sobre locales únicos.')
    registrar('Clasificación de área: ' + str(p['AREA_ESCOLAR'].value_counts().to_dict()))
    registrar(f'Coordenadas escolares dentro de rangos geográficos: {int(coordenadas_validas.sum()):,}/{len(p):,}')
    registrar('Distribución D_ESTADO (hasta 8 valores):')
    for clave, cantidad in p['D_ESTADO'].fillna('(vacío)').value_counts().head(8).items():
        registrar(f'  {clave!s}: {cantidad:,}')
    registrar('NOTA: se excluyen estados explícitamente INACTIVO; otros códigos de estado requieren verificar el diccionario.')

    # Sumar alumnos y docentes; contar solo escuelas con gestión interpretable.
    d = p.groupby('CODGEO', as_index=False).agg(
        MIN_REGISTROS_PADRON=('COD_MOD', 'size'),
        MIN_SERVICIOS_EDUCATIVOS=('COD_MOD', 'nunique'),
        MIN_ALUMNOS=('TALUM', 'sum'),
        MIN_ALUMNOS_VALIDOS=('TALUM', 'count'),
        MIN_DOCENTES=('TDOC', 'sum'),
        MIN_DOCENTES_VALIDOS=('TDOC', 'count'),
        MIN_COLEGIOS_PUBLICOS=('ES_PUBLICO', 'sum'),
        MIN_GESTION_CONOCIDA=('ES_PUBLICO', 'count'),
    )
    d['MIN_COLEGIOS_PRIVADOS'] = d['MIN_GESTION_CONOCIDA'] - d['MIN_COLEGIOS_PUBLICOS']
    d['MIN_RATIO_ALUMNOS_DOCENTE'] = (
        d['MIN_ALUMNOS'] / d['MIN_DOCENTES'].replace(0, float('nan'))
    ).where((d['MIN_DOCENTES_VALIDOS'] > 0) & (d['MIN_ALUMNOS_VALIDOS'] > 0))
    d['MIN_PROP_PUBLICOS'] = d['MIN_COLEGIOS_PUBLICOS'] / d['MIN_GESTION_CONOCIDA'].replace(0, float('nan'))
    d['MIN_PROP_PRIVADOS'] = d['MIN_COLEGIOS_PRIVADOS'] / d['MIN_GESTION_CONOCIDA'].replace(0, float('nan'))

    # Estadísticas por distrito y área escolar: el examen exige ambas dimensiones.
    por_area = p.groupby(['CODGEO', 'AREA_ESCOLAR'], as_index=False).agg(
        SERVICIOS=('COD_MOD', 'nunique'),
        ALUMNOS=('TALUM', 'sum'),
        DOCENTES=('TDOC', 'sum'),
    )
    por_area['RATIO_ALUMNOS_DOCENTE'] = por_area['ALUMNOS'] / por_area['DOCENTES'].replace(0, float('nan'))
    por_area.rename(columns={'CODGEO': 'UBIGEO'}, inplace=True)

    pivot = por_area.pivot(index='UBIGEO', columns='AREA_ESCOLAR', values='SERVICIOS').fillna(0)
    for categoria in ('URBANA', 'RURAL', 'NO_IDENTIFICADA'):
        if categoria not in pivot.columns:
            pivot[categoria] = 0
    pivot = pivot[['URBANA', 'RURAL', 'NO_IDENTIFICADA']].reset_index()
    pivot.rename(columns={
        'URBANA': 'MIN_SERVICIOS_URBANOS',
        'RURAL': 'MIN_SERVICIOS_RURALES',
        'NO_IDENTIFICADA': 'MIN_SERVICIOS_AREA_DESCONOCIDA',
    }, inplace=True)
    d = d.merge(pivot, left_on='CODGEO', right_on='UBIGEO', how='left', validate='one_to_one')
    d.drop(columns='CODGEO', inplace=True)

    # Promedio de coordenadas por local escolar único para no dar más peso a
    # locales con varios servicios educativos (distintos COD_MOD).
    geo = p.loc[coordenadas_validas & p['CODLOCAL'].ne(''),
                ['CODGEO', 'CODLOCAL', 'LAT', 'LON']].drop_duplicates(['CODGEO', 'CODLOCAL'])
    por_geo = geo.groupby('CODGEO', as_index=False).agg(
        MIN_LAT_MEDIA=('LAT', 'mean'),
        MIN_LON_MEDIA=('LON', 'mean'),
        MIN_LOCALES_CON_COORDENADAS=('CODLOCAL', 'nunique'),
    ).rename(columns={'CODGEO': 'UBIGEO'})
    d = d.merge(por_geo, on='UBIGEO', how='left', validate='one_to_one')

    # Padlocal agrega un indicador de locales; no se usa para inventar coordenadas.
    locales = l.loc[l['CODLOCAL'].ne('')].groupby('CODGEO', as_index=False).agg(
        MIN_LOCALES_PADLOCAL=('CODLOCAL', 'nunique'),
    ).rename(columns={'CODGEO': 'UBIGEO'})
    d = d.merge(locales, on='UBIGEO', how='left', validate='one_to_one')
    registrar(f'Distritos en Padrón: {d["UBIGEO"].nunique():,}')
    registrar(f'Distritos en Padlocal: {l["CODGEO"].nunique():,}')

    if d['UBIGEO'].duplicated().any():
        raise ValueError('La tabla distrital contiene UBIGEO duplicados. Se detiene para no multiplicar estudiantes.')

    return d, por_area


def main():
    registrar('FASE 1 - INTEGRACIÓN ESCALE 2025 CON ENAHO')
    registrar('Origen: archivos MINEDU entregados y ENAHO integrado del programa 06.')
    for carpeta in (DIR_INTERIM, DIR_PROCESSED, DIR_OUTPUTS):
        carpeta.mkdir(parents=True, exist_ok=True)

    padron = leer_dbf_desde_zip(
        '00_Padron.zip', 'Padron.dbf',
        ['COD_MOD', 'CODLOCAL', 'CODGEO', 'D_GESTION', 'DAREACENSO',
         'D_ESTADO', 'TALUM', 'TDOC', 'NLAT_IE', 'NLONG_IE'],
    )
    padlocal = leer_dbf_desde_zip(
        '00_Padlocal.zip', 'Padlocal.dbf',
        ['CODLOCAL', 'CODGEO', 'D_ESTADO'],
    )

    distritos, distritos_area = construir_indicadores(padron, padlocal)
    distritos.to_csv(SALIDA_DISTRITO, index=False, encoding='utf-8-sig')
    distritos_area.to_csv(SALIDA_AREA, index=False, encoding='utf-8-sig')

    if not RUTA_ENAHO.exists():
        raise FileNotFoundError(
            f'Falta {RUTA_ENAHO}. Primero ejecuta el programa 06.'
        )
    enaho = pd.read_csv(RUTA_ENAHO, dtype=str, encoding='utf-8-sig',
                        keep_default_na=False, low_memory=False)
    enaho.columns = enaho.columns.str.strip().str.upper()
    if 'UBIGEO' not in enaho.columns:
        raise ValueError('La tabla ENAHO no contiene UBIGEO.')
    enaho['UBIGEO'] = enaho['UBIGEO'].map(limpiar_ubigeo)

    registrar('\n--- Unión ENAHO + ESCALE por distrito ---')
    registrar(f'Filas originales ENAHO: {len(enaho):,}')
    integracion = enaho.merge(
        distritos, on='UBIGEO', how='left', validate='many_to_one',
        indicator=True,
    )
    encontrados = int(integracion['_merge'].eq('both').sum())
    no_encontrados = len(integracion) - encontrados
    registrar(f'Coincidencias con distrito MINEDU: {encontrados:,}/{len(enaho):,} ({encontrados/max(len(enaho),1):.1%})')
    registrar(f'Sin coincidencia distrital: {no_encontrados:,}')
    registrar(f'Distritos distintos sin coincidencia: {integracion.loc[integracion["_merge"].eq("left_only"), "UBIGEO"].nunique():,}')
    integracion.drop(columns='_merge', inplace=True)

    if len(integracion) != len(enaho):
        raise RuntimeError('Error: el merge alteró la cantidad de estudiantes.')
    llave_persona = ['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO']
    if all(x in integracion.columns for x in llave_persona):
        duplicados = int(integracion.duplicated(llave_persona).sum())
        registrar(f'Duplicados por llave de persona: {duplicados:,}')
        if duplicados:
            raise RuntimeError('Error: se detectaron personas duplicadas.')

    for columna in ['MIN_RATIO_ALUMNOS_DOCENTE', 'MIN_PROP_PUBLICOS',
                    'MIN_LAT_MEDIA', 'MIN_LON_MEDIA', 'MIN_LOCALES_PADLOCAL']:
        if columna in integracion.columns:
            registrar(f'{columna}: {int(integracion[columna].notna().sum()):,}/{len(integracion):,} estudiantes con indicador')

    integracion.to_csv(SALIDA_FINAL, index=False, encoding='utf-8-sig')
    registrar('\n--- Archivos generados ---')
    registrar(f'Indicadores distritales: {SALIDA_DISTRITO}')
    registrar(f'Indicadores distrito+área: {SALIDA_AREA}')
    registrar(f'Dataset maestro: {SALIDA_FINAL}')
    registrar(f'Tamaño final: {len(integracion):,} filas y {len(integracion.columns):,} columnas')
    registrar('NOTA: sin imputación, escalado, clasificación ni entrenamiento. Ratios = suma alumnos / suma docentes.')
    registrar('NOTA: es información del Padrón de instituciones, NO demuestra que un estudiante asiste a una escuela concreta.')
    registrar('NOTA: validar diccionarios MINEDU para gestión, área y estado antes de interpretar resultados definitivos.')
    SALIDA_REPORTE.write_text('\n'.join(LOG) + '\n', encoding='utf-8')
    registrar(f'Reporte: {SALIDA_REPORTE}')


if __name__ == '__main__':
    main()
