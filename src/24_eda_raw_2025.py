"""PASO 24. EDA de fuentes RAW ENAHO/ESCALE entregadas al examen.

Lee los 4 ZIP ENAHO de panel (observaciones con AÑO_25=2025) y
los 2 ZIP DBF MINEDU SIN cambiar los datos originales ni usar el CSV maestro.

Salida: outputs/eda_raw_2025 y outputs/figures/eda_raw_*.png.
Requiere: python -m pip install matplotlib numpy dbfread
Ejecutar: python src/24_eda_raw_2025.py
"""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import tempfile
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
INEI = BASE / 'data/raw/inei'
MINEDU = BASE / 'data/raw/minedu'
OUT = BASE / 'outputs/eda_raw_2025'
FIG = BASE / 'outputs/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SOURCES = {
    '100': ('1041-Modulo1474.zip', 'enaho01-2021-2025-100-panel.csv',
            ['P1121', 'P1144', 'UBIGEO', 'FACTOR07'], ['P1121', 'P1144']),
    '200': ('1041-Modulo1479.zip', 'enaho01-2021-2025-200-panel.csv',
            ['P207', 'P208A', 'UBIGEO'], ['P207']),
    '300': ('1041-Modulo1475.zip', 'enaho01a-2021-2025-300-panel.csv',
            ['P208A', 'P306', 'P307', 'P314A'], ['P306', 'P314A']),
    'SUMARIA': ('1041-Modulo1478.zip', 'sumaria-2021-2025-panel.csv',
                ['POBREZA', 'INGHOG2D', 'GASHOG2D', 'FACTOR07'], ['POBREZA']),
}

SUMMARY = []
MISSING = []
FREQ = []
NUMERIC = []
CHARTS = {}
LOG = []


def say(line=''):
    print(line)
    LOG.append(str(line))


def save_csv(file, rows, fields):
    with (OUT / file).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def num(s):
    s = str(s).strip().replace(',', '.')
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def describe(source, var, values):
    if not values:
        return
    a = np.asarray(values, dtype=float)
    NUMERIC.append({'fuente': source, 'variable': var, 'n_validos': len(a),
                    'minimo': round(float(a.min()), 3),
                    'mediana': round(float(np.median(a)), 3),
                    'media': round(float(a.mean()), 3),
                    'p95': round(float(np.percentile(a, 95)), 3),
                    'maximo': round(float(a.max()), 3)})


def draw_bar(data, title, ylabel, filename, xlabel='Código observado'):
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    keys = list(data.keys())
    values = [data[k] for k in keys]
    bars = ax.bar(keys, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=165)
    plt.close(fig)
    CHARTS[filename] = str(FIG / filename)


def process_enaho(mod, spec):
    zip_file, csv_name, fields, cat_fields = spec
    path = INEI / zip_file
    if not path.is_file():
        raise FileNotFoundError(f'No existe {path}. Revisa data/raw/inei.')
    total_panel, total_2025 = 0, 0
    available = Counter()
    categories = {x: Counter() for x in cat_fields}
    ages, ages_school, incomes, expenses = [], [], [], []
    with ZipFile(path) as z:
        match = [x for x in z.namelist() if Path(x).name.casefold() == csv_name.casefold()]
        if not match:
            raise FileNotFoundError(f'{csv_name} no está dentro de {path.name}')
        internal = match[0]
        with z.open(internal) as probe:
            sample = probe.read(65536)
        try:
            sample.decode('utf-8-sig')
            encoding = 'utf-8-sig'
        except UnicodeDecodeError:
            encoding = 'latin1'
        with z.open(internal) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding=encoding, newline=''))
            head = [x.strip().lstrip('\ufeff').upper() for x in next(reader)]
            index = {x: i for i, x in enumerate(head)}
            if 'AÑO_25' not in index:
                raise ValueError(f'Falta AÑO_25 en {internal}.')
            positions = {x: index.get(x + '_25') for x in fields}
            available_fields = [x for x in fields if positions[x] is not None]
            missing_fields = [x for x in fields if positions[x] is None]
            say(f'{mod}: columnas solicitadas ausentes en RAW: {missing_fields}')
            for rec in reader:
                total_panel += 1
                if index['AÑO_25'] >= len(rec) or rec[index['AÑO_25']].strip() != '2025':
                    continue
                total_2025 += 1
                row = {x: (rec[j].strip() if j is not None and j < len(rec) else '')
                       for x, j in positions.items()}
                for var in available_fields:
                    v = row[var]
                    if v:
                        available[var] += 1
                    if var in categories:
                        categories[var][v or 'SIN_DATO'] += 1
                if mod in ('200', '300'):
                    age = num(row.get('P208A', ''))
                    if age is not None:
                        ages.append(age)
                        if 6 <= age <= 17:
                            ages_school.append(age)
                if mod == 'SUMARIA':
                    for var, target in [('INGHOG2D', incomes), ('GASHOG2D', expenses)]:
                        v = num(row.get(var, ''))
                        if v is not None:
                            target.append(v)
    SUMMARY.append({'fuente': 'ENAHO ' + mod, 'registros_panel': total_panel,
                    'registros_corte_2025': total_2025, 'columnas_raw': len(head)})
    say(f'ENAHO {mod}: RAW panel={total_panel:,}; AÑO_25=2025={total_2025:,}; columnas={len(head)}')
    for var in fields:
        nonempty = available[var]
        MISSING.append({'fuente': 'ENAHO ' + mod, 'variable': var,
                        'n': total_2025, 'presentes': nonempty,
                        'faltantes': total_2025 - nonempty,
                        'porcentaje_faltante': round(100*(total_2025-nonempty)/total_2025, 2) if total_2025 else 0,
                        'existe_en_raw': int(positions[var] is not None)})
    for var, ctr in categories.items():
        for code, n in ctr.most_common(15):
            FREQ.append({'fuente': 'ENAHO ' + mod, 'variable': var,
                         'codigo_sin_interpretar': code, 'conteo': n,
                         'porcentaje_fuente': round(100*n/total_2025, 2) if total_2025 else 0})
    if mod == '200':
        describe('ENAHO 200', 'P208A (todas las edades con valor)', ages)
        # Filtrado solo para VISUALIZACION, no cambia los originales.
        draw_bar({str(i): ages_school.count(i) for i in range(6,18)},
                 'EDA RAW ENAHO 200: edades 6 a 17 (corte 2025)', 'Personas',
                 'eda_raw_edades_6_17_2025.png', 'Edad en años')
    if mod == '300':
        for var in ('P306', 'P314A'):
            draw_bar(dict(categories[var].most_common(8)),
                     f'EDA RAW ENAHO 300: frecuencias {var} (todas las edades, corte 2025)',
                     'Respuestas', f'eda_raw_{var.lower()}_2025.png')
    if mod == 'SUMARIA':
        describe('ENAHO SUMARIA', 'INGHOG2D', incomes)
        describe('ENAHO SUMARIA', 'GASHOG2D', expenses)
        # Representación robusta: la transformación SOLO ES VISUAL, no toca los valores raw.
        fig, ax = plt.subplots(figsize=(8,4.5))
        for vals, label in ((incomes, 'Ingreso del hogar'), (expenses, 'Gasto del hogar')):
            a = np.asarray([x for x in vals if x >= 0], dtype=float)
            if len(a):
                ax.hist(np.log1p(a), bins=55, alpha=0.45, label=label)
        ax.set_title('EDA RAW Sumaria: distribución de ingreso y gasto (2025)')
        ax.set_xlabel('log(1 + valor no negativo), solo para visualizar')
        ax.set_ylabel('Hogares')
        ax.legend()
        fig.tight_layout()
        f = 'eda_raw_ingreso_gasto_2025.png'
        fig.savefig(FIG / f, dpi=165)
        plt.close(fig)
        CHARTS[f] = str(FIG / f)


def process_dbf(label, zip_filename, dbf_name, fields):
    try:
        from dbfread import DBF
    except ImportError as e:
        raise ImportError('Instala dbfread: python -m pip install dbfread') from e
    path = MINEDU / zip_filename
    if not path.is_file():
        raise FileNotFoundError(f'No existe {path}')
    with ZipFile(path) as z:
        matches = [x for x in z.namelist() if Path(x).name.casefold() == dbf_name.casefold()]
        if not matches:
            raise FileNotFoundError(f'No se encontró {dbf_name} dentro de {zip_filename}')
        with tempfile.TemporaryDirectory(prefix='eda_raw_') as d:
            dst = Path(d) / dbf_name
            with z.open(matches[0]) as s, dst.open('wb') as w:
                shutil.copyfileobj(s, w)
            table = DBF(str(dst), encoding='latin1', char_decode_errors='replace',
                        ignore_missing_memofile=True, load=False)
            head = list(table.field_names)
            cols = [f for f in fields if f in head]
            counts = Counter()
            freq = {f:Counter() for f in cols if f in ('D_GESTION', 'DAREACENSO', 'D_ESTADO')}
            n = 0
            for row in table:
                n += 1
                for col in cols:
                    v = row.get(col)
                    if v is not None and str(v).strip():
                        counts[col] += 1
                    if col in freq:
                        freq[col][str(v).strip() if v is not None and str(v).strip() else 'SIN_DATO'] += 1
    SUMMARY.append({'fuente': label, 'registros_panel': '', 'registros_corte_2025': n,
                    'columnas_raw': len(head)})
    say(f'{label}: RAW={n:,} filas; columnas={len(head)}')
    for col in fields:
        MISSING.append({'fuente': label, 'variable': col, 'n': n,
                        'presentes': counts[col], 'faltantes': n-counts[col],
                        'porcentaje_faltante': round(100*(n-counts[col])/n,2) if n else 0,
                        'existe_en_raw': int(col in head)})
    for col, ctr in freq.items():
        for code, count in ctr.most_common(15):
            FREQ.append({'fuente': label, 'variable': col, 'codigo_sin_interpretar': code,
                         'conteo': count, 'porcentaje_fuente': round(100*count/n,2) if n else 0})
    if label == 'MINEDU Padron':
        # Texto original sin recodificar gestión. Frecuencias sin IDs escolares.
        ctr = freq.get('D_GESTION', Counter())
        draw_bar(dict(ctr.most_common(7)),
                 'EDA RAW MINEDU Padrón: gestión educativa', 'Servicios educativos',
                 'eda_raw_minedu_gestion_2025.png', 'Categoría original')


def main():
    say('EDA SOBRE ARCHIVOS CRUDOS RAW — ENAHO / ESCALE 2025')
    say('SOLO LECTURA. NO se usa el dataset procesado, no se imputa, no se entrena.')
    say('Panel entregado 2021–2025: se estudia AÑO_25=2025 sin modificar los ZIP originales.')
    for name, spec in SOURCES.items():
        process_enaho(name, spec)
    process_dbf('MINEDU Padron', '00_Padron.zip', 'Padron.dbf',
                ['CODGEO', 'COD_MOD', 'D_GESTION', 'DAREACENSO', 'D_ESTADO',
                 'TALUM', 'TDOC', 'NLAT_IE', 'NLONG_IE'])
    process_dbf('MINEDU Padlocal', '00_Padlocal.zip', 'Padlocal.dbf',
                ['CODLOCAL', 'CODGEO', 'D_ESTADO'])
    save_csv('eda_resumen_fuentes_2025.csv', SUMMARY,
             ['fuente','registros_panel','registros_corte_2025','columnas_raw'])
    save_csv('eda_faltantes_raw_2025.csv', MISSING,
             ['fuente','variable','n','presentes','faltantes','porcentaje_faltante','existe_en_raw'])
    save_csv('eda_frecuencias_raw_2025.csv', FREQ,
             ['fuente','variable','codigo_sin_interpretar','conteo','porcentaje_fuente'])
    save_csv('eda_estadisticas_numericas_raw_2025.csv', NUMERIC,
             ['fuente','variable','n_validos','minimo','mediana','media','p95','maximo'])
    say('')
    say('Archivos EDA creados:')
    for f in sorted(OUT.glob('*.csv')):
        say('  ' + str(f.relative_to(BASE)))
    for f in CHARTS:
        say('  outputs/figures/' + f)
    say('Advertencia: los valores de P307 y otros códigos no deben interpretarse sin cuestionario.')
    say('Un valor faltante por salto de encuesta no necesariamente es un error.')
    (OUT/'reporte_eda_raw_2025.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
    say('Informe: outputs/eda_raw_2025/reporte_eda_raw_2025.txt')


if __name__ == '__main__':
    main()
