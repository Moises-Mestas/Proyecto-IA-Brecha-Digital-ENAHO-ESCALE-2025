"""Paso 17: auditoria NO DESTRUCTIVA del objetivo, matricula y factor muestral.

Ejecutar desde la raiz del proyecto:
    python src/17_auditar_objetivo_y_pesos_2025.py

Entrada UNICA: data/processed/dataset_enaho_escale_2025.csv
Salidas: outputs/reporte_auditoria_objetivo_2025.txt y tablas agregadas.
No modifica datasets, modelos, figuras, articulos ni ejecuta entrenamiento.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / 'data' / 'processed' / 'dataset_enaho_escale_2025.csv'
SALIDAS = BASE / 'outputs'
CARPETA = SALIDAS / 'auditoria_clasificacion'
REPORTE = SALIDAS / 'reporte_auditoria_objetivo_2025.txt'

OBJETIVOS = ('EDU_P306', 'EDU_P307', 'EDU_P314A')
PESOS_CANDIDATOS = ('SUM_FACTOR07', 'FACTOR07', 'FACPOB07', 'SUM_FACPOB07')


def formatear_conteo(valor):
    return f'{int(valor):,}'


def porcentaje(numero, total):
    if total == 0:
        return 'no calculable'
    return f'{100 * numero / total:.2f}%'


def decodificar_respuesta(serie):
    return serie.fillna('').astype(str).str.strip().replace('', 'SIN_DATO')


def convertir_numero(serie):
    """Convierte numericos del INEI con espacios y coma decimal si aparece."""
    texto = serie.fillna('').astype(str).str.strip().str.replace(' ', '', regex=False)
    texto = texto.str.replace(',', '.', regex=False)
    return pd.to_numeric(texto.mask(texto.eq('')), errors='coerce')


def main():
    if not DATOS.is_file():
        raise FileNotFoundError(f'No encuentro el dataset maestro: {DATOS}')
    SALIDAS.mkdir(parents=True, exist_ok=True)
    CARPETA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATOS, dtype=str, encoding='utf-8-sig',
                     keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    campos_minimos = ('AÑO', 'CONGLOME', 'EDU_P306', 'EDU_P314A')
    ausentes = [x for x in campos_minimos if x not in df.columns]
    if ausentes:
        raise ValueError(f'Faltan columnas esenciales: {ausentes}')
    for campo in set(campos_minimos) | set(OBJETIVOS).intersection(df.columns):
        df[campo] = decodificar_respuesta(df[campo])
    if not df['AÑO'].eq('2025').all():
        raise ValueError('Hay años distintos de 2025. Detener para revisar.')
    if df['CONGLOME'].eq('SIN_DATO').any():
        raise ValueError('Hay conglomerados faltantes. Detener para revisar.')

    salida = []
    def nota(texto=''):
        salida.append(str(texto))
        print(texto, flush=True)

    nota('EXAMEN IA ENAHO–ESCALE 2025 | PASO 17 | AUDITORIA OBJETIVO Y PESOS')
    nota('SOLO LECTURA. No entrena modelos ni cambia archivos de pasos anteriores.')
    nota(f'Dataset maestro: {formatear_conteo(len(df))} filas | {len(df.columns)} columnas')
    nota('P314A: 1=SI uso internet el mes anterior; 2=NO uso internet el mes anterior.')
    nota('P306: 1=SI matriculado; 2=NO matriculado en el año.')
    nota('P307: NO asignar significado definitivo hasta revisar cuestionario.')
    nota('')

    for columna in OBJETIVOS:
        if columna not in df:
            nota(f'{columna}: COLUMNA NO DISPONIBLE')
            continue
        nota(f'--- Frecuencias de {columna} ---')
        for codigo, cantidad in df[columna].value_counts(dropna=False).items():
            nota(f'  codigo {codigo}: {formatear_conteo(cantidad)} ({porcentaje(cantidad, len(df))})')
        nota('')

    tabla = pd.crosstab(df['EDU_P306'], df['EDU_P314A'], dropna=False)
    tabla.index.name = 'CODIGO_P306'
    tabla.columns.name = 'CODIGO_P314A'
    ruta_tabla = CARPETA / 'cruce_matricula_internet_2025.csv'
    tabla.to_csv(ruta_tabla, encoding='utf-8-sig')
    nota('--- Cruce matricula 2025 y uso de internet en mes anterior (conteos) ---')
    nota(tabla.to_string())
    nota('')

    validos_internet = df['EDU_P314A'].isin(['1', '2'])
    no_uso = df['EDU_P314A'].eq('2')
    matriculados = df['EDU_P306'].eq('1')
    no_matriculados = df['EDU_P306'].eq('2')
    matriculados_validos = matriculados & validos_internet
    no_matriculados_validos = no_matriculados & validos_internet
    nota('--- Objetivos que SI se pueden construir con las preguntas confirmadas ---')
    nota(f'A) No uso reciente, todos los estudiantes con respuesta valida: {formatear_conteo(validos_internet.sum())}; '
         f'positivos={formatear_conteo((validos_internet & no_uso).sum())} '
         f'({porcentaje((validos_internet & no_uso).sum(), validos_internet.sum())}).')
    nota(f'B) No uso reciente ENTRE MATRICULADOS (P306=1): {formatear_conteo(matriculados_validos.sum())}; '
         f'positivos={formatear_conteo((matriculados_validos & no_uso).sum())} '
         f'({porcentaje((matriculados_validos & no_uso).sum(), matriculados_validos.sum())}).')
    nota(f'Comparacion descriptiva entre NO matriculados (P306=2): '
         f'con respuesta valida={formatear_conteo(no_matriculados_validos.sum())}; '
         f'no uso={formatear_conteo((no_matriculados_validos & no_uso).sum())} '
         f'({porcentaje((no_matriculados_validos & no_uso).sum(), no_matriculados_validos.sum())}).')
    nota('IMPORTANTE: B es indicador de NO USO entre estudiantes MATRICULADOS, '
         'NO equivale a falta de competencias digitales ni a asistencia escolar confirmada.')
    nota('No se puede afirmar EXCLUSION DIGITAL EFECTIVA completa sin operacionalizacion/diccionario adicional.')
    nota('')

    if 'EDU_P307' in df:
        cruce_307 = pd.crosstab([df['EDU_P306'], df['EDU_P307']], df['EDU_P314A'],
                               dropna=False)
        # Evita una combinación cartesiana grande o filas sin observaciones.
        cruce_307 = cruce_307.loc[cruce_307.sum(axis=1).gt(0)]
        ruta_307 = CARPETA / 'cruce_p306_p307_internet_2025.csv'
        cruce_307.to_csv(ruta_307, encoding='utf-8-sig')
        nota(f'Cruce exploratorio codigos P306/P307/P314A: {ruta_307.relative_to(BASE)}')
        nota('P307 se conserva como CODIGO; verificar texto exacto en el cuestionario.')
        nota('')

    nota('--- Evaluacion del desbalance de clases ---')
    for nombre, mask in [('A - Todos con respuesta valida', validos_internet),
                         ('B - Matriculados con respuesta valida', matriculados_validos)]:
        sub = df.loc[mask, 'EDU_P314A']
        n_si = int(sub.eq('1').sum())
        n_no = int(sub.eq('2').sum())
        if n_si and n_no:
            w_si = len(sub) / (2 * n_si)
            w_no = len(sub) / (2 * n_no)
            nota(f'{nombre}: n={formatear_conteo(len(sub))}; clase SI={formatear_conteo(n_si)}; '
                 f'clase NO={formatear_conteo(n_no)}; relacion mayor/menor={max(n_si,n_no)/min(n_si,n_no):.3f}')
            nota(f'  Pesos balanceados TEORICOS para esa poblacion: '
                 f'SI={w_si:.4f}, NO={w_no:.4f}.')
        else:
            nota(f'{nombre}: datos insuficientes para ambas clases.')
    nota('Los pesos anteriores son DESCRIPTIVOS; si se entrena de nuevo, '
         'hay que calcularlos dentro de cada fold usando solo su entrenamiento.')
    nota('')

    nota('--- Factores de expansion disponibles en el maestro ---')
    presentes = []
    for campo in PESOS_CANDIDATOS:
        if campo not in df.columns:
            nota(f'{campo}: NO ESTA EN EL CSV MAESTRO')
            continue
        valores = convertir_numero(df[campo])
        finitos = valores.notna() & np.isfinite(valores)
        validos = finitos & valores.gt(0)
        presentes.append((campo, int(validos.sum())))
        nota(f'{campo}: numéricos finitos={formatear_conteo(finitos.sum())}; '
             f'positivos={formatear_conteo(validos.sum())}/{formatear_conteo(len(df))}; '
             f'no positivos/invalidos={formatear_conteo((~validos).sum())}')
        if validos.any():
            nota(f'  min/mediana/max positivos: {valores[validos].min():.5g} / '
                 f'{valores[validos].median():.5g} / {valores[validos].max():.5g}')
            for nombre, mask in [('A', validos_internet), ('B', matriculados_validos)]:
                cubiertos = int((validos & mask).sum())
                nota(f'  objetivo {nombre}: pesos validos={formatear_conteo(cubiertos)}/'
                     f'{formatear_conteo(mask.sum())}')
        nota('')
    nota('La existencia de un factor NO prueba su idoneidad para esta subpoblacion, '
         'ni permite declarar estimaciones nacionales sin estratos, conglomerados y diseño muestral.')
    if any(c == 'SUM_FACTOR07' and n > 0 for c, n in presentes):
        nota('SUM_FACTOR07 tiene valores positivos: verificar el diccionario ENAHO antes de ponderar.')
    elif any(n > 0 for _, n in presentes):
        nota('Hay factores positivos, pero se requiere verificar cual corresponde al modulo/estimando.')
    else:
        nota('No hay factores positivos entre los nombres esperados; revisar su disponibilidad.')
    nota('')

    nota('--- Control de fuga de datos y arquitectura ---')
    nota(f'Conglomerados distintos en maestro: {formatear_conteo(df["CONGLOME"].nunique())}')
    nota('Paso 11 usa GroupShuffleSplit por CONGLOME y GroupKFold, '
         'y excluye EDU_* como predictores, segun el codigo inspeccionado.')
    nota('Paso 11: Regresion Logistica y Random Forest si usan pesos de clase; '
         'HistGradientBoosting ganador y MLP no usan pesos de clase.')
    nota('Modelos avanzados explicitamente nombrados por la guia entre los entrenados: '
         'Random Forest y MLP. HistGradientBoosting es boosting, pero no figura por nombre en la lista.')
    nota('PENDIENTE para reforzar cumplimiento: evaluar un tercer modelo de la lista '
         '(XGBoost, LightGBM o CatBoost) SOLO mediante CV en TRAIN; '
         'no seleccionar usando el test ya consultado.')
    nota('No usar CLUSTER_EXPLORATORIO, P314A, P306 u otras variables EDU_* '
         'como predictores directos del desenlace en la prueba original.')
    nota('')
    nota('--- Recomendaciones tras esta auditoria ---')
    nota('1) Elegir y documentar el objetivo medible antes de cambiar modelos.')
    nota('2) Si el objetivo se redefine (p. ej., B), es un experimento NUEVO: '
         'partir train/test por conglomerado y volver a evaluar, no reutilizar métricas anteriores.')
    nota('3) Para el modelo actual, ensayar manejo del desbalance SOLO en CV de train; '
         'conservar intactos el clasificador final previo y SHAP.')
    nota('4) Analizar pertinencia del factor de expansión con documentación oficial recibida.')
    nota('5) Documentar claramente limitaciones de no uso, matrícula y competencias.')
    nota(f'CSV cruce principal: {ruta_tabla.relative_to(BASE)}')
    REPORTE.write_text('\n'.join(salida) + '\n', encoding='utf-8')
    print(f'\nREPORTE GUARDADO: {REPORTE}', flush=True)


if __name__ == '__main__':
    main()
