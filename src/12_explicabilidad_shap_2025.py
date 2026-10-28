"""FASE 4 (3 puntos): Explicabilidad SHAP del clasificador ENAHO–ESCALE.

Ejecutar desde la raiz del proyecto:
    python src/12_explicabilidad_shap_2025.py

Necesita:
  data/processed/dataset_enaho_escale_2025.csv
  outputs/models/clasificador_2025.joblib  (creado en practica 11)

Genera: beeswarm top 10, 2 dependencias y 2 waterfall, tablas CSV
con importancias SHAP y reporte de verificaciones.

IMPORTANTE: SHAP EXPLICA EL MODELO, NO DEMUESTRA CAUSALIDAD.
SHAP para HistGradientBoosting usa su salida sin transformar (log-odds).
Para obtener una probabilidad, consultar predict_proba del pipeline.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent.parent
CSV_MAESTRO = BASE / "data" / "processed" / "dataset_enaho_escale_2025.csv"
ARCHIVO_MODELO = BASE / "outputs" / "models" / "clasificador_2025.joblib"
CARPETA_FIG = BASE / "outputs" / "figures"
CARPETA_TABLAS = BASE / "outputs" / "shap"
REPORTE = BASE / "outputs" / "reporte_shap_2025.txt"

OBJETIVO = "EDU_P314A"
NUMERICAS = [
    "EDAD", "SUM_INGHOG2D", "SUM_GASHOG2D",
    "MIN_RATIO_ALUMNOS_DOCENTE", "MIN_PROP_PUBLICOS",
]
CATEGORICAS = ["P207", "SUM_POBREZA", "VIV_P1121", "VIV_P1144"]
PREDICTORAS = NUMERICAS + CATEGORICAS
N_EXPLICACIONES = 300  # Muestra del TEST; no modifica el modelo.


def guardar_reporte(lineas, texto=""):
    mensaje = str(texto)
    print(mensaje, flush=True)
    lineas.append(mensaje)


def cargar_test():
    """Recrear exactamente el split del programa 11, sin volver a entrenar."""
    if not CSV_MAESTRO.is_file():
        raise FileNotFoundError(f"No existe el CSV maestro: {CSV_MAESTRO}")
    if not ARCHIVO_MODELO.is_file():
        raise FileNotFoundError(
            f"No existe el modelo entrenado: {ARCHIVO_MODELO}\n"
            "Ejecuta primero src/11_clasificacion_2025.py"
        )

    df = pd.read_csv(CSV_MAESTRO, dtype=str, encoding="utf-8-sig",
                     keep_default_na=False, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    faltantes = sorted(set(PREDICTORAS + [OBJETIVO, "CONGLOME", "AÑO"]) - set(df))
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias: {faltantes}")

    for columna in PREDICTORAS + [OBJETIVO, "CONGLOME", "AÑO"]:
        df[columna] = df[columna].str.strip()
    if not df["AÑO"].eq("2025").all():
        raise ValueError("Se esperaban exclusivamente datos de 2025.")
    df = df.loc[df[OBJETIVO].isin(["1", "2"])].copy()
    if df.empty:
        raise ValueError("No hay respuestas validas para EDU_P314A.")

    X = df[PREDICTORAS].copy()
    for columna in NUMERICAS:
        X[columna] = pd.to_numeric(
            X[columna].str.replace(",", ".", regex=False).replace("", np.nan),
            errors="coerce",
        )
    for columna in CATEGORICAS:
        X[columna] = X[columna].replace("", np.nan).astype(object)
    y = df[OBJETIVO].eq("2").astype(int)
    grupos = df["CONGLOME"].astype(str)

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.25,
                               random_state=RANDOM_STATE)
    indices_train, indices_test = next(divisor.split(X, y, groups=grupos))
    if set(grupos.iloc[indices_train]) & set(grupos.iloc[indices_test]):
        raise AssertionError("Hay conglomerados repetidos en train y test.")

    X_test = X.iloc[indices_test].copy()
    y_test = y.iloc[indices_test].copy()
    modelo = joblib.load(ARCHIVO_MODELO)  # Cargar solo archivos propios de confianza.
    if not hasattr(modelo, "named_steps") or not {
        "preprocesar", "modelo"
    }.issubset(set(modelo.named_steps)):
        raise TypeError("El archivo .joblib no contiene el Pipeline del programa 11.")
    return modelo, X_test, y_test, len(indices_train), len(indices_test)


def nombres_legibles(nombres):
    """Convertir one-hot a etiquetas legibles para la sustentacion."""
    etiquetas = []
    for original in nombres:
        if original.startswith("numericas__"):
            etiquetas.append(original.removeprefix("numericas__"))
        elif original.startswith("categoricas__"):
            nombre = original.removeprefix("categoricas__")
            pareja = next((c for c in CATEGORICAS if nombre.startswith(c + "_")), None)
            if pareja:
                etiquetas.append(f"{pareja}={nombre[len(pareja) + 1:]}")
            else:
                etiquetas.append(nombre)
        else:
            etiquetas.append(original)
    if len(etiquetas) != len(set(etiquetas)):
        raise ValueError("Hay nombres de variables transformadas repetidos.")
    return etiquetas


def main():
    CARPETA_FIG.mkdir(parents=True, exist_ok=True)
    CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)
    reporte = []
    guardar_reporte(reporte, "FASE 4 - EXPLICABILIDAD SHAP ENAHO–ESCALE 2025")
    guardar_reporte(reporte, "Objetivo: EDU_P314A codigo 2 = NO USO INTERNET MES ANTERIOR")
    guardar_reporte(reporte, "Resultados SHAP expresados en LOG-ODDS de no uso de internet.")

    pipeline, Xtest, ytest, ntrain, ntest = cargar_test()
    guardar_reporte(reporte, f"Train: {ntrain:,} | Test: {ntest:,}")
    probabilidades = pipeline.predict_proba(Xtest)[:, 1]
    auc = roc_auc_score(ytest, probabilidades)
    ap = average_precision_score(ytest, probabilidades)
    guardar_reporte(reporte, f"Modelo guardado: {type(pipeline.named_steps['modelo']).__name__}")
    guardar_reporte(reporte, f"Verificacion modelo guardado: ROC-AUC={auc:.4f}, AP={ap:.4f}")
    if abs(auc - 0.8372) > .025 or abs(ap - 0.6780) > .035:
        guardar_reporte(reporte, "ATENCION: metricas diferentes a practica 11. "
                        "Revisar si cambió el CSV o el modelo antes de interpretar las figuras.")

    # Solo el conjunto de TEST se utiliza para EXPLICAR. No se reajusta el modelo.
    rng = np.random.default_rng(RANDOM_STATE)
    cantidad = min(N_EXPLICACIONES, len(Xtest))
    indices_muestra = np.sort(rng.choice(len(Xtest), size=cantidad, replace=False))
    X_explicar = Xtest.iloc[indices_muestra].copy()
    y_explicar = ytest.iloc[indices_muestra].to_numpy()
    p_explicar = probabilidades[indices_muestra]

    procesador = pipeline.named_steps["preprocesar"]
    bosque = pipeline.named_steps["modelo"]  # HistGradientBoosting seleccionado en CV
    if bosque.__class__.__name__ != "HistGradientBoostingClassifier":
        raise TypeError(
            "Este programa está diseñado para el ganador de práctica 11: "
            "HistGradientBoostingClassifier. Si ganó otro modelo, "
            "necesitamos adaptar el explicador."
        )

    X_trans = np.asarray(procesador.transform(X_explicar), dtype=float)
    nombres_originales = list(procesador.get_feature_names_out())
    nombres = nombres_legibles(nombres_originales)
    if X_trans.shape[1] != len(nombres):
        raise AssertionError("Las columnas transformadas no coinciden con nombres.")
    guardar_reporte(reporte, f"Estudiantes explicados (muestra TEST): {cantidad}")
    guardar_reporte(reporte, f"Variables originales: {len(PREDICTORAS)}")
    guardar_reporte(reporte, f"Variables tras One-Hot: {len(nombres)}")

    # Explicador de arboles para el modelo de boosting entrenado.
    # Sus valores positivos aumentan el LOG-ODDS de la clase 1 (no uso).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        explicador = shap.TreeExplainer(bosque, model_output="raw")
        valores = explicador(X_trans)
    if valores.values.ndim != 2:
        raise ValueError(f"Forma SHAP no esperada: {valores.values.shape}")
    valores.feature_names = nombres
    valores.data = X_trans

    # Comprobar aditividad SHAP = salida raw del modelo.
    salida_raw = np.asarray(bosque._raw_predict(X_trans)).reshape(-1)
    reconstruccion = np.asarray(valores.base_values).reshape(-1) + valores.values.sum(axis=1)
    error_maximo = float(np.max(np.abs(salida_raw - reconstruccion)))
    guardar_reporte(reporte, f"Comprobacion aditividad (error maximo log-odds): {error_maximo:.8f}")
    if error_maximo > 1e-3:
        raise AssertionError("Los valores SHAP no reconstruyen la salida raw del modelo.")

    importancia = pd.DataFrame({
        "variable_transformada": nombres,
        "importancia_shap_media_absoluta_logodds": np.abs(valores.values).mean(axis=0),
    }).sort_values("importancia_shap_media_absoluta_logodds", ascending=False)
    ruta_importancia = CARPETA_TABLAS / "importancia_shap_top_variables_2025.csv"
    importancia.to_csv(ruta_importancia, index=False, encoding="utf-8-sig")
    guardar_reporte(reporte, "\nTOP 10 IMPORTANCIA SHAP (magnitud absoluta, NO causalidad):")
    for _, fila in importancia.head(10).iterrows():
        guardar_reporte(reporte, f"  {fila['variable_transformada']}: "
                        f"{fila['importancia_shap_media_absoluta_logodds']:.4f}")

    # (1) Beeswarm de las 10 variables transformadas con mayor importancia.
    plt.close("all")
    shap.plots.beeswarm(valores, max_display=10, show=False)
    plt.title("SHAP — No uso de internet (log-odds)")
    plt.tight_layout()
    ruta_beeswarm = CARPETA_FIG / "shap_beeswarm_top10_2025.png"
    plt.savefig(ruta_beeswarm, dpi=170, bbox_inches="tight")
    plt.close("all")

    # (2 y 3) Dos gráficas de dependencia para variables NUMERICAS.
    # Se seleccionan las dos más influyentes entre las cinco numéricas.
    numeric_indices = [nombres.index(c) for c in NUMERICAS if c in nombres]
    orden_numericas = sorted(numeric_indices,
                            key=lambda i: np.abs(valores.values[:, i]).mean(),
                            reverse=True)
    rutas_dependencia = []
    for numero, indice in enumerate(orden_numericas[:2], start=1):
        plt.close("all")
        shap.plots.scatter(valores[:, indice], show=False)
        plt.title(f"Dependencia SHAP: {nombres[indice]} (log-odds)")
        plt.tight_layout()
        ruta = CARPETA_FIG / f"shap_dependencia_{numero}_2025.png"
        plt.savefig(ruta, dpi=170, bbox_inches="tight")
        plt.close("all")
        rutas_dependencia.append(ruta)
        guardar_reporte(reporte, f"Dependencia {numero}: {nombres[indice]}")

    # (4 y 5) Dos casos anonimizados del conjunto de prueba:
    # uno con no uso real y otro con uso real, ambos bien clasificados si posible.
    positivos = np.flatnonzero((y_explicar == 1) & (p_explicar >= .5))
    negativos = np.flatnonzero((y_explicar == 0) & (p_explicar < .5))
    if len(positivos) == 0 or len(negativos) == 0:
        raise ValueError("La muestra explicada no incluye ambos tipos correctamente "
                         "clasificados. Ampliar N_EXPLICACIONES.")
    ejemplo_positivo = positivos[np.argmax(p_explicar[positivos])]
    ejemplo_negativo = negativos[np.argmin(p_explicar[negativos])]
    ejemplos = [("no_uso_real", ejemplo_positivo), ("uso_real", ejemplo_negativo)]
    for etiqueta, indice in ejemplos:
        plt.close("all")
        shap.plots.waterfall(valores[indice], max_display=10, show=False)
        plt.title(f"SHAP individual — {etiqueta.replace('_', ' ')}")
        plt.tight_layout()
        ruta = CARPETA_FIG / f"shap_waterfall_{etiqueta}_2025.png"
        plt.savefig(ruta, dpi=170, bbox_inches="tight")
        plt.close("all")
        guardar_reporte(
            reporte,
            f"Waterfall {etiqueta}: indice local anonimo={indice}, "
            f"real={int(y_explicar[indice])}, "
            f"probabilidad_no_uso={p_explicar[indice]:.3f}"
        )

    guardar_reporte(reporte, "\nARCHIVOS GENERADOS:")
    for ruta in [ruta_beeswarm, *rutas_dependencia,
                 CARPETA_FIG / "shap_waterfall_no_uso_real_2025.png",
                 CARPETA_FIG / "shap_waterfall_uso_real_2025.png",
                 ruta_importancia]:
        guardar_reporte(reporte, f"  {ruta.relative_to(BASE)}")
    guardar_reporte(reporte, "\nIMPORTANTE:")
    guardar_reporte(reporte, "- SHAP explica la prediccion del MODELO, no causas del no uso.")
    guardar_reporte(reporte, "- Variables con '=1' o '=2' son dummies one-hot.")
    guardar_reporte(reporte, "- SHAP se calcula en log-odds; no interpretar sus "
                    "numeros como puntos porcentuales.")
    guardar_reporte(reporte, "- Revisar diccionarios de P207, P1121, P1144 y pobreza ")
    guardar_reporte(reporte, "  antes de describir significados sociales definitivos.")
    guardar_reporte(reporte, "- Las conclusiones son del panel 2025 entregado, "
                    "no estimaciones nacionales representativas.")
    REPORTE.write_text("\n".join(reporte) + "\n", encoding="utf-8")
    print(f"\nReporte guardado: {REPORTE}", flush=True)


if __name__ == "__main__":
    main()
