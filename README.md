# Proyecto IA: Brecha digital y educación en el Perú (ENAHO–ESCALE 2025)

**Autor:** Moises Joaquin Mestas Maque  
**Carrera:** Ingeniería de Sistemas · **Curso:** Inteligencia Artificial  
**Tecnologías:** Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib  
**Fuentes:** Microdatos ENAHO (INEI) y Padrón/Padlocal (MINEDU–ESCALE) proporcionados en la evaluación.

> **Propósito:** realizar EDA sobre datos crudos, limpiar e integrar fuentes, caracterizar estudiantes con clustering, predecir un indicador de no uso reciente de internet y explicar las predicciones con SHAP. Se incluye regresión como actividad opcional.

## Para el docente: empezar aquí

| Requisito de evaluación | Evidencia principal |
|---|---|
| **Informe científico de 6 páginas, con EDA** | [Abrir PDF definitivo](reports/Informe_Final_IA_ENAHO_ESCALE_2025_CON_EDA.pdf) |
| **1. Datos originales** | [`data/raw/inei/`](data/raw/inei/) y [`data/raw/minedu/`](data/raw/minedu/) |
| **2. EDA sobre archivos raw** | [Código EDA](src/24_eda_raw_2025.py) · [Estadísticas](outputs/eda_raw_2025/eda_estadisticas_numericas_raw_2025.csv) · [Faltantes](outputs/eda_raw_2025/eda_faltantes_raw_2025.csv) · [Figura de edades](outputs/figures/eda_raw_edades_6_17_2025.png) |
| **3. Preprocesamiento e integración** | [Limpieza](src/05_limpiar_enaho_2025.py) · [Unión ENAHO](src/06_integrar_enaho_2025.py) · [Integración MINEDU CORREGIDA](src/07_integrar_minedu_escale_2025_CORREGIDO.py) |
| **4. Dataset procesado obligatorio** | [Abrir CSV maestro](data/processed/dataset_enaho_escale_2025.csv) · [Inventario de columnas](outputs/diccionario_columnas_maestro_2025.csv) |
| **5. Clustering: K-Means y GMM** | [Código](src/09_clustering_estudiantes_2025.py) · [Comparación](outputs/clustering/comparacion_modelos_2025.csv) · [Perfiles](outputs/clustering/perfiles_clusters_2025.csv) · [PCA](outputs/figures/clustering_pca_2025.png) · [Silueta](outputs/figures/clustering_silueta_2025.png) |
| **6. Clasificación y evaluación** | [Código](src/11_clasificacion_2025.py) · [Comparación CV](outputs/clasificacion/comparacion_modelos_cv_2025.csv) · [Métricas test](outputs/clasificacion/metricas_test_2025.csv) · [ROC](outputs/figures/clasificacion_roc_2025.png) · [Matriz de confusión](outputs/figures/clasificacion_matriz_confusion_2025.png) |
| **7. Optimización y desbalance** | [Optimización](src/15_optimizar_hiperparametros_2025.py) · [XGBoost y pesos](src/18_comparar_xgboost_y_balanceo_2025.py) · [Búsqueda](outputs/optimizacion/busqueda_hiperparametros_2025.csv) |
| **8. Clasificación de estudiantes asistentes** | [Código](src/19_cerrar_clasificacion_2025.py) · [Comparación CV](outputs/cierre_2025/cierre19_comparacion_cv.csv) · [Test](outputs/cierre_2025/cierre19_metricas_test.csv) |
| **9. Regresión opcional** | [Código](src/20_indice_brecha_educativa_digital_2025.py) · [CV](outputs/indice_brecha_2025/cierre20_cv_regresion.csv) · [Test](outputs/indice_brecha_2025/cierre20_metricas_test.csv) · [Real vs. predicho](outputs/figures/cierre20_regresion_real_vs_predicho.png) |
| **10. Explicabilidad SHAP** | [Código](src/12_explicabilidad_shap_2025.py) · [Importancia](outputs/shap/importancia_shap_top_variables_2025.csv) · [Beeswarm](outputs/figures/shap_beeswarm_top10_2025.png) · [Dependencia 1](outputs/figures/shap_dependencia_1_2025.png) · [Waterfall](outputs/figures/shap_waterfall_no_uso_real_2025.png) |

**Ruta rápida de revisión:** PDF final → EDA raw → dataset maestro → comparación K-Means/GMM → resultados de clasificación → regresión opcional → SHAP → código correspondiente.

## ¿Qué hicimos? (en palabras sencillas)

1. **EDA RAW:** revisamos los archivos originales, contamos registros, identificamos faltantes y graficamos edades, matrícula, uso de internet, ingresos/gastos y gestión educativa.
2. **Preprocesamiento:** seleccionamos el corte 2025, corregimos espacios vacíos y conservamos las llaves como texto para no perder ceros de UBIGEO.
3. **Integración:** unimos ENAHO 100, 200, 300 y Sumaria por persona/hogar, y añadimos información MINEDU agregada por distrito.
4. **Clustering:** comparamos K-Means y GMM con silueta, Davies–Bouldin y Calinski–Harabasz; visualizamos los grupos mediante PCA.
5. **Clasificación:** probamos Regresión Logística, Random Forest, MLP, HistGradientBoosting y XGBoost, con validación cruzada por conglomerado y control del desbalance.
6. **Regresión opcional:** entrenamos modelos para estimar un índice continuo educativo-digital definido en el proyecto.
7. **SHAP:** mostramos qué variables afectan las predicciones del modelo, sin confundir asociación con causalidad.

## Resultados principales

| Análisis | Resultado |
|---|---|
| Dataset integrado | **24.195 estudiantes** de 6–17 años, **75 columnas**, 0 llaves duplicadas |
| K-Means | **2 clusters**, silueta **0,2179** (muestra): separación limitada |
| Clasificación original (HistGradientBoosting, test) | ROC-AUC **0,8372**, AP **0,6780**, F1 macro **0,7185** |
| Clasificación entre asistentes (XGBoost, test) | ROC-AUC **0,8366**, AP **0,6421** |
| Regresión de índice propuesto (XGBoost, test) | RMSE **0,2331**, R² **0,2072** |
| SHAP | Edad: mayor contribución absoluta media en la clasificación original |

Los archivos ENAHO entregados proceden de un **panel 2021–2025, filtrado al año 2025**, no de una descarga anual independiente. Los indicadores MINEDU corresponden al contexto **distrital**, no a la institución concreta de cada estudiante.

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/                 # Fuentes ENAHO y MINEDU del examen
│   ├── interim/             # Datos intermedios de limpieza/integración
│   └── processed/           # Dataset final y clusters
├── notebooks/               # No requerido: se trabajó con scripts .py
├── src/                     # Código reproducible
├── outputs/
│   ├── figures/             # Gráficos EDA, PCA, SHAP, ROC, etc.
│   ├── clustering/          # Resultados agregados
│   ├── clasificacion/       # Métricas agregadas
│   ├── indice_brecha_2025/  # Métricas de regresión
│   └── ...
├── reports/                 # PDF definitivo y anexos
├── README.md
├── requirements.txt
└── .gitignore
```

> **Nota:** los archivos `.txt` auxiliares de `outputs/` y los modelos binarios `.joblib` no se publican en esta versión. El código y los CSV de métricas permiten revisar los resultados. El archivo `diccionario_columnas_maestro_2025.csv` es un **inventario de columnas**; conviene complementar definiciones de variables derivadas cuando corresponda.

## Reproducir en VS Code

**Entorno:** Python 3.12 recomendado. Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Ejecutar en este orden (desde la raíz):**

```powershell
# EDA sobre datos crudos
python src/24_eda_raw_2025.py

# Limpieza e integración
python src/05_limpiar_enaho_2025.py
python src/06_integrar_enaho_2025.py
python src/07_integrar_minedu_escale_2025_CORREGIDO.py
python src/08_auditar_dataset_maestro_2025.py

# Clustering
python src/09_clustering_estudiantes_2025.py
python src/10_analizar_clusters_y_objetivo_2025.py
python src/13_grafico_silueta_2025.py

# Clasificación y SHAP
python src/11_clasificacion_2025.py
python src/12_explicabilidad_shap_2025.py

# Experimentos complementarios
python src/15_optimizar_hiperparametros_2025.py
python src/18_comparar_xgboost_y_balanceo_2025.py
python src/19_cerrar_clasificacion_2025.py --iteraciones 20
python src/20_indice_brecha_educativa_digital_2025.py --iteraciones 20
```

**Prevención de fuga de datos:** separación de entrenamiento y prueba por `CONGLOME` y `GroupKFold` sobre entrenamiento, con imputación, escalado y codificación ajustados únicamente en cada fold. Semilla aleatoria fija: `42`.

**PDF final:** [Informe científico ENAHO–ESCALE 2025](reports/Informe_Final_IA_ENAHO_ESCALE_2025_CON_EDA.pdf). El PDF incorpora el EDA y los resultados; la maquetación final se hizo como documento y no se regenera idénticamente con una sola orden del pipeline.

## Límites e interpretación

- `P314A=2` indica **no utilizó internet el mes anterior**; no mide todas las competencias digitales.
- El índice de regresión es una propuesta metodológica del proyecto, **no un indicador oficial**.
- Los resultados no son estimaciones representativas del Perú porque no se aplicó inferencia con el diseño muestral complejo.
- SHAP explica el modelo, **no demuestra causalidad**.
- Antes de compartir públicamente los ZIP originales y los CSV por estudiante, confirmar las condiciones de redistribución del material y evitar información identificable.
