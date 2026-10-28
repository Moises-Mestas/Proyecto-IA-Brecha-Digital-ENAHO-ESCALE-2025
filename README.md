# Proyecto de Inteligencia Artificial: Brecha digital y educación en el Perú (ENAHO–ESCALE 2025)

**Curso:** Inteligencia Artificial · **Carrera:** Ingeniería de Sistemas  
**Autor:** Moises Joaquin Mestas Maque  
**Modalidad:** evaluación práctica e informe científico · **Herramientas:** Python y VS Code

> **Objetivo:** explorar datos oficiales proporcionados para el examen, integrar las condiciones del hogar con información educativa distrital, identificar perfiles mediante *clustering* y entrenar modelos supervisados para estudiar la brecha digital. Se incluye regresión opcional y explicabilidad SHAP.

## 👨‍🏫 Para el docente: empezar aquí

| ¿Qué revisar? | Evidencia principal |
|---|---|
| **1. Informe final (6 páginas, con EDA RAW)** | [Abrir PDF final](reports/Informe_Final_IA_ENAHO_ESCALE_2025_CON_EDA.pdf) |
| **2. EDA de archivos originales, antes de limpiar** | [Reporte EDA](outputs/eda_raw_2025/reporte_eda_raw_2025.txt) · [Código](src/24_eda_raw_2025.py) · [Gráfico de edades](outputs/figures/eda_raw_edades_6_17_2025.png) |
| **3. Limpieza e integración ENAHO–ESCALE** | [Limpieza](src/05_limpiar_enaho_2025.py) · [Unión ENAHO](src/06_integrar_enaho_2025.py) · [Unión MINEDU corregida](src/07_integrar_minedu_escale_2025_CORREGIDO.py) · [Calidad](outputs/reporte_calidad_dataset_2025.txt) |
| **4. Clustering (K-Means y GMM)** | [Resultados](outputs/reporte_clustering_2025.txt) · [Comparación de modelos](outputs/clustering/comparacion_modelos_2025.csv) · [PCA](outputs/figures/clustering_pca_2025.png) · [Silueta](outputs/figures/clustering_silueta_2025.png) |
| **5. Clasificación supervisada** | [Código](src/11_clasificacion_2025.py) · [Resultados originales](outputs/reporte_clasificacion_2025.txt) · [ROC](outputs/figures/clasificacion_roc_2025.png) · [Matriz](outputs/figures/clasificacion_matriz_confusion_2025.png) |
| **6. Optimización y desbalance** | [Optimización](outputs/reporte_optimizacion_2025.txt) · [XGBoost y pesos](outputs/reporte_paso18_xgboost_balanceo_objetivo_A.txt) |
| **7. Clasificación entre matriculados que asisten** | [Resultados del experimento](outputs/reporte_cierre19_clasificacion_asistentes_2025.txt) · [Código](src/19_cerrar_clasificacion_2025.py) |
| **8. Regresión opcional** | [Resultados](outputs/reporte_cierre20_indice_brecha_2025.txt) · [Código](src/20_indice_brecha_educativa_digital_2025.py) · [Real vs. predicho](outputs/figures/cierre20_regresion_real_vs_predicho.png) |
| **9. Explicabilidad SHAP** | [Reporte](outputs/reporte_shap_2025.txt) · [Beeswarm](outputs/figures/shap_beeswarm_top10_2025.png) · [Código](src/12_explicabilidad_shap_2025.py) |
| **10. Comprobaciones de integridad y rúbrica** | [Conciliación MINEDU](outputs/reporte_conciliacion_minedu_2025.txt) · [Auditoría ampliada](outputs/reporte_revision_rubrica_extendida_2025.txt) |

**Recorrido recomendado (3–5 minutos):** abrir el PDF → ver el reporte EDA → comprobar la tabla maestra y la conciliación MINEDU → leer las métricas de clustering y clasificación → observar el beeswarm SHAP → revisar los scripts que generaron cada resultado.

## 1. ¿Qué se hizo? (resumen sencillo)

1. **EDA de datos crudos (`raw`):** contamos registros, revisamos faltantes, edades, matrícula, uso de internet, ingresos/gastos y gestión educativa. Se leyeron los ZIP originales, sin entrenar modelos ni sustituir sus valores.
2. **Preprocesamiento:** seleccionamos el corte 2025, tratamos espacios en blanco como faltantes y conservamos las llaves y el UBIGEO como texto (incluidos sus ceros iniciales).
3. **Integración:** unimos ENAHO 100 (vivienda), 200 (personas), 300 (educación/TIC) y Sumaria (hogar), y añadimos indicadores distritales del Padrón y Padlocal MINEDU por UBIGEO.
4. **Clustering:** probamos K-Means y Gaussian Mixture Models (GMM), comparamos valores de `k=2` a `k=6` y usamos Silhouette, Davies–Bouldin y Calinski–Harabasz. Visualizamos con PCA.
5. **Clasificación:** predecimos **no uso de internet durante el mes anterior** (`P314A=2`). Comparamos una línea base y modelos avanzados. El entrenamiento/prueba se separó por **CONGLOME**; el escalado, la imputación y la codificación se ajustaron solo en entrenamiento.
6. **Mejoras de clasificación:** optimizamos hiperparámetros, comparamos pesos de clase y XGBoost, e hicimos otro experimento entre estudiantes matriculados que asisten.
7. **Regresión (opcional):** diseñamos un índice continuo educativo-digital y comparamos regresores. **Este índice es una construcción del proyecto, no un indicador oficial.**
8. **SHAP:** explicamos las predicciones mediante beeswarm, dos dependencias y dos casos waterfall.
9. **Informe:** consolidamos metodología, resultados, imágenes, limitaciones y EDA en un PDF de 6 páginas.

## 2. Datos utilizados y calidad

**Fuentes:** únicamente los archivos que proporcionó el docente: ENAHO (INEI) y Padrón/Padlocal de ESCALE (MINEDU). El ENAHO proporcionado fue un **panel 2021–2025, filtrado a 2025**; no se presenta como si fuera el archivo anual independiente.

| Fuente original (corte 2025) | Registros del EDA |
|---|---:|
| ENAHO 100 | 44.599 |
| ENAHO 200 | 115.145 |
| ENAHO 300 | 106.958 |
| ENAHO Sumaria | 33.702 |
| MINEDU Padrón | 112.724 |
| MINEDU Padlocal | 69.642 |

**Dataset integrado:** 24.195 estudiantes de 6 a 17 años, **75 columnas**, 0 llaves duplicadas y 100 % de UBIGEO válidos. La conciliación independiente de `MIN_PROP_PUBLICOS` encontró **0 distritos con discrepancias**. Los indicadores MINEDU representan el **contexto distrital**, no necesariamente la institución donde estudia cada encuestado.

**Dónde comprobarlo:** `outputs/reporte_calidad_dataset_2025.txt`, `outputs/reporte_conciliacion_minedu_2025.txt`, `outputs/eda_raw_2025/`. El CSV maestro queda en `data/processed/dataset_enaho_escale_2025.csv` **solo en la copia local del proyecto**, no en el repositorio público.

## 3. Principales resultados

| Experimento | Resultado verificado | Lectura correcta |
|---|---|---|
| Clustering (K-Means, `k=2`) | Silhouette **0,2179** (muestra de evaluación); grupos de **13.377** y **10.818** | Dos perfiles exploratorios con separación limitada; no es una división perfecta. |
| Clasificación original (HistGradientBoosting) | Test: ROC-AUC **0,8372**, AP **0,6780**, F1 macro **0,7185**, recall de no uso **0,4808** | Distingue clases razonablemente, pero deja sin detectar muchos casos positivos. |
| Ajuste de hiperparámetros HistGB | AP medio CV: **0,659338 → 0,662398** | Mejora pequeña **solo en validación cruzada**; no se reemplazó el modelo original en test. |
| XGBoost con pesos (objetivo original) | Recall medio CV **0,7542**; precisión **0,5299** | Detecta más casos de no uso, con más falsas alarmas. |
| Clasificación de asistentes (XGBoost) | Test: ROC-AUC **0,8366**, AP **0,6421**, recall **0,4353**; `n=4.435` | Nuevo experimento entre matriculados que asisten; no comparar directamente su AP con otra cohorte. |
| Regresión del índice propuesto (XGBoost) | Test: RMSE **0,2331**, MAE **0,1954**, R² **0,2072** | Explica una parte limitada de la variación del índice propio. |
| SHAP del clasificador original | Mayor contribución absoluta media: **EDAD** | Explica la salida del **modelo**, no demuestra causas sociales. |

**Códigos verificados con el cuestionario:** `P314A=1` sí usó internet el mes anterior; `P314A=2` no usó. `P306=1` está matriculado; `P306=2` no. El experimento educativo adicional usa `P307=1` como asistencia, que debe contrastarse con el cuestionario 2025 del docente.

## 4. Cómo se evitó la fuga de datos (*data leakage*)

- Entrenamiento y prueba **separados por CONGLOME**, sin conglomerados compartidos.
- **GroupKFold (3 folds)** dentro del entrenamiento para comparar y seleccionar modelos.
- Imputación, escalado, One-Hot y, cuando correspondió, pesos de clase calculados **dentro de cada entrenamiento**.
- No se utilizaron la etiqueta `P314A`, otros campos `EDU_*`, IDs de persona, UBIGEO ni clusters exploratorios como predictores de la clasificación original.
- Los componentes utilizados para construir el índice continuo se excluyeron como predictores directos en la regresión.
- Semilla aleatoria utilizada: **42**.

## 5. Estructura del proyecto

```text
PROYECTO_IA_ENAHO_ESCALE_2025/
├── data/
│   ├── raw/               # ZIP originales ENAHO y MINEDU (solo local)
│   ├── interim/           # archivos depurados/intermedios (solo local)
│   └── processed/         # CSV maestro y grupos (solo local)
├── notebooks/             # vacío: se trabajó con scripts .py
├── src/                   # scripts 01–24; usar el 07_CORREGIDO
├── outputs/
│   ├── eda_raw_2025/     # tablas del EDA
│   ├── figures/          # imágenes y gráficos
│   ├── clustering/        # métricas y perfiles
│   ├── clasificacion/     # clasificación original
│   ├── auditoria_clasificacion/
│   ├── cierre_2025/      # experimento educativo
│   ├── indice_brecha_2025/
│   ├── optimizacion/
│   ├── shap/
│   └── models/           # modelos joblib (solo local)
├── reports/
│   └── Informe_Final_IA_ENAHO_ESCALE_2025_CON_EDA.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

Los scripts `01` y `02` son de exploración inicial. El archivo `07_integrar_minedu_escale_2025.py` es una versión **anterior**: para reproducir los resultados se debe ejecutar **`07_integrar_minedu_escale_2025_CORREGIDO.py`**.

## 6. Instalación y reproducción en VS Code (Windows)

**Requisito:** Python 3.12 recomendado. Los microdatos de ENAHO y MINEDU **no se distribuyen en este repositorio**: deben obtenerse del material entregado por el docente, de acuerdo con sus condiciones de uso.

```powershell
# Desde la carpeta principal del proyecto
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Crear las carpetas de datos en caso de clonar el repositorio (Git no incluye las fuentes):

```powershell
New-Item -ItemType Directory -Force data/raw/inei, data/raw/minedu | Out-Null
```

Ubicar las fuentes originales con estos nombres:

```text
data/raw/inei/1041-Modulo1474.zip  # ENAHO 100
data/raw/inei/1041-Modulo1479.zip  # ENAHO 200
data/raw/inei/1041-Modulo1475.zip  # ENAHO 300
data/raw/inei/1041-Modulo1478.zip  # Sumaria
data/raw/minedu/00_Padron.zip
data/raw/minedu/00_Padlocal.zip
```

**Reproducir por etapas** (ejecutar desde la raíz del proyecto y en este orden):

```powershell
# EDA sobre las fuentes sin procesar
python src/24_eda_raw_2025.py

# Limpieza e integración
python src/05_limpiar_enaho_2025.py
python src/06_integrar_enaho_2025.py
python src/07_integrar_minedu_escale_2025_CORREGIDO.py
python src/08_auditar_dataset_maestro_2025.py
python src/16_comprobar_consistencia_minedu_2025.py

# Clustering y caracterización
python src/09_clustering_estudiantes_2025.py
python src/10_analizar_clusters_y_objetivo_2025.py
python src/13_grafico_silueta_2025.py

# Clasificación original y SHAP
python src/11_clasificacion_2025.py
python src/12_explicabilidad_shap_2025.py

# Experimentos adicionales (pueden consumir más recursos)
python src/15_optimizar_hiperparametros_2025.py
python src/17_auditar_objetivo_y_pesos_2025.py
python src/18_comparar_xgboost_y_balanceo_2025.py
python src/19_cerrar_clasificacion_2025.py --iteraciones 20
python src/20_indice_brecha_educativa_digital_2025.py --iteraciones 20
python src/23_verificar_rubrica_extendida_2025.py
```

**Nota sobre el PDF:** `src/21_generar_informe_consolidado_2025.py` combina el PDF IEEE base con los experimentos 19–20; necesita disponer previamente del PDF IEEE base. El **PDF definitivo con EDA** se entrega ya generado en `reports/`; su etapa de incorporación del EDA no está automatizada en los scripts anteriores. No prometer regeneración idéntica del PDF final con una sola orden.

## 7. Consideraciones científicas y de privacidad

- **No uso reciente de internet** no equivale a ausencia total de conectividad, falta de competencias digitales o bajo rendimiento académico.
- El **índice educativo-digital** fue definido por el equipo; sus pesos son una decisión metodológica, no una medición oficial.
- `SUM_FACTOR07` fue examinado como **análisis de sensibilidad**; no se estimaron resultados representativos de todo el Perú mediante el diseño muestral complejo.
- Clustering y SHAP muestran patrones o explicaciones del modelo, **no relaciones causales**.
- No subir a GitHub microdatos individuales, predicciones fila por fila ni modelos `.joblib` sin una revisión de privacidad y derechos de distribución. El `.gitignore` incluido ayuda a prevenir su publicación.

**Referencia de evaluación:** *Guía del proyecto IA: Brecha digital y rendimiento escolar en el Perú*, proporcionada por el docente, corte 2025. Los resultados presentados proceden de los reportes de ejecución almacenados en `outputs/`.
