# Anexo: no uso reciente de internet entre estudiantes que asisten

**Definicion:** ENAHO P306=1 (matriculado), P307=1 (asiste), P314A=2 (no usó internet el mes anterior). La variable no mide habilidades digitales ni notas escolares. No confundir con exclusión digital efectiva completa.

**Cohorte:** 17,992 personas, train 13,557, test 4,435; sin conglomerados compartidos.

## Resultados de validación cruzada (solo train)

| Modelo | AP | ROC-AUC | F1 macro | Recall no uso | Precisión no uso |
|---|---:|---:|---:|---:|---:|
| XGBoost_optimizado_sin_pesos | 0.6564 | 0.8338 | 0.7099 | 0.4537 | 0.6854 |
| XGBoost_sin_pesos | 0.6560 | 0.8337 | 0.7109 | 0.4635 | 0.6739 |
| XGBoost_optimizado_con_pesos | 0.6553 | 0.8335 | 0.7136 | 0.7653 | 0.5123 |
| XGBoost_con_pesos_clase | 0.6546 | 0.8330 | 0.7169 | 0.7617 | 0.5178 |
| HistGB_sin_pesos | 0.6539 | 0.8315 | 0.7106 | 0.4643 | 0.6729 |
| HistGB_con_pesos_clase | 0.6537 | 0.8302 | 0.7134 | 0.7528 | 0.5144 |
| MLP_Red_Neuronal | 0.6531 | 0.8324 | 0.7094 | 0.4627 | 0.6690 |
| Regresion_Logistica_BASE | 0.6446 | 0.8315 | 0.7140 | 0.7707 | 0.5118 |
| Random_Forest_pesos_clase | 0.6444 | 0.8266 | 0.7190 | 0.7322 | 0.5275 |
| XGBoost_factor_sensibilidad | 0.6435 | 0.8268 | 0.7135 | 0.7515 | 0.5146 |

## Modelo seleccionado por CV: XGBoost_optimizado_sin_pesos

**Test final:** ap=0.6421, roc_auc=0.8366, f1_macro=0.7017, balanced_accuracy=0.6813, accuracy=0.8072, recall_no_uso=0.4353, precision_no_uso=0.6587

**Matriz de confusión** (filas real uso/no uso; columnas pred uso/no uso): `[[3109, 244], [611, 471]]`.

**Precauciones:** el experimento C es distinto del A; NO comparar su test directamente con las cifras del modelo original. SUM_FACTOR07 se analizó solo como sensibilidad en entrenamiento, sin afirmar inferencia nacional. SHAP explica predicciones, no relaciones causales.

**Figuras:** `outputs/figures/cierre19_*.png`. **Reportes:** `outputs/cierre_2025/`.
