# Alternativa B - Indice continuo educativo-digital (propuesto)

**Definicion reproducible:** I = 0.50 × I(no usó internet el mes anterior, P314A=2) + 0.30 × I(no asiste: P306=2 o P307=2 si matriculado) + 0.20 × ratio_alumnos_docente/(ratio_alumnos_docente + 15).

**Aviso:** constructo exploratorio no oficial, sin validación externa; no mide calificaciones ni competencias digitales. La escuela del distrito no implica institución a la que asiste cada estudiante.

Se analizaron **23,618** registros completos; train=17,690, test=5,928. GroupKFold 3 particiones y conjunto de test separado por conglomerado. **Se excluyeron de X las variables que construyen directamente Y.**

## Modelos: CV de TRAIN

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| XGBoost_Reg_optimizado | 0.2369 | 0.1971 | 0.1886 |
| XGBoost_Reg | 0.2374 | 0.1968 | 0.1855 |
| MLP_Reg | 0.2384 | 0.1974 | 0.1785 |
| HistGB_Reg | 0.2387 | 0.1975 | 0.1761 |
| Ridge_BASE | 0.2403 | 0.2031 | 0.1651 |
| Random_Forest_Reg | 0.2414 | 0.1981 | 0.1578 |
| Media_BASE | 0.2631 | 0.2346 | -0.0008 |

**Seleccionado:** XGBoost_Reg_optimizado. **Test:** RMSE=0.2331, MAE=0.1954, R²=0.2072.

Sensibilidad del indice al peso de componentes: correlaciones registradas en `outputs/indice_brecha_2025/`. No se infiere causalidad ni se declara representatividad poblacional.
