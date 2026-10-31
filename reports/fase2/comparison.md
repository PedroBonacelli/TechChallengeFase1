## Comparativo — modelos originais (Fase 1) vs. otimizados por GA

### Métricas no conjunto de teste (114 amostras, 42 malignos)

| Modelo | Experimento vencedor | Recall maligno | F1 maligno | Acurácia | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| Reg. Logística | `intensivo` | 0.9762 -> **0.9762** (=) | 0.9535 -> 0.9535 (=) | 0.9649 -> 0.9649 (=) | 0.9947 -> 0.9970 (+0.0023) | 1 -> 1 (+0) |
| Árvore de Decisão | `equilibrado` | 0.9286 -> **0.9286** (=) | 0.9176 -> 0.8387 (-0.0789) | 0.9386 -> 0.8684 (-0.0702) | 0.9289 -> 0.8889 (-0.0400) | 3 -> 3 (+0) |
| Random Forest | `equilibrado` | 0.9286 -> **0.9286** (=) | 0.9630 -> 0.9176 (-0.0453) | 0.9737 -> 0.9386 (-0.0351) | 0.9957 -> 0.9934 (-0.0023) | 3 -> 3 (+0) |

### Fitness de validação cruzada no treino (a métrica que o GA otimizou)

| Modelo | Fitness original | Fitness otimizado | Ganho |
|---|---|---|---|
| Reg. Logística | 0.9604 | **0.9732** | +0.0128 |
| Árvore de Decisão | 0.8906 | **0.9331** | +0.0425 |
| Random Forest | 0.9361 | **0.9533** | +0.0172 |

### Hiperparâmetros encontrados

**Reg. Logística** (experimento `intensivo`)

| Hiperparâmetro | Fase 1 | Otimizado pelo GA |
|---|---|---|
| `C` | 1 | **1.039** |
| `penalty` | `l2` | **`l1`** |
| `solver` | `lbfgs` | **`liblinear`** |
| `max_iter` | `5000` | **`4000`** |
| `l1_ratio` | `None` | `None` |
| `class_weight` | `balanced` | `balanced` |

**Árvore de Decisão** (experimento `equilibrado`)

| Hiperparâmetro | Fase 1 | Otimizado pelo GA |
|---|---|---|
| `criterion` | `gini` | `gini` |
| `max_depth` | `5` | **`19`** |
| `min_samples_split` | `2` | **`9`** |
| `min_samples_leaf` | `1` | **`3`** |
| `max_features` | `None` | **`log2`** |
| `ccp_alpha` | 0 | **0.02186** |
| `class_weight` | `balanced` | `balanced` |

**Random Forest** (experimento `equilibrado`)

| Hiperparâmetro | Fase 1 | Otimizado pelo GA |
|---|---|---|
| `n_estimators` | `300` | **`150`** |
| `max_depth` | `8` | **`19`** |
| `min_samples_split` | `2` | **`3`** |
| `min_samples_leaf` | `1` | **`6`** |
| `max_features` | `sqrt` | **0.5** |
| `criterion` | `gini` | `gini` |
| `class_weight` | `balanced` | `balanced` |
| `bootstrap` | `True` | `True` |
