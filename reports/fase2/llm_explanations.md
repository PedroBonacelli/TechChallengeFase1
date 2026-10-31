# Etapa 6 — Explicações geradas pela LLM

**Provedor:** modo offline (forçado)

As explicações abaixo foram geradas a partir de payloads JSON fechados (salvos em `llm_payloads.json`), nunca dos dados brutos. Todo número citado no texto deve existir no payload correspondente — a Etapa 7 verifica isso automaticamente.

## Interpretação da otimização por GA

> *Público: equipe clínica / responsável técnico decidindo sobre adoção do modelo otimizado.*

O QUE MUDOU:
  - logistic_regression: mantém o mesmo número de casos malignos detectados (falsos negativos 1 -> 1 de 42 malignos no teste).
  - decision_tree: mantém o mesmo número de casos malignos detectados (falsos negativos 3 -> 3 de 42 malignos no teste).
  - random_forest: mantém o mesmo número de casos malignos detectados (falsos negativos 3 -> 3 de 42 malignos no teste).

TROCAS ENVOLVIDAS:
  - logistic_regression: falsos positivos 3 -> 3 (variação de +0 biópsias desnecessárias).
  - decision_tree: falsos positivos 4 -> 12 (variação de +8 biópsias desnecessárias).
  - random_forest: falsos positivos 0 -> 4 (variação de +4 biópsias desnecessárias).

QUAL MODELO USAR: pelo critério de minimizar casos malignos não detectados, Reg. Logística otimizado é o que deixa passar menos casos malignos: 1 não detectado(s) de 42, ao custo de 3 encaminhamento(s) desnecessário(s).

RESSALVAS: conjunto de teste pequeno (114 amostras, 42 malignos), então cada caso vale cerca de 1 ponto percentual e diferenças de 1-2 casos estão dentro do ruído amostral. A otimização cobriu apenas hiperparâmetros — não houve mudança de features, de pré-processamento nem de limiar de decisão. A adoção de qualquer um dos modelos é decisão do profissional responsável pelo serviço, à luz do protocolo da instituição.

[texto gerado por template determinístico, sem LLM]

---

## Explicações por caso

### Caso #53 — rótulo real: maligno

**Por que este caso foi selecionado:** falso negativo do modelo logistic_regression — maligno classificado como benigno

| Modelo | Sugestão | P(maligno) | Confiança |
|---|---|---|---|
| logistic_regression | benigno | 0.1738 | 0.8262 |
| decision_tree | maligno | 0.8341 | 0.8341 |
| random_forest | benigno | 0.4466 | 0.5534 |

*Divergência entre modelos: sim*

**Explicação gerada:**

AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso.
  - logistic_regression: sugere benigno (probabilidade de maligno 17,4%).
  - decision_tree: sugere maligno (probabilidade de maligno 83,4%).
  - random_forest: sugere benigno (probabilidade de maligno 44,7%).

PRINCIPAIS FATORES:
  - worst texture: -0,77 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - radius error: -0,45 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - worst concave points: 0,37 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - worst perimeter: 0,11 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.

GRAU DE CONFIANÇA: o modelo de referência (logistic_regression) acerta 96,5% dos casos do conjunto de teste e identifica 97,6% dos casos malignos. Neste caso específico a confiança é de 82,6%, dentro da faixa em que o modelo é mais estável.

CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança — recomenda-se revisão humana da lâmina e investigação complementar antes de qualquer conclusão. A decisão sobre a conduta é do profissional responsável.

[texto gerado por template determinístico, sem LLM (caso #53)]

---

### Caso #51 — rótulo real: benigno

**Por que este caso foi selecionado:** os 3 modelos divergem neste caso

| Modelo | Sugestão | P(maligno) | Confiança |
|---|---|---|---|
| logistic_regression | maligno | 0.5480 | 0.5480 |
| decision_tree | maligno | 0.9324 | 0.9324 |
| random_forest | benigno | 0.3491 | 0.6509 |

*Divergência entre modelos: sim*

**Explicação gerada:**

AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso.
  - logistic_regression: sugere maligno (probabilidade de maligno 54,8%).
  - decision_tree: sugere maligno (probabilidade de maligno 93,2%).
  - random_forest: sugere benigno (probabilidade de maligno 34,9%).

PRINCIPAIS FATORES:
  - worst texture: 1,07 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - radius error: -0,36 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - compactness error: -0,75 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para maligno.
  - worst smoothness: -1,17 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.

GRAU DE CONFIANÇA: o modelo de referência (logistic_regression) acerta 96,5% dos casos do conjunto de teste e identifica 97,6% dos casos malignos. Neste caso específico a confiança é de 54,8% — abaixo do limiar de 65% adotado, o que coloca o caso na faixa em que o modelo erra com mais frequência.

CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança — recomenda-se revisão humana da lâmina e investigação complementar antes de qualquer conclusão. A decisão sobre a conduta é do profissional responsável.

[texto gerado por template determinístico, sem LLM (caso #51)]

---

### Caso #84 — rótulo real: maligno

**Por que este caso foi selecionado:** os 3 modelos divergem neste caso

| Modelo | Sugestão | P(maligno) | Confiança |
|---|---|---|---|
| logistic_regression | maligno | 0.5554 | 0.5554 |
| decision_tree | maligno | 0.8341 | 0.8341 |
| random_forest | benigno | 0.4458 | 0.5542 |

*Divergência entre modelos: sim*

**Explicação gerada:**

AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso.
  - logistic_regression: sugere maligno (probabilidade de maligno 55,5%).
  - decision_tree: sugere maligno (probabilidade de maligno 83,4%).
  - random_forest: sugere benigno (probabilidade de maligno 44,6%).

PRINCIPAIS FATORES:
  - worst texture: -0,87 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - worst perimeter: 0,33 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - radius error: -0,48 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - worst symmetry: 0,78 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.

GRAU DE CONFIANÇA: o modelo de referência (logistic_regression) acerta 96,5% dos casos do conjunto de teste e identifica 97,6% dos casos malignos. Neste caso específico a confiança é de 55,5% — abaixo do limiar de 65% adotado, o que coloca o caso na faixa em que o modelo erra com mais frequência.

CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança — recomenda-se revisão humana da lâmina e investigação complementar antes de qualquer conclusão. A decisão sobre a conduta é do profissional responsável.

[texto gerado por template determinístico, sem LLM (caso #84)]

---

### Caso #3 — rótulo real: benigno

**Por que este caso foi selecionado:** os 3 modelos divergem neste caso

| Modelo | Sugestão | P(maligno) | Confiança |
|---|---|---|---|
| logistic_regression | benigno | 0.4157 | 0.5843 |
| decision_tree | maligno | 0.9324 | 0.9324 |
| random_forest | maligno | 0.6038 | 0.6038 |

*Divergência entre modelos: sim*

**Explicação gerada:**

AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso.
  - logistic_regression: sugere benigno (probabilidade de maligno 41,6%).
  - decision_tree: sugere maligno (probabilidade de maligno 93,2%).
  - random_forest: sugere maligno (probabilidade de maligno 60,4%).

PRINCIPAIS FATORES:
  - worst perimeter: 0,32 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - worst concave points: -0,34 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - radius error: -0,23 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - worst symmetry: -0,80 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.

GRAU DE CONFIANÇA: o modelo de referência (logistic_regression) acerta 96,5% dos casos do conjunto de teste e identifica 97,6% dos casos malignos. Neste caso específico a confiança é de 58,4% — abaixo do limiar de 65% adotado, o que coloca o caso na faixa em que o modelo erra com mais frequência.

CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança — recomenda-se revisão humana da lâmina e investigação complementar antes de qualquer conclusão. A decisão sobre a conduta é do profissional responsável.

[texto gerado por template determinístico, sem LLM (caso #3)]

---

### Caso #45 — rótulo real: benigno

**Por que este caso foi selecionado:** os 3 modelos divergem neste caso

| Modelo | Sugestão | P(maligno) | Confiança |
|---|---|---|---|
| logistic_regression | benigno | 0.3876 | 0.6124 |
| decision_tree | maligno | 0.9324 | 0.9324 |
| random_forest | benigno | 0.4388 | 0.5612 |

*Divergência entre modelos: sim*

**Explicação gerada:**

AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso.
  - logistic_regression: sugere benigno (probabilidade de maligno 38,8%).
  - decision_tree: sugere maligno (probabilidade de maligno 93,2%).
  - random_forest: sugere benigno (probabilidade de maligno 43,9%).

PRINCIPAIS FATORES:
  - worst texture: -0,67 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.
  - worst concave points: 0,41 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - worst perimeter: 0,14 desvios-padrão acima da média do grupo de referência; empurra a sugestão para maligno.
  - radius error: -0,19 desvios-padrão abaixo da média do grupo de referência; empurra a sugestão para benigno.

GRAU DE CONFIANÇA: o modelo de referência (logistic_regression) acerta 96,5% dos casos do conjunto de teste e identifica 97,6% dos casos malignos. Neste caso específico a confiança é de 61,2% — abaixo do limiar de 65% adotado, o que coloca o caso na faixa em que o modelo erra com mais frequência.

CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança — recomenda-se revisão humana da lâmina e investigação complementar antes de qualquer conclusão. A decisão sobre a conduta é do profissional responsável.

[texto gerado por template determinístico, sem LLM (caso #45)]

---

## Consumo da LLM nesta execução

```json
{
  "chamadas": 6,
  "chamadas_api": 0,
  "chamadas_offline": 6,
  "erros": 0,
  "input_tokens": 0,
  "output_tokens": 0,
  "custo_estimado_usd": 0.0,
  "latencia_media_s": 0.0
}
```