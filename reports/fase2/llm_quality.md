# Etapa 7 — Avaliação da qualidade das interpretações da LLM

**Provedor:** modo offline (forçado)

## Critério de avaliação

Duas camadas independentes:

**Camada 1 — verificações determinísticas** (código, sem LLM). Nota 0-5, descontando: 2,0 por qualquer número citado que não exista no payload de origem; 2,0 por linguagem que feche diagnóstico; 0,5 por jargão de ML não traduzido; 0,5 por seção ausente; 0,5 por não registrar que a decisão é do profissional. Os dois descontos de 2,0 são falhas de segurança clínica; os de 0,5, falhas de usabilidade.

**Camada 2 — LLM como juiz.** Um segundo prompt, com instrução explícita de ceticismo, atribui nota 1-5 em cinco critérios: fidelidade numérica, ausência de diagnóstico, clareza clínica, acionabilidade e calibração da incerteza.

Quando as duas discordam no critério numérico, a Camada 1 prevalece — ela compara contra o payload, o juiz compara contra a própria leitura. A coluna `divergencia_fidelidade` registra esses casos, porque eles medem a confiabilidade do juiz.

## Resultados

| item                        | origem   |   nota_deterministica | nota_juiz   |   numeros_citados |   numeros_inventados |   linguagem_conclusiva |   jargao_ml | formato_ok   |   palavras |
|:----------------------------|:---------|----------------------:|:------------|------------------:|---------------------:|-----------------------:|------------:|:-------------|-----------:|
| interpretação da otimização | offline  |                     5 |             |                24 |                    0 |                      0 |           0 | True         |        207 |
| caso #53                    | offline  |                     5 |             |                11 |                    0 |                      0 |           0 | True         |        185 |
| caso #51                    | offline  |                     5 |             |                12 |                    0 |                      0 |           0 | True         |        196 |
| caso #84                    | offline  |                     5 |             |                12 |                    0 |                      0 |           0 | True         |        196 |
| caso #3                     | offline  |                     5 |             |                12 |                    0 |                      0 |           0 | True         |        197 |
| caso #45                    | offline  |                     5 |             |                12 |                    0 |                      0 |           0 | True         |        197 |

## Exemplos documentados

### Boa resposta — interpretação da otimização (nota determinística 5.0)

*Por que é boa:* nenhum número sem respaldo no payload, sem linguagem conclusiva, as quatro seções presentes, 207 palavras.

```
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
```

## Consumo da LLM nesta avaliação

```json
{
  "chamadas": 0
}
```