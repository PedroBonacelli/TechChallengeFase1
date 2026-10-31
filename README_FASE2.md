# Fase 2 — Otimização por Algoritmo Genético + Integração com LLM

Segunda fase do Tech Challenge (IADT). Pega os três modelos de
diagnóstico treinados na Fase 1 sobre o Breast Cancer Wisconsin, otimiza
seus hiperparâmetros com um **algoritmo genético implementado do zero**, e
acrescenta uma camada de **LLM** que traduz os resultados numéricos em
texto que um médico consiga usar.

> Para a Fase 1 (EDA, modelagem, CNN), ver [README.md](README.md).

## O resultado em uma frase

O GA melhorou o fitness de validação cruzada nos três modelos
(+1,3 / +4,3 / +1,7 pontos percentuais), **mas esse ganho não transferiu
para o conjunto de teste** — e o relatório técnico mostra por quê, com a
correlação medida entre as duas coisas (Spearman **−1,00** na Árvore,
**−0,63** na Floresta).

O experimento corretivo fecha o argumento: trocando a régua de avaliação
de CV 5-fold para CV 5×3, **cada conjunto de hiperparâmetros vence apenas
no estimador contra o qual foi otimizado — 6 de 6 comparações**. O ganho
de 4,25 pontos da Árvore encolhe para 0,61 num estimador menos ruidoso;
86% dele era ruído de uma partição específica.

O achado negativo é tratado como resultado, não escondido.

Leia [`reports/fase2/RELATORIO_TECNICO_FASE2.md`](reports/fase2/RELATORIO_TECNICO_FASE2.md)
primeiro.

## Estrutura

```
src/
├── ga/                      # algoritmo genético
│   ├── search_space.py      # Etapa 1 — codificação dos genes
│   ├── operators.py         # Etapa 2 — seleção, cruzamento, mutação
│   ├── fitness.py           # Etapa 2 — função fitness (CV no treino)
│   ├── engine.py            # laço geracional com elitismo
│   ├── data.py              # carrega os dados processados da Fase 1
│   ├── run_experiments.py   # Etapa 3 — 3 modelos x 3 experimentos
│   └── compare.py           # Etapa 4 — comparativo e gráficos
├── llm/                     # camada de linguagem natural
│   ├── prompts.py           # Etapa 6 — prompts, com histórico de versões
│   ├── client.py            # cliente Anthropic + fallback offline
│   ├── offline.py           # gerador por template (sem LLM)
│   ├── explain.py           # Etapa 6 — geração das explicações
│   └── evaluate_quality.py  # Etapa 7 — avaliação em duas camadas
├── monitoring.py            # Etapa 5 — logging estruturado e monitoramento
└── lambda_handler.py        # Etapa 11 — handler de inferência serverless

tests/                       # Etapa 8 — 201 testes
infra/                       # Etapa 11 — Terraform (não aplicado)
reports/fase2/               # tabelas, figuras e relatórios
logs/                        # log legível + log estruturado JSONL
```

## Como executar

Pré-requisito: o pipeline da Fase 1 já rodou e `data/processed/` existe.
Se não:

```bash
python src/load_data.py
python src/preprocessing.py
```

### Ambiente

A Fase 2 reaproveita o ambiente da Fase 1 e acrescenta três pacotes:

```bash
pip install -r requirements.txt          # Fase 1
pip install -r requirements-fase2.txt    # Fase 2
```

Para a Etapa 6, copie `.env.example` para `.env` e preencha a chave:

```bash
cp .env.example .env
# edite .env: ANTHROPIC_API_KEY=sk-ant-...
```

Sem chave nada quebra — a camada de LLM cai num gerador determinístico
por template e se identifica como tal. Os testes nunca chamam a API.

### Pipeline

Os módulos da Fase 2 rodam com `-m` (são pacote, não script solto):

```bash
# Etapa 3 — otimização (3 modelos x 3 configurações, ~28 min)
python -m src.ga.run_experiments
python -m src.ga.run_experiments --quick          # versão rápida, pra validar

# Etapa 4 — comparativo, análise de generalização e gráficos
python -m src.ga.compare

# Etapa 6 — explicações em linguagem natural
python -m src.llm.explain
python -m src.llm.explain --offline               # sem chamar a API

# Etapa 7 — avaliação da qualidade das explicações
python -m src.llm.evaluate_quality --comparar-prompts

# Etapa 8 — testes
python -m pytest
```

Experimento corretivo do sobreajuste de fold (ver o relatório técnico):

```bash
python -m src.ga.run_experiments --fitness-robusto
```

Inspeção dos componentes isolados:

```bash
python -m src.ga.search_space   # imprime os 3 espaços de busca
python -m src.ga.data           # confere os dados carregados
```

### Documento de entrega

```bash
python reports/fase2/build_entrega_docx.py
```

Gera `reports/fase2/entrega_final_fase2.docx` — o documento no molde do
`reports/entrega_final.md` da Fase 1, com **as pendências em vermelho**.
Ao abrir no Word: **Ctrl+A** e **F9** para gerar o sumário (ele é um
campo, abre vazio), depois **Arquivo → Exportar → Criar PDF**.

O script é a fonte do documento; o `.docx` é saída regenerável —
reexecutar sobrescreve edições feitas no Word.

## As decisões que definem a solução

**Codificação real normalizada.** Todo cromossomo é um vetor em
`[0,1]^n`; o decodificador traduz cada posição pro domínio do
hiperparâmetro (categórico, inteiro ou contínuo). Cruzamento fica
genérico, escala logarítmica sai de graça, e limite nunca é violado. A
mutação, mesmo assim, é sensível ao tipo: gene categórico é re-sorteado,
porque perturbação gaussiana quase nunca atravessa a fronteira de uma
faixa. Ver [`src/ga/search_space.py`](src/ga/search_space.py).

**Fitness medido por CV no treino, nunca no teste.** Se o fitness fosse
medido no conjunto de teste, o GA otimizaria contra ele e todo número da
Etapa 4 seria vazamento. O teste fica intocado até a avaliação final.

**Recall do maligno domina a função objetivo** (peso 0,60, contra 0,25 do
F1 e 0,15 da acurácia). É a tradução numérica do custo clínico
assimétrico já estabelecido na Fase 1: um falso negativo é um câncer sem
tratamento; um falso positivo é uma biópsia a mais. O F1 entra para
impedir a solução degenerada de chamar tudo de maligno.

**Orçamento igual nos três experimentos** (600 avaliações cada). Se um
tivesse o dobro do orçamento, um resultado melhor não diria nada sobre a
estratégia — só que treinou mais modelos.

**A LLM nunca vê dado bruto nem chama modelo.** Recebe um payload JSON
fechado, com todos os números já calculados em Python. É o que torna
cada afirmação auditável — e a Etapa 7 verifica isso programaticamente,
número por número, contra o payload que gerou o texto.

## O que cada etapa produziu

| Etapa | Entrega | Onde |
|---|---|---|
| 1 | Codificação dos genes, 3 espaços de busca | `src/ga/search_space.py` |
| 2 | Seleção (torneio/roleta), cruzamento (uniforme/1-ponto/BLX), mutação, fitness | `src/ga/operators.py`, `fitness.py` |
| 3 | 9 execuções do GA, log por geração | `reports/fase2/ga_*.csv`, `logs/ga_events.jsonl` |
| 4 | Tabela comparativa, análise de generalização, experimento corretivo, 7 figuras | `reports/fase2/comparison.md`, `figures/` |
| 5 | Logging estruturado JSONL, monitor de previsões com detecção de drift | `src/monitoring.py` |
| 6 | Explicações por caso e da otimização, 3 versões de prompt documentadas | `reports/fase2/llm_explanations.md` |
| 7 | Avaliação em 2 camadas (auditoria determinística + LLM como juiz) | `reports/fase2/llm_quality.md` |
| 8 | 201 testes (unitários, propriedades, integração ponta a ponta) | `tests/` |
| 9 | Relatório técnico | `reports/fase2/RELATORIO_TECNICO_FASE2.md` |
| 10 | Roteiro do vídeo | `reports/fase2/roteiro_video.md` |
| 11 | Terraform, Dockerfile, handler Lambda (não aplicado) | `infra/`, `Dockerfile.fase2` |

## Artefatos gerados

```
reports/fase2/
├── entrega_final_fase2.docx      # documento de entrega (abrir no Word)
├── build_entrega_docx.py         # gera o .docx acima; é a fonte dele
├── RELATORIO_TECNICO_FASE2.md    # leia primeiro
├── comparison.md / .csv           # original vs. otimizado
├── comparison_detail.json         # matrizes de confusão, hiperparâmetros
├── generalizacao.csv              # CV vs. teste nos 9 vencedores
├── generalizacao_correlacao.csv   # Spearman entre as duas coisas
├── comparison_robusto.csv         # experimento corretivo: as duas réguas de CV
├── baseline_vs_fase1.csv          # confere o baseline refeito
├── ga_history.csv                 # fitness por geração (convergência)
├── ga_best.csv / ga_experiments.json / ga_winners.json
├── ga_*_robusto.*                 # mesmos artefatos, rodada com CV 5x3
├── llm_explanations.md            # explicações geradas
├── llm_payloads.json              # a entrada exata de cada explicação
├── llm_quality.md / .csv          # avaliação da Etapa 7
├── roteiro_video.md
└── figures/
    ├── 30_ga_convergencia.png
    ├── 31_ga_melhor_vs_medio.png
    ├── 32_ga_diversidade.png
    ├── 33_comparativo_metricas.png
    ├── 34_matrizes_confusao.png
    ├── 35_cv_vs_teste.png          # a evidência do sobreajuste de seleção
    └── 36_experimento_robusto.png  # a evidência cruzada (duas réguas)

logs/
├── ga_experiments.log             # log legível
├── ga_events.jsonl                # log estruturado (uma linha por evento)
├── llm_events.jsonl               # telemetria de cada chamada à LLM
└── ...
```

## Aviso

Sistema de **apoio** à decisão clínica, construído como exercício
acadêmico. Não substitui avaliação de profissional de saúde, e o
conjunto de teste (114 amostras) é pequeno demais para qualquer
conclusão clínica — cada caso vale ~0,9 ponto percentual nas métricas.
