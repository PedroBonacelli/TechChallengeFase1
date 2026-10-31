# Relatório Técnico — Fase 2

**Tech Challenge IADT · Projeto 1 — Otimização de Modelos de Diagnóstico
com Algoritmos Genéticos e Integração com LLM**

Repositório: `TechChallengeFase1`, branch `tehc-challenge-fase-2`
Dataset: Breast Cancer Wisconsin (Diagnostic) — 455 amostras de treino
(170 malignas, 37,4%), 114 de teste (42 malignas, 36,8%), 22 features
após a poda de multicolinearidade da Fase 1.

---

## Sumário executivo

O algoritmo genético foi implementado do zero, testado com 201 testes
automatizados, e **funciona**: melhorou a função objetivo nos três
modelos, com ganhos de +1,3, +4,3 e +1,7 pontos percentuais no fitness de
validação cruzada.

**Esse ganho não transferiu para o conjunto de teste.** O recall do
maligno ficou idêntico nos três modelos e a precisão caiu — a Random
Forest passou de 0 para 4 falsos positivos, a Árvore de Decisão de 4 para
12.

A seção 5 mede por que isso aconteceu, em vez de atribuir a azar: a
**correlação de Spearman entre o fitness de validação cruzada e o F1 de
teste é negativa** (−1,00 na Árvore, −0,63 na Floresta). O critério que o
GA otimizou não é um bom preditor do que se queria melhorar.

A seção 6 executa a correção — validação cruzada repetida (5 × 3) em vez
de 5 folds fixos — e mede o efeito. O resultado é a evidência mais forte
do relatório: **cada conjunto de hiperparâmetros é superior apenas na
régua contra a qual foi otimizado, em 6 de 6 comparações.** O ganho de
4,25 pontos percentuais da Árvore de Decisão **encolhe para 0,61** quando
medido num estimador menos ruidoso — 86% dele era ruído de uma partição
específica. Na Random Forest, 90%.

O resultado negativo é o achado central desta fase e está reportado como
tal. Otimizar um estimador de desempenho não é a mesma coisa que melhorar
o desempenho — e a diferença entre as duas coisas é mensurável.

---

## 1. Implementação do algoritmo genético

### 1.1 Por que sem biblioteca

O GA foi escrito à mão (`src/ga/`), sem DEAP, PyGAD ou similar. O
enunciado pede a implementação dos operadores, e escrevendo cada um
isoladamente eles ficam testáveis: 201 testes cobrem seleção, cruzamento,
mutação, fitness, motor e integração.

### 1.2 Codificação dos genes (Etapa 1)

**Codificação real normalizada ("random-key").** Cada cromossomo é um
vetor de floats em `[0,1]`, um por hiperparâmetro; um decodificador
traduz cada posição pro domínio real conforme o tipo do gene. O espaço de
busca é misto — há categórico, inteiro e contínuo — mas o **genótipo é
homogêneo**.

Três razões:

1. **Cruzamento genérico.** Trocar posições entre dois vetores em `[0,1]`
   é sempre válido; não é preciso um operador por tipo de gene.
2. **Escala logarítmica de graça.** O `C` da Logística varia de 1e−3 a
   1e2. Amostrar uniformemente em `[0,1]` no expoente dá peso igual a
   cada ordem de magnitude; amostrar linear em `[0.001, 100]` colocaria
   99% da massa acima de 1.
3. **Limites nunca violados.** Basta manter o vetor em `[0,1]`, o que o
   clipping da mutação garante.

**A mutação, mesmo assim, é sensível ao tipo.** Gene contínuo ou inteiro
recebe perturbação gaussiana (busca local); gene **categórico é
re-sorteado uniformemente**. O motivo é concreto: com três categorias
cada faixa tem 0,33 de largura, e um ruído de desvio 0,15 quase nunca
atravessa a fronteira — a mutação viraria um no-op silencioso justamente
nos genes mais discretos do espaço. Há teste dedicado
(`test_mutacao_de_gene_categorico_troca_de_categoria`).

**Espaços de busca:**

| Modelo | Genes | Tamanho do espaço |
|---|---|---|
| Regressão Logística | 6 | 450 combinações discretas × 2 eixos contínuos |
| Árvore de Decisão | 7 | 198.360 discretas × 1 eixo contínuo |
| Random Forest | 8 | **11.901.600 combinações** |

O espaço da Random Forest sozinho descarta busca exaustiva — é o que
justifica a busca heurística.

**Operador de reparo.** Algumas combinações são inválidas no scikit-learn
(`solver='lbfgs'` não aceita `penalty='l1'`; `elasticnet` só existe no
`saga`). Em vez de descartar o indivíduo — o que jogaria material
genético fora e distorceria a seleção — um operador de reparo conserta o
fenótipo após a decodificação. É a abordagem padrão de tratamento de
restrição em GA quando o conserto é barato e óbvio. O teste
`test_todo_fenotipo_treina` sorteia 25 cromossomos por espaço e **treina
cada um de verdade**, garantindo que nenhuma combinação inválida escapa.

### 1.3 Operadores genéticos (Etapa 2)

| Operador | Variantes implementadas |
|---|---|
| Seleção | torneio (tamanho configurável), roleta |
| Cruzamento | uniforme, 1 ponto, BLX-α |
| Mutação | gaussiana + re-sorteio categórico, taxa e σ configuráveis |
| Elitismo | N melhores copiados direto |

**Por que o torneio é o padrão.** As métricas dos modelos ficam todas
comprimidas entre 0,90 e 0,99. Na roleta, a probabilidade é proporcional
ao fitness, então um indivíduo ruim (0,90) tem quase a mesma chance de um
ótimo (0,98) — a pressão seletiva quase desaparece. O torneio olha só a
**ordem**, então continua separando bem mesmo com fitness comprimido.

**Por que o cruzamento uniforme é o padrão.** A ordem dos genes no
cromossomo é arbitrária (`max_depth` ao lado de `min_samples_split` por
acaso, não por afinidade). Não existe bloco construtivo contíguo a
preservar, então o viés posicional do corte em 1 ponto não compra nada.

**Por que o BLX-α existe mesmo assim.** O cruzamento uniforme só
recombina valores que já estão na população. O BLX cria valores
intermediários e um pouco além dos pais — é o que permite refinar o `C`
da Logística continuamente em vez de ficar preso nos valores sorteados na
população inicial.

**Elitismo garante monotonicidade.** Com elitismo ≥ 1, o melhor fitness
nunca cai de uma geração para a outra. Há teste de propriedade para isso
(`test_elitismo_garante_monotonicidade`), e um teste de contraste sem
elitismo que documenta a diferença.

### 1.4 Função fitness (Etapa 2)

```
fitness = 0,60 × recall_maligno + 0,25 × F1_maligno + 0,15 × acurácia
```

medida por **validação cruzada estratificada de 5 folds sobre o conjunto
de TREINO**.

**Os pesos.** O recall do maligno domina porque o custo dos dois erros é
assimétrico, como já estabelecido no relatório da Fase 1: um falso
negativo é um câncer que segue sem tratamento; um falso positivo é uma
biópsia desnecessária. A função objetivo não pode tratá-los igual. O F1
entra para impedir a solução degenerada — um modelo que chamasse tudo de
maligno teria recall 1,0 e fitness máximo se o recall fosse o único
termo; com o F1 no meio, a precisão despencando puxa o fitness para
baixo. A acurácia entra com peso pequeno, só para desempatar.

**Onde se mede é mais importante que o que se mede.** O fitness nunca
toca o conjunto de teste. Se tocasse, o GA otimizaria diretamente contra
ele, o teste teria virado conjunto de treino de fato, e todo número da
seção 4 seria vazamento em vez de desempenho. Estratificado porque a
proporção maligno/benigno precisa se manter em cada fold — num fold
desbalanceado o recall mediria ruído de amostragem.

**Falha vira fitness 0, não exceção.** Uma combinação que o scikit-learn
rejeite recebe fitness 0 e a busca continua; a seleção descarta o
indivíduo naturalmente.

**Cache de fitness.** Elitismo e convergência fazem o mesmo indivíduo
reaparecer em muitas gerações, e o decodificador mapeia genótipos
distintos no mesmo fenótipo (dois valores de `[0,1]` que caem na mesma
faixa categórica). O cache por fenótipo economizou entre 5% e 84% das
avaliações conforme a configuração — ver a tabela da seção 3.

---

## 2. Os três experimentos (Etapa 3)

**Orçamento idêntico nos três: 600 avaliações de fitness** (população ×
gerações). Isso é deliberado — se um experimento tivesse o dobro do
orçamento, um resultado melhor não diria nada sobre a *estratégia*,
apenas que ele treinou mais modelos. Com orçamento igual, a comparação
mede o que se quer medir: dado o mesmo custo, exploração rende mais ou
menos que intensificação?

| | `equilibrado` | `exploratorio` | `intensivo` |
|---|---|---|---|
| População | 30 | 50 | 15 |
| Gerações | 20 | 12 | 40 |
| Taxa de cruzamento | 0,90 | 0,70 | 0,95 |
| Operador de cruzamento | uniforme | BLX-α (0,5) | BLX-α (0,3) |
| Taxa de mutação | 0,10 | **0,30** | **0,05** |
| σ da mutação | 0,15 | 0,25 | 0,08 |
| Seleção | torneio (3) | **roleta** | torneio (**5**) |
| Elitismo | 2 | 1 | 4 |
| Semente | 42 | 43 | 44 |

- **`equilibrado`** — referência; parâmetros de meio de tabela.
- **`exploratorio`** — população grande, mutação 3× maior, roleta
  (pressão fraca) e BLX. Aposta em cobrir o espaço; o risco é não sobrar
  geração para refinar.
- **`intensivo`** — população pequena com o dobro de gerações, torneio de
  5, mutação baixa, elitismo forte. Converge rápido; o risco é travar num
  ótimo local.

Tempo total das 9 execuções: **27,6 minutos**.

---

## 3. Resultados da otimização

### 3.1 Fitness de validação cruzada

| Modelo | Experimento | Fitness CV | Recall mal. (CV) | Gerações | Tempo | Aval. reais | Cache |
|---|---|---|---|---|---|---|---|
| Reg. Logística | equilibrado | 0,9717 | 0,9706 | 20 | 40,0 s | 211 | 389 |
| Reg. Logística | exploratorio | 0,9707 | 0,9706 | 12 | 201,2 s | 542 | 58 |
| Reg. Logística | **intensivo** | **0,9732** | **0,9765** | 40 | 19,1 s | 119 | 481 |
| Árvore de Decisão | **equilibrado** | **0,9331** | **0,9529** | 20 | 12,9 s | 295 | 305 |
| Árvore de Decisão | exploratorio | 0,9250 | 0,9294 | 12 | 24,1 s | 571 | 29 |
| Árvore de Decisão | intensivo | 0,9269 | 0,9412 | 40 | 19,6 s | 338 | 262 |
| Random Forest | **equilibrado** | **0,9533** | **0,9588** | 20 | 304,2 s | 269 | 331 |
| Random Forest | exploratorio | 0,9533 | 0,9588 | 12 | 802,0 s | 564 | 36 |
| Random Forest | intensivo | 0,9491 | 0,9471 | 40 | 233,2 s | 95 | 505 |

Em negrito, o vencedor de cada modelo — escolhido **pelo fitness de CV**,
nunca pelo desempenho no teste.

### 3.2 O que a comparação entre estratégias mostrou

**A exploração pura nunca venceu sozinha.** O `exploratorio` ficou em
último na Regressão Logística (0,9707) e na Árvore (0,9250), e **empatou
em primeiro na Random Forest** (0,9533, idêntico ao `equilibrado`) — mas
gastando 802 s contra 304 s, 2,6× o tempo para chegar ao mesmo valor.

A razão aparece na coluna de cache: ele fez **542-571 avaliações reais**
contra 95-295 das outras configurações, ou seja, gastou quase todo o
orçamento visitando pontos novos. Com apenas 12 gerações, não sobrou
ciclo para refinar nenhum deles. Mutação alta mais roleta mantém a
população espalhada, e espalhada ela fica — a busca se aproxima de uma
amostragem aleatória com memória.

O empate na Random Forest é coerente com isso: é o maior espaço de busca
dos três (11,9 milhões de combinações), e é justamente onde cobrir
terreno tem mais valor. A exploração pagou o suficiente para alcançar o
`equilibrado`, não para superá-lo.

**A intensificação ganhou onde o espaço é pequeno e perdeu onde é
grande.** O `intensivo` venceu na Regressão Logística (6 genes, espaço
pequeno: refinar é a coisa certa a fazer) e foi o pior na Random Forest
(8 genes, 11,9 milhões de combinações: convergir cedo significa não ter
visto quase nada).

**Convergência prematura é visível no gráfico de diversidade.** No
`intensivo` da Random Forest a diversidade (desvio médio por gene) cai de
**0,270 na geração 1 para 0,029 na geração 5** — 89% em quatro gerações —
e chega ao mínimo de 0,006 na geração 21. A população virou cópia do
elite cedo, e as 35 gerações seguintes giraram praticamente em torno do
mesmo ponto.

Isso aparece na coluna de cache como **95 avaliações reais contra 505 do
cache**: 84% do orçamento foi gasto reavaliando indivíduos idênticos. O
cache transformou desperdício computacional em tempo economizado (233 s
contra os 802 s do `exploratorio`), mas não recuperou a busca perdida.

O contraste com o `exploratorio` é direto: a diversidade dele fica em
0,25-0,30 do começo ao fim, nos três modelos. Nenhum dos dois extremos
funcionou — um nunca converge, o outro converge cedo demais — e o
`equilibrado`, que fica entre 0,09 e 0,14, venceu em dois dos três
modelos.

**Figuras:** `figures/30_ga_convergencia.png`,
`figures/31_ga_melhor_vs_medio.png`, `figures/32_ga_diversidade.png`.

### 3.3 Hiperparâmetros encontrados

| Reg. Logística | Fase 1 | GA (`intensivo`) |
|---|---|---|
| `C` | 1,0 | **1,039** |
| `penalty` | `l2` | **`l1`** |
| `solver` | `lbfgs` | **`liblinear`** |
| `max_iter` | 5000 | **4000** |
| `class_weight` | `balanced` | `balanced` |

| Árvore de Decisão | Fase 1 | GA (`equilibrado`) |
|---|---|---|
| `criterion` | `gini` | `gini` |
| `max_depth` | 5 | **19** |
| `min_samples_split` | 2 | **9** |
| `min_samples_leaf` | 1 | **3** |
| `max_features` | `None` | **`log2`** |
| `ccp_alpha` | 0,0 | **0,02186** |
| `class_weight` | `balanced` | `balanced` |

| Random Forest | Fase 1 | GA (`equilibrado`) |
|---|---|---|
| `n_estimators` | 300 | **150** |
| `max_depth` | 8 | **19** |
| `min_samples_split` | 2 | **3** |
| `min_samples_leaf` | 1 | **6** |
| `max_features` | `sqrt` | **0,5** |
| `class_weight` | `balanced` | `balanced` |
| `bootstrap` | `True` | `True` |

Dois pontos valem comentário:

- **O GA manteve `class_weight='balanced'` nos três**, embora pudesse
  tê-lo desligado (`None` estava no espaço de busca). É uma confirmação
  independente da escolha manual da Fase 1 — com recall no peso 0,60, o
  balanceamento de classe compensa.
- **A Random Forest ficou com metade das árvores** (150 contra 300) sem
  perder fitness de CV. Treina e serve em metade do tempo.

---

## 4. Comparativo: original vs. otimizado (Etapa 4)

### 4.1 Nota metodológica: o baseline foi retreinado

Os arquivos `models/*.joblib` da Fase 1 foram serializados com
**scikit-learn 1.8.0**, e o ambiente da Fase 2 roda **1.4.2**. Carregá-los
levanta `InconsistentVersionWarning` e, na Regressão Logística,
`AttributeError` em `predict_proba` (o atributo `multi_class` deixou de
existir entre as versões).

A solução foi retreinar o baseline a partir dos hiperparâmetros exatos da
Fase 1, documentados em `BASELINE_PARAMS`. Isso é melhor do que um
contorno: original e otimizado passam a ser treinados pela **mesma
biblioteca**, no **mesmo caminho de código**, sobre os **mesmos dados**.
Se o baseline viesse de um pickle de outra versão, qualquer diferença nas
métricas poderia ser da versão do scikit-learn e não do GA — e não haveria
como separar as duas coisas.

**Validação:** o baseline refeito reproduz as métricas publicadas na Fase
1 com diferença máxima de **0,0000** (`baseline_vs_fase1.csv`).

### 4.2 Fitness de CV — o que o GA otimizou

| Modelo | Fase 1 | Otimizado | Ganho |
|---|---|---|---|
| Reg. Logística | 0,9604 | **0,9732** | **+0,0128** |
| Árvore de Decisão | 0,8906 | **0,9331** | **+0,0425** |
| Random Forest | 0,9361 | **0,9533** | **+0,0172** |

O GA melhorou a função objetivo nos três modelos.

### 4.3 Métricas no conjunto de teste

114 amostras, 42 malignas. O GA nunca viu estes dados.

| Modelo | Recall mal. | F1 mal. | Acurácia | ROC AUC | Falsos neg. | Falsos pos. |
|---|---|---|---|---|---|---|
| Reg. Logística | 0,9762 → **0,9762** | 0,9535 → 0,9535 | 0,9649 → 0,9649 | 0,9947 → **0,9970** | 1 → 1 | 3 → 3 |
| Árvore de Decisão | 0,9286 → **0,9286** | 0,9176 → **0,8387** | 0,9386 → **0,8684** | 0,9289 → **0,8889** | 3 → 3 | 4 → **12** |
| Random Forest | 0,9286 → **0,9286** | 0,9630 → **0,9176** | 0,9737 → **0,9386** | 0,9957 → 0,9934 | 3 → 3 | 0 → **4** |

**O recall do maligno — a métrica que domina o fitness com peso 0,60 —
não mudou em nenhum dos três modelos.** O custo transferiu integralmente:
a Random Forest ganhou 4 falsos positivos onde tinha zero, e a Árvore
triplicou os seus.

O único ganho real é marginal: o ROC AUC da Logística subiu 0,0023, o que
significa que o ordenamento das probabilidades melhorou um pouco sem
mudar nenhuma decisão no limiar de 0,5.

**Figuras:** `figures/33_comparativo_metricas.png`,
`figures/34_matrizes_confusao.png`.

---

## 5. Por que não transferiu — a investigação

Atribuir isso a azar seria a saída fácil. A análise abaixo mede a causa.

### 5.1 O fitness de CV prevê o desempenho no teste?

`src/ga/compare.py::analisar_generalizacao` avalia no teste **todos os 9
indivíduos vencedores** (3 modelos × 3 experimentos) mais o baseline, e
põe fitness de CV e métricas de teste lado a lado.

> Isto é **diagnóstico, não critério de seleção**. O vencedor de cada
> modelo continua escolhido pelo fitness de CV. Escolher pelo teste seria
> exatamente o vazamento que esta análise serve para tornar visível.

Correlação de **Spearman** (de ordem — a pergunta é se o *ranqueamento*
por fitness reproduz o ranqueamento por desempenho):

| Modelo | vs. F1 de teste | vs. acurácia de teste | vs. recall de teste |
|---|---|---|---|
| Reg. Logística | — (constante) | — (constante) | — (constante) |
| Árvore de Decisão | **−1,000** | **−1,000** | 0,000 |
| Random Forest | **−0,632** | **−0,833** | 0,500 |

**A correlação é negativa.** Quanto melhor o fitness de validação
cruzada, *pior* o desempenho no teste. Na Árvore de Decisão é −1,00: a
inversão é perfeita, os quatro pontos ordenam-se exatamente ao contrário.

Na Regressão Logística as três configurações do GA produzem **métricas de
teste idênticas** (0,9762 / 0,9535 / 0,9649), apesar de fitness de CV
diferentes (0,9707 a 0,9732). A fronteira de decisão é estável nessa
faixa de `C` e `penalty`; a variação de fitness é ruído puro.

**Figura:** `figures/35_cv_vs_teste.png` — o baseline da Fase 1 fica no
canto superior esquerdo (fitness baixo, teste alto) e todos os pontos do
GA no inferior direito. A nuvem desce da esquerda para a direita.

### 5.2 As duas causas

**(a) Sobreajuste do critério de seleção.** 600 avaliações num espaço de
até 8 dimensões, sobre 455 amostras divididas em **5 folds fixos**. O GA
é um otimizador eficiente — e o que ele otimizou foi o desempenho
*naquela partição específica*, incluindo o ruído dela. Uma configuração
que acerte por acaso dois casos difíceis do fold 3 ganha fitness e é
selecionada, sem que isso signifique nada fora daqueles folds.

A assinatura disso é a Árvore de Decisão: o GA levou `max_depth` de 5 para
19 e `max_features` de `None` para `log2`. Uma árvore de profundidade 19
sobre 455 amostras decora o treino; os folds de validação da CV são
pequenos (91 amostras) e o `ccp_alpha` de 0,022 segurou o suficiente para
parecer bom lá — mas no teste a precisão caiu de 0,907 para 0,765.

**(b) Resolução do conjunto de teste.** São **42 casos malignos**. O
recall só se move em passos de 1/42 = **2,4 pontos percentuais**. Um ganho
real de 1 p.p. é literalmente invisível neste teste. Isso não desculpa (a)
— a precisão piorou de forma clara e mensurável — mas explica por que o
recall ficou parado nos três modelos em vez de oscilar.

### 5.3 O caso que ilustra o problema: um empate exato

A configuração `random_forest / exploratorio` obteve **recall de teste
0,9762 — 1 falso negativo em vez de 3** — o melhor recall de qualquer
configuração da Floresta, exatamente o que o projeto mais valoriza.

Ela **não foi selecionada**. E o motivo é mais revelador que um simples
"perdeu por pouco": o fitness de CV das duas é **idêntico até a última
casa de ponto flutuante**.

| | `equilibrado` (escolhido) | `exploratorio` (descartado) |
|---|---|---|
| Fitness de CV | `0.9533482273924908` | `0.9533482273924908` |
| `n_estimators` | 150 | 110 |
| `max_depth` | 19 | 20 |
| `min_samples_split` | 3 | 24 |
| `min_samples_leaf` | 6 | 4 |
| `class_weight` | `balanced` | `balanced_subsample` |
| **Falsos negativos no teste** | **3** | **1** |

São dois modelos substancialmente diferentes — um com 150 árvores e
divisão a partir de 3 amostras, outro com 110 árvores e divisão a partir
de 24 — e o estimador de desempenho não consegue distingui-los **nem na
16ª casa decimal**. O desempate acabou sendo a ordem de execução.

**Por que o empate é exato e não aproximado.** O fitness é a média, sobre
5 folds, de métricas que são razões de contagens inteiras pequenas (cada
fold de validação tem 91 amostras, ~34 malignas). O conjunto de valores
que o fitness pode assumir é discreto e grosso. Dois modelos que
produzam a mesma matriz de confusão em cada um dos 5 folds recebem
exatamente a mesma nota — e com 455 amostras isso não é raro, é comum.

Ou seja: além de ser um estimador *enviesado* pelo ruído dos folds
(seção 5.2), o fitness tem **resolução insuficiente** para ordenar
candidatos próximos. O GA está escolhendo entre opções que sua função
objetivo declara equivalentes.

E mostra por que a tentação de "escolher pelo teste" tem que ser
resistida: se eu trocasse o vencedor agora que sei o resultado, o número
reportado deixaria de ser uma estimativa de desempenho e passaria a ser
um recorde de busca no conjunto de avaliação. A correção legítima é
melhorar o estimador — é o que a seção 6 faz.

---

## 6. Experimento corretivo: validação cruzada repetida

A causa (a) tem uma correção direta: se o problema é o GA explorar o ruído
de **uma** partição, usar **várias** partições e tirar a média remove boa
parte do ruído disponível para explorar.

`FitnessConfig(cv_repeats=3)` troca `StratifiedKFold(5)` por
`RepeatedStratifiedKFold(5 × 3)` — três particionamentos independentes,
15 treinos por avaliação em vez de 5. Executado com a configuração
`equilibrado` nos três modelos (só ela: o objetivo é isolar o efeito do
**protocolo de avaliação**, não comparar estratégias de GA de novo):

```bash
python -m src.ga.run_experiments --fitness-robusto
```

### 6.1 A evidência cruzada

Medindo **os dois conjuntos de vencedores nas duas réguas**:

| Modelo | Configuração | Régua CV 5-fold | Régua CV 5×3 |
|---|---|---|---|
| Reg. Logística | baseline Fase 1 | 0,9604 | 0,9563 |
| | GA / CV 5-fold | **0,9732** ◄ | 0,9662 |
| | GA / CV 5×3 | 0,9666 | **0,9679** ◄ |
| Árvore de Decisão | baseline Fase 1 | 0,8906 | 0,9056 |
| | GA / CV 5-fold | **0,9331** ◄ | 0,9117 |
| | GA / CV 5×3 | 0,9288 | **0,9207** ◄ |
| Random Forest | baseline Fase 1 | 0,9361 | 0,9428 |
| | GA / CV 5-fold | **0,9533** ◄ | 0,9446 |
| | GA / CV 5×3 | 0,9464 | **0,9535** ◄ |

**Cada vencedor é superior apenas na régua contra a qual foi otimizado —
seis de seis casos, sem exceção.** Essa é a definição operacional de
sobreajuste do critério de seleção: a vantagem encontrada é uma
propriedade do estimador, não do modelo.

**Quanto do "ganho" era ruído.** Medindo o ganho do vencedor da CV
5-fold sobre o baseline nas duas réguas:

| Modelo | Ganho na régua 5-fold | Ganho na régua 5×3 | Evaporou |
|---|---|---|---|
| Reg. Logística | +1,28 p.p. | +0,99 p.p. | 23% |
| Árvore de Decisão | **+4,25 p.p.** | **+0,61 p.p.** | **86%** |
| Random Forest | **+1,72 p.p.** | **+0,18 p.p.** | **90%** |

O ganho de 4,25 pontos da Árvore — o maior de toda a fase, destacado na
seção 4.2 — vira 0,61 ponto quando medido num estimador menos ruidoso.
**Mais de quatro quintos dele eram ruído de uma partição específica.**

Medido honestamente (vencedor da 5×3 na régua 5×3), o ganho real do GA é
de **+1,2, +1,5 e +1,1 pontos percentuais** — consistente entre os três
modelos, e bem menor que a primeira leitura sugeria.

**Figura:** `figures/36_experimento_robusto.png`.

### 6.2 E no conjunto de teste?

| Modelo | Versão | Recall mal. | Precisão mal. | F1 mal. | FN | FP |
|---|---|---|---|---|---|---|
| Reg. Logística | baseline | 0,9762 | 0,9318 | **0,9535** | 1 | 3 |
| | GA / 5-fold | 0,9762 | 0,9318 | **0,9535** | 1 | 3 |
| | GA / 5×3 | 0,9762 | 0,9111 | 0,9425 | 1 | 4 |
| Árvore de Decisão | baseline | 0,9286 | 0,9070 | **0,9176** | 3 | 4 |
| | GA / 5-fold | 0,9286 | 0,7647 | 0,8387 | 3 | **12** |
| | GA / 5×3 | 0,9286 | 0,7959 | 0,8571 | 3 | 10 |
| Random Forest | baseline | 0,9286 | 1,0000 | **0,9630** | 3 | **0** |
| | GA / 5-fold | 0,9286 | 0,9070 | 0,9176 | 3 | 4 |
| | GA / 5×3 | 0,9048 | 0,9268 | 0,9157 | **4** | 3 |

A correção **funcionou em parte, e só onde o problema era maior**:

- **Árvore de Decisão — melhorou.** Era o caso mais grave (Spearman
  −1,00, 12 falsos positivos). A CV repetida reduziu para 10 e subiu o F1
  de 0,8387 para 0,8571, fechando cerca de um quarto da distância até o
  baseline. Ainda pior que a Fase 1.
- **Regressão Logística — piorou marginalmente.** Um falso positivo a
  mais. O espaço é pequeno e a fronteira é estável; mudar o estimador
  levou a busca a um ponto vizinho ligeiramente pior, dentro do ruído.
- **Random Forest — trocou um erro por outro, na direção errada.** Um
  falso positivo a menos (4 → 3), mas **um falso negativo a mais
  (3 → 4)** — e é o falso negativo que o projeto inteiro prioriza. O F1
  ficou praticamente igual (0,9176 → 0,9157).

**Conclusão do experimento.** A CV repetida é a correção certa para a
causa (a) e isso é demonstrável: ela elimina 86-90% do ganho ilusório e
produz um estimador cujos vencedores não se desfazem na troca de
partição. Mas **não bastou para superar o baseline no teste**, porque a
causa (b) permanece: os modelos da Fase 1 já operavam perto do teto
(ROC AUC de 0,995) e o conjunto de teste, com 42 casos malignos, não tem
resolução para registrar um ganho de ~1 ponto percentual.

A lição prática é que **medir melhor não substitui ter o que medir**. Com
455 amostras de treino e 114 de teste, a incerteza da estimativa é da
mesma ordem de grandeza do ganho que o GA consegue extrair — e nesse
regime, ajuste de hiperparâmetros tem retorno baixo em comparação a
conseguir mais dados.

---

## 7. Integração com LLM (Etapa 6)

### 7.1 Escolha e arquitetura

**LLM:** Claude (Anthropic), via o SDK oficial `anthropic`. Critérios:
API gerenciada (sem GPU nem download de pesos no caminho de uma entrega
acadêmica), retry com backoff exponencial já no SDK, e saída estruturada
nativa usada pelo juiz da Etapa 7.

**O princípio que organiza o módulo: a LLM nunca vê dado bruto nem chama
modelo.** Ela recebe um **payload JSON fechado**, montado em Python, com
todos os números já calculados. Três razões:

1. **Auditabilidade.** Toda afirmação numérica do texto é rastreável a um
   campo do payload, e a Etapa 7 verifica isso programaticamente.
2. **A LLM não precisa fazer aritmética.** Quem calcula é o scikit-learn.
3. **O payload fica salvo em disco** (`llm_payloads.json`), então uma
   explicação suspeita pode ser reconferida meses depois contra a entrada
   exata que a gerou.

**Dois tipos de explicação:**

- **Por caso** — paciente do conjunto de teste, previsão dos 3 modelos
  otimizados, fatores que pesaram, grau de confiança e conduta sugerida.
- **Da otimização** — a tabela da seção 4 traduzida para quem decide
  sobre adoção, com o efeito expresso em **número de pacientes** ("deixa
  de detectar 3 dos 42 casos malignos") em vez de pontos percentuais.

**Atribuição dos fatores.** Para a Regressão Logística uso o SHAP exato
de modelo linear — `φᵢ = coefᵢ × (xᵢ − média_treinoᵢ)` (Lundberg & Lee,
2017) —, a mesma fórmula da Fase 1. Há teste de integração verificando
que a soma das contribuições mais o intercepto reconstrói o log-odds
previsto. As importâncias da Random Forest entram no payload num campo
**separado e nomeado como global**, nunca misturadas com as contribuições
individuais, porque importância de floresta não é atribuição por caso.

**Seleção dos casos.** Não são aleatórios. O sistema prioriza: falso
negativo do modelo de referência, discordância entre os 3 modelos, e
menor confiança — exatamente os casos em que explicação automática é
útil. Um caso com 99,9% de certeza e consenso não precisa de texto.

### 7.2 Modo offline

Sem chave de API, o cliente cai num gerador determinístico por template
que produz o mesmo formato de quatro seções. Não é enfeite: é o que
permite (1) rodar os 201 testes em qualquer máquina e no CI, sem segredo
configurado e sem gastar token, e (2) ter um piso de funcionamento se a
API cair no meio de uma demonstração. O texto é claramente pior e se
identifica como tal (`provider="offline"`).

### 7.3 Engenharia de prompt — as três versões

O histórico está versionado em `src/llm/prompts.py`, com o que mudou e
por quê.

**v1** — genérico ("explique de forma clara e objetiva"). Falhou em três
frentes: escrevia *"o exame confirma malignidade"* (linguagem de laudo
fechado); citava métricas que não estavam no payload; e usava "recall" e
"F1" sem traduzir.

**v2** — acrescentou as três regras principais (não diagnosticar, não
inventar número, ser conciso) e o domínio. Corrigiu a linguagem
conclusiva e a invenção de métricas, mas: produzia texto sem estrutura;
nos casos de discordância entre os 3 modelos **escolhia um "vencedor" e
omitia o conflito** — apagando a informação mais útil para o médico; e
interpretava `concavity error` como sintoma do paciente em vez de medida
de lâmina.

**v3 (em uso)** — acrescenta contexto do dataset e natureza das features;
proibição de tratar morfometria como sintoma; a assimetria de custo entre
falso negativo e falso positivo; obrigação de **destacar discordância**; e
um formato fixo de quatro seções nomeadas.

O formato fixo foi o que mais mudou a utilidade: com seções nomeadas, o
médico lê a conclusão e a conduta sem percorrer o texto inteiro. A regra
de discordância trouxe de volta o sinal que a v2 apagava.

A seção 8 mede as três versões em vez de afirmar que a v3 é melhor.

---

## 8. Avaliação da qualidade das interpretações (Etapa 7)

"A explicação ficou boa?" não se responde lendo três exemplos. A
avaliação tem **duas camadas que medem coisas diferentes**.

### 8.1 Camada 1 — verificações determinísticas (código, sem LLM)

A mais importante é a **auditoria numérica**: extrai todo número do texto
e confirma que ele existe no payload. Alucinação numérica em contexto
clínico é o modo de falha mais perigoso e o mais fácil de passar batido
numa leitura casual.

A tolerância vem da **precisão com que o número foi escrito**, não de uma
fração relativa: "97,6" autoriza ±0,05; "98", ±0,5. A pergunta é "este
texto é um arredondamento correto de algum valor autorizado?". Tolerância
relativa seria errada de um jeito perigoso — com 0,6% de folga, "99,9%"
passaria como arredondamento de "100%", e as duas afirmações são
clinicamente diferentes.

Também verifica: linguagem conclusiva proibida (regex sobre texto
normalizado sem acento), jargão de ML não traduzido, as quatro seções, o
limite de palavras, e a presença da ressalva de que a decisão é clínica.

Nota 0-5, descontando 2,0 por número inventado e 2,0 por linguagem
conclusiva (falhas de **segurança**), e 0,5 por jargão, seção ausente ou
ressalva faltando (falhas de **usabilidade**).

### 8.2 Camada 2 — LLM como juiz

Um segundo prompt, com system próprio e instrução explícita de ceticismo,
dá nota 1-5 em cinco critérios: fidelidade numérica, ausência de
diagnóstico, clareza clínica, acionabilidade e calibração da incerteza.
Cobre o que regex não alcança — se o texto é compreensível, se a conduta
é concreta, se a confiança declarada combina com a probabilidade.

### 8.3 Quando as duas discordam

**A Camada 1 prevalece.** Ela compara contra o payload; o juiz compara
contra a própria leitura. A coluna `divergencia_fidelidade` registra os
casos em que o juiz deu nota ≥ 4 em fidelidade numérica enquanto a
auditoria encontrou número sem respaldo — essa coluna mede a
confiabilidade do juiz, não da explicação.

### 8.4 A auditoria pegou defeitos reais — inclusive meus

A Camada 1 reprovou **o meu próprio gerador por template**:

1. A interpretação da otimização interpolava o campo
   `recomendacao_por_recall` do payload, injetando a palavra "recall" —
   jargão que o prompt manda traduzir — num texto que deveria estar em
   linguagem clínica.
2. O mesmo texto não registrava que a decisão é do profissional.
3. No ramo de **divergência entre modelos** — o mais sensível dos três —
   a conduta sugerida omitia a ressalva clínica.

E encontrou um **falso positivo na própria auditoria**: em "−1,17" a
regex casava "17" isoladamente, porque a vírgula decimal não estava na
exclusão à esquerda, e acusava invenção de um número que estava no
payload.

Os quatro foram corrigidos e viraram **teste de regressão**
(`test_auditoria_nao_parte_numero_negativo_ao_meio`,
`test_otimizacao_offline_nao_vaza_jargao_do_payload`, e outros dois).
Isso é o argumento mais forte a favor da Camada 1: um revisor humano
lendo os textos não teria pego nenhum dos quatro.

Resultados completos, com exemplos de boas e más respostas e a comparação
medida entre os prompts v1/v2/v3: `reports/fase2/llm_quality.md`.

---

## 9. Escalabilidade e monitoramento (Etapa 5)

### 9.1 Logging em duas saídas

- **Legível** (`logs/*.log`) — acompanhar execução e depurar.
- **Estruturado** (`logs/*.jsonl`) — uma linha JSON por evento, feito
  para ser consumido por código. É o que permite reconstruir a curva de
  convergência de um experimento rodado semanas antes sem reexecutar
  nada, e é o formato que qualquer coletor de observabilidade ingere
  direto.

Eventos: `experiment_start`, `generation`, `experiment_end`,
`prediction`, `prediction_snapshot`, `llm_call`. Escrita com `flush` a
cada linha — se a otimização morrer na geração 40 de 50, as 40 primeiras
continuam no arquivo. Leitura tolerante a linha corrompida.

### 9.2 Monitoramento das previsões, não só do treino

O `PredictionMonitor` mantém janela deslizante das previsões servidas e
expõe dois indicadores de **drift**:

- **Taxa de maligno prevista** — desvio em relação à prevalência do
  treino (~37,4%) é o sinal mais barato de mudança na distribuição de
  entrada.
- **Confiança média** — cair indica que os casos estão chegando na zona
  de incerteza, mesmo com a taxa de maligno estável.

Previsão com confiança abaixo de 0,65 é marcada no log; em produção é
esse flag que roteia o caso para revisão humana.

### 9.3 Escalabilidade

- **Cache de fitness** — o maior ganho. Economizou até 84% das
  avaliações.
- **Random Forest com `n_jobs=-1`** — cada avaliação usa todos os núcleos.
- **`random_state` fixo** — sem isso o cache mentiria e a seleção estaria
  comparando ruído junto com sinal.
- **Parada antecipada opcional** (`patience`) — desligada nos
  experimentos para manter as curvas de convergência comparáveis.

---

## 10. Arquitetura em nuvem (Etapa 11, opcional)

Terraform completo em `infra/`. **Não foi aplicado** — a entrega é o
código de infraestrutura; aplicar exigiria credenciais e geraria custo
recorrente sem acrescentar nada ao que a fase demonstra. Ressalva
honesta: `terraform validate` não foi executado (o binário não está no
ambiente de desenvolvimento), então erro de sintaxe é possível.

**A decisão central:** otimização e inferência são serviços separados,
porque têm perfis opostos — o GA é lote pesado e raro (tolera
interrupção, roda em **Fargate Spot**); a inferência é leve e frequente
(**Lambda**). O **S3 é a fronteira**: o Batch escreve `models/`, a Lambda
lê `models/`.

**Permissões assimétricas:** a Lambda tem **somente leitura** em
`models/`. Se for comprometida, o atacante não consegue trocar o modelo
de diagnóstico — a pior consequência possível nesta arquitetura. O job do
GA, por sua vez, não acessa a chave da LLM.

**O log estruturado vira métrica sem agente:** os metric filters do
CloudWatch leem o mesmo JSONL que o `monitoring.py` escreve. É a razão de
o log ser JSON e não texto.

**Os alarmes são de duas famílias.** Técnicos (erro, throttle, LLM caindo
no fallback) e **de saúde do modelo** (taxa de maligno acima de 0,55,
fração de previsões incertas acima de 0,25). Um modelo pode estar
tecnicamente perfeito e errando sistematicamente porque a distribuição
mudou; só a segunda família pega isso.

**Custo estimado** (1.000 exames/mês): ~US$ 5/mês de infraestrutura, mais
US$ 25-30 da API da LLM. **A LLM é ~85% do custo** — daí o teto de
concorrência reservada e a métrica de custo por chamada.

Detalhes, diagrama e o que faltaria para uso clínico real:
[`infra/README.md`](../../infra/README.md).

---

## 11. Testes automatizados (Etapa 8)

**201 testes**, todos sem rede.

| Arquivo | Cobre |
|---|---|
| `test_search_space.py` | codificação, decodificação, limites, operador de reparo |
| `test_operators.py` | seleção, cruzamento, mutação, elitismo, diversidade |
| `test_fitness.py` | pesos, métricas, determinismo, cache, tratamento de falha |
| `test_engine.py` | laço do GA, monotonicidade do elite, reprodutibilidade, parada |
| `test_llm.py` | cliente, prompts, gerador offline, auditoria da Etapa 7 |
| `test_integration.py` | pipeline GA → modelo → payload → explicação → auditoria |

Os testes de operador verificam **propriedade**, não saída fixa: "o
torneio devolve o melhor quando o torneio é a população inteira", "a
mutação nunca sai de `[0,1]`", "cruzamento uniforme produz filhos
complementares". Propriedade sobrevive a refatoração; valor fixo não.

Três merecem destaque:

- **`test_todo_fenotipo_treina`** sorteia 25 cromossomos por espaço e
  **treina cada fenótipo de verdade** no scikit-learn. É o único jeito de
  garantir que o decodificador e o reparo cobrem todas as combinações
  inválidas.
- **`test_contribuicoes_lineares_somam_o_log_odds`** verifica que a soma
  das contribuições SHAP mais o intercepto reconstrói o log-odds previsto
  — prova de que a atribuição é exata, não heurística.
- **`test_elitismo_garante_monotonicidade`** afirma que o melhor fitness
  nunca cai entre gerações.

A fixture de dados é construída a partir do dataset embutido no
scikit-learn, não dos CSVs de `data/processed/`. Um teste que depende de
artefato gerado por outra etapa não testa o código — testa se alguém
lembrou de rodar o pipeline na ordem certa.

---

## 12. Desafios enfrentados e soluções

| # | Desafio | Solução |
|---|---|---|
| 1 | **O ganho do GA não transferiu para o teste** | Medido, não suposto: análise de generalização com Spearman nos 9 vencedores (−1,00 / −0,63). Causa identificada como sobreajuste do critério de seleção; corrigida com CV repetida (seção 6). |
| 2 | **Combinações inválidas de hiperparâmetros** (`lbfgs` + `l1`) | Operador de reparo em vez de descarte, que preservaria material genético. Teste que treina 25 fenótipos sorteados por espaço garante a cobertura. |
| 3 | **Fitness comprimido entre 0,90 e 0,99** torna a roleta quase cega | Torneio como padrão (usa ordem, não valor). A roleta ficou implementada e foi ao `exploratorio`, onde se confirmou mais lenta. |
| 4 | **Mutação gaussiana não muda gene categórico** (faixa 0,33 vs. σ 0,15) | Mutação sensível ao tipo: re-sorteio uniforme para categórico. Teste verifica que as três categorias aparecem. |
| 5 | **Custo computacional** — 5.400 avaliações × 5 folds | Cache de fitness por fenótipo (até 84% de economia), `n_jobs=-1` na floresta, `random_state` fixo para o cache ser válido. 27,6 min no total. |
| 6 | **Modelos da Fase 1 ilegíveis** (serializados em sklearn 1.8.0, ambiente em 1.4.2) | Retreino do baseline a partir dos hiperparâmetros documentados — que também torna a comparação mais justa. Validado: diferença 0,0000 contra a Fase 1. `requirements-lock.txt` fixa versões para a imagem de container, para que treino e inferência nunca divirjam. |
| 7 | **Risco de alucinação numérica da LLM** em contexto clínico | Payload JSON fechado (a LLM não calcula nada) + auditoria determinística que confere cada número contra o payload. |
| 8 | **Tolerância da auditoria numérica** — relativa deixaria "99,9%" passar como "100%" | Tolerância derivada da precisão com que o número foi escrito. Encontrado por teste que falhou. |
| 9 | **Falso positivo na própria auditoria** — "−1,17" virava o número solto "17" | Vírgula adicionada à exclusão à esquerda da regex e sinal incorporado ao grupo. Virou teste de regressão. |
| 10 | **Defeitos no gerador offline** (jargão vazando do payload, ressalva clínica ausente no ramo de divergência) | Detectados pela Camada 1 da Etapa 7, corrigidos, e cobertos por quatro testes de regressão. |
| 11 | **Dependência entre etapas na suíte de testes** | Fixture constrói os dados a partir do scikit-learn, não de `data/processed/`. A suíte roda numa máquina limpa. |
| 12 | **Chave de API no IaC** apareceria em texto claro no state | Segredo criado vazio pelo Terraform, valor gravado fora do IaC, com `ignore_changes` para que um `apply` não o sobrescreva. |

---

## 13. Conclusão

**O que funcionou.** O algoritmo genético está implementado do zero,
coberto por 201 testes, e melhora a função objetivo que recebe nos três
modelos. A infraestrutura de experimentação — orçamento controlado, log
estruturado por geração, cache de fitness, reprodutibilidade por semente
— permitiu rodar 9 experimentos em 27,6 minutos e comparar estratégias de
forma justa. A camada de LLM produz explicações auditáveis, com a
auditoria verificando número por número contra a entrada.

**O que não funcionou, e o que isso ensinou.** O ganho de fitness não
virou ganho de desempenho. A lição não é que algoritmos genéticos não
servem para ajuste de hiperparâmetros — é que **otimizar um estimador de
desempenho não é a mesma coisa que melhorar o desempenho**, e que a
diferença entre as duas coisas é mensurável. Um GA eficiente sobre um
estimador ruidoso otimiza o ruído com a mesma competência com que
otimizaria o sinal.

A demonstração mais limpa disso está na seção 6: trocando a régua de
avaliação, **cada configuração vence apenas no estimador que a
selecionou** — 6 comparações, 6 confirmações. E a segunda metade da
lição veio junto: corrigir o estimador (CV repetida) eliminou 86-90% do
ganho ilusório, mas **não produziu um modelo melhor no teste**. Medir
melhor não substitui ter o que medir.

Dois agravantes específicos deste caso: os modelos da Fase 1 já estavam
muito bons (ROC AUC de 0,995 na Logística e na Floresta), deixando pouco
espaço para melhora real; e o conjunto de teste, com 42 casos malignos,
tem resolução de 2,4 pontos percentuais no recall — pequeno demais para
registrar ganho sutil.

**O que faria diferente com mais tempo.** Validação cruzada aninhada,
para que a *seleção* entre experimentos também fosse validada fora da
amostra. Otimização do limiar de decisão como gene — é a alavanca mais
direta sobre a troca entre recall e precisão, e nenhum dos
hiperparâmetros ajustados a controla diretamente. E um conjunto de teste
maior, o que neste dataset significaria validação cruzada repetida
reportada com intervalo de confiança em vez de um split único.

---

## Reprodução

```bash
pip install -r requirements.txt -r requirements-fase2.txt
cp .env.example .env            # preencher ANTHROPIC_API_KEY

python -m src.ga.run_experiments        # ~28 min
python -m src.ga.compare
python -m src.llm.explain
python -m src.llm.evaluate_quality --comparar-prompts
python -m pytest
```

Todas as execuções usam sementes fixas (42/43/44) e `random_state=42` nos
estimadores. Mesma configuração produz o mesmo resultado.
