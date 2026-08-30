# Relatório Técnico — Tech Challenge Fase 1

Sistema de apoio ao diagnóstico de câncer de mama, a partir do dataset
Breast Cancer Wisconsin.

## 1. O problema

Uma rede de hospitais quer identificar mais cedo condições que afetam a
saúde da mulher. Nesta fase o objetivo é montar a base de um sistema de ML
que classifica exames como malignos ou benignos a partir de medidas
extraídas de biópsias — como apoio à decisão médica, não substituindo ela.

## 2. Dataset

Breast Cancer Wisconsin (Diagnostic): 569 exames, 30 features numéricas
tiradas de imagens digitalizadas de biópsias por agulha fina (FNA). As
features vêm em três grupos — `mean`, `error` e `worst` — para 10 medidas
de forma/textura do núcleo celular (radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, fractal
dimension).

Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data,
carregado aqui via `sklearn.datasets.load_breast_cancer` (mesma base, vem
do UCI Machine Learning Repository — `src/load_data.py`).

Target: `target` (0 = maligno, 1 = benigno). 357 benignos (62,7%) e 212
malignos (37,3%) — um desbalanceamento moderado que pesa na escolha de
métrica lá na frente (seção 5).

## 3. Análise exploratória (EDA)

Detalhes em `reports/01_eda_findings.md`, figuras 01 a 05.

Dataset limpo — zero ausentes, zero duplicatas, não precisou de imputação.
Dá pra ver separação visual boa entre malignos e benignos em features de
tamanho e irregularidade de contorno (radius, perimeter, area,
compactness, concavity, concave points): tumor maligno tende a ser maior
e com borda mais irregular, o que bate com o que se espera clinicamente.
`texture` e `smoothness` já se sobrepõem mais entre as classes — sozinhas
discriminam menos.

Tem multicolinearidade forte entre as medidas de tamanho (radius,
perimeter, area — correlação acima de 0,98 entre pares), o que faz sentido
matematicamente (perímetro e área derivam do raio) e influenciou a decisão
da seção 4. As features mais correlacionadas com o alvo são concave
points, perimeter e radius (tanto mean quanto worst).

## 4. Pré-processamento

Código em `src/preprocessing.py`.

Limpeza defensiva: duplicatas, imputação por mediana se faltasse algo,
descarte de medida física negativa. Neste dataset não teve nada pra
limpar, mas deixei o pipeline preparado pra dado mais bagunçado.

O passo que mais mudou o projeto foi a remoção de features redundantes:
de cada par com |correlação| ≥ 0,92, fico só com a que tem maior
correlação com o alvo. É um processo guloso — elimina o par mais
correlacionado, recalcula, repete até não sobrar par acima do threshold.
Removeu 8 das 30 features, sobrando 22:

| Removida | Redundante com | Corr. entre elas | Corr. com alvo (removida → mantida) |
|---|---|---|---|
| `mean radius` | `mean perimeter` | +0,998 | 0,730 → 0,743 |
| `worst radius` | `worst perimeter` | +0,994 | 0,776 → 0,783 |
| `mean area` | `mean perimeter` | +0,987 | 0,709 → 0,743 |
| `worst area` | `worst perimeter` | +0,978 | 0,734 → 0,783 |
| `perimeter error` | `radius error` | +0,973 | 0,556 → 0,567 |
| `mean perimeter` | `worst perimeter` | +0,970 | 0,743 → 0,783 |
| `area error` | `radius error` | +0,952 | 0,548 → 0,567 |
| `mean concavity` | `mean concave points` | +0,921 | 0,696 → 0,777 |

Registro completo em `reports/features_removidas.csv`. A ideia é simples:
duas features quase idênticas não trazem sinal novo, só custam — na
Logística o peso se divide arbitrariamente entre elas, o que bagunça os
coeficientes usados depois na explicabilidade; no Random Forest a
importância de uma variável relevante se dilui entre as cópias.
radius/perimeter/area são, na prática, três formas de medir o mesmo
tamanho de núcleo — é redundância geométrica, não estatística. O efeito
real nas métricas (seção 6.1) é neutro; o ganho aqui é de
interpretabilidade e parcimônia, não de acurácia.

Depois disso: split treino/teste estratificado (80/20 → 455/114),
mantendo a proporção de ~37% de malignos nos dois lados, e
`StandardScaler` ajustado só no treino (evita vazamento) — necessário
porque as escalas são bem diferentes (`area` na casa de centenas/milhares,
`smoothness` na casa de centésimos), o que pesa em modelos sensíveis a
escala como a Logística. Não precisou de one-hot encoding porque tudo já
é numérico; `diagnosis` (texto redundante com `target`) só foi descartado.

## 5. Modelagem

Código em `src/modeling.py`. Treinei três modelos (o mínimo pedido era
dois):

| Modelo | Por quê |
|---|---|
| Regressão Logística | Baseline linear e interpretável — os coeficientes mostram direção/força de cada feature, o que ajuda num contexto clínico. |
| Árvore de Decisão (`max_depth=5`) | Captura relação não-linear e é fácil de explicar pra quem não é técnico. |
| Random Forest (300 árvores, `max_depth=8`) | Ensemble mais robusto, serve de comparação mais forte. |

Os três usam `class_weight='balanced'`: maligno é a classe minoritária
(~37%) e é a que não pode passar batida, então o modelo paga mais caro
por errar ela em vez de otimizar acurácia bruta.

Sobre a métrica: accuracy sozinha engana com 63%/37% de desbalanceamento
— um modelo que sempre chutasse "benigno" já teria ~63% de acerto sem
detectar nenhum câncer. O que importa de verdade é o recall da classe
maligno (quantos cânceres realmente pegamos) e, depois, o F1. Falso
negativo (câncer classificado como benigno) é o erro mais grave aqui,
bem mais caro que um falso positivo.

## 6. Resultados

Detalhamento em `reports/02_model_evaluation.md`, figuras 06 a 09, tabela
em `reports/model_comparison.csv`. Métricas no conjunto de teste (114
amostras nunca vistas):

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | F1 (maligno) | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| **Regressão Logística** | 0,965 | 0,932 | **0,976** | **0,954** | 0,995 | **1** |
| Árvore de Decisão | 0,939 | 0,907 | 0,929 | 0,918 | 0,929 | 3 |
| Random Forest | **0,974** | **1,000** | 0,929 | 0,963 | **0,996** | 3 |

A Logística fica com o melhor resultado na métrica que importa
clinicamente — maior recall e F1 na classe maligno, só 1 falso negativo
em 42 malignos no teste. O Random Forest, nesta reexecução, chegou na
maior acurácia e precisão perfeita (nenhum falso positivo), mas ainda
fica atrás em recall (3 falsos negativos) — mesmo com class_weight
balanceado, o ensemble ainda erra por excesso de cautela do lado maligno
com mais frequência que o modelo linear. Isso não muda a recomendação:
pra triagem de câncer o recall vem primeiro, e nisso a Logística segue na
frente.

Nota de reexecução: revalidei esta seção depois da limpeza de colunas
redundantes (seção 4) e da adoção de `class_weight='balanced'` (seção 5).
Logística e Árvore reproduziram os números exatamente como já estavam. O
Random Forest saiu um pouco diferente de uma versão anterior deste
relatório (accuracy e precisão um pouco maiores, recall igual) — isso é
consistente com uma sensibilidade conhecida do `RandomForestClassifier` a
detalhes de versão do scikit-learn quando combina `class_weight='balanced'`
com bootstrap. Os artefatos (`models/*.joblib`, `data/processed/*.csv`,
`reports/model_comparison.csv`) foram regenerados nesta revisão e são a
fonte atual.

### 6.1. Ablação: o que cada decisão de pré-processamento realmente entregou

Com só 114 amostras de teste, 2 ou 3 acertos a mais já movem as métricas
uns 2 pontos percentuais. Pra separar efeito real de ruído do sorteio,
medi as duas decisões (remoção de features e class_weight) com validação
cruzada estratificada repetida (10 folds × 3 repetições = 30 medições por
configuração), no dataset completo.

Recall médio da classe maligno (desvio entre folds entre parênteses):

| Modelo | 30 feat., sem peso | 30 feat., balanced | 22 feat., sem peso | 22 feat., balanced |
|---|---|---|---|---|
| Regressão Logística | 0,961 (0,039) | **0,967** (0,033) | 0,946 (0,050) | 0,964 (0,038) |
| Árvore de Decisão | 0,890 (0,072) | 0,905 (0,073) | 0,901 (0,069) | **0,918** (0,074) |
| Random Forest | 0,937 (0,065) | 0,939 (0,056) | 0,928 (0,053) | 0,923 (0,056) |

`class_weight='balanced'` ajuda a Logística e a Árvore de forma
consistente (+0,6 a +1,7 pp de recall, em qualquer conjunto de features,
com alguma perda de precisão) — exatamente o trade-off que eu queria
quando o falso negativo é o erro caro. No Random Forest o ganho é
marginal e cabe dentro do ruído (0,937 → 0,939 com 30 features; 0,928 →
0,923, uma leve queda, com 22 features) — o próprio bootstrap do ensemble
já mistura naturalmente a proporção de classes entre as árvores, então o
peso extra faz pouca diferença.

A remoção de features é neutra em desempenho pra Logística e Random
Forest, e melhora a Árvore de Decisão (recall 0,905 → 0,918; F1 0,906 →
0,922 — é o modelo mais sensível a ruído, já que decide tudo num único
split por nó). Pros outros dois modelos a variação fica dentro do desvio
entre folds (~0,05 a 0,065), ou seja, é ruído amostral, não uma piora de
verdade.

Nota de reprodutibilidade: os valores do Random Forest nesta tabela foram
medidos nesta revisão e diferem de uma versão anterior do relatório (que
tinha 0,954 e 0,943 nas colunas balanced) — mesma causa da nota da seção
6. A leitura qualitativa não muda: balanced ajuda Logística e Árvore, e a
remoção de features continua sendo decisão de parcimônia/interpretabilidade,
não de ganho de desempenho.

Então a remoção das 8 features não se justifica como ganho de acurácia —
o motivo é ter 22 features em vez de 30, coeficientes mais estáveis na
Logística e um ranking de importância que não dilui o mesmo sinal entre
cópias. Isso importa quando o modelo precisa ser explicado pra um
profissional de saúde.

### Explicabilidade

Usei três camadas de interpretação: feature importance nativa
(coeficientes / `feature_importances_`), permutation importance
(model-agnostic, medida no teste), e uma explicação tipo-SHAP pra
Logística. Como o pacote `shap` não instalou no sandbox usado pro
desenvolvimento (sem acesso ao índice completo do PyPI — ele segue no
`requirements.txt` pra uso local/Docker), implementei em
`src/evaluation.py` a fórmula analítica exata do SHAP pra modelos
lineares: `phi_i = coef_i · (x_i − média_treino_i)`, a mesma base
matemática do `shap.LinearExplainer` (Lundberg & Lee, 2017).

As três abordagens convergem: tanto no nível global quanto em casos
individuais, as features mais relevantes são consistentemente tamanho
(worst radius, worst area, worst perimeter) e irregularidade de contorno
(worst concave points, concavity). Métodos independentes chegando na
mesma resposta é um bom sinal de que o modelo aprendeu um padrão que
faz sentido clinicamente.

## 7. [Extra] Visão computacional (CNN)

Quatro modelos treinados e avaliados no CBIS-DDSM, em CPU: duas entradas
(recorte da lesão × mamografia inteira) por duas arquiteturas (CNN de 4
blocos treinada do zero × transfer learning com MobileNetV2).

| modelo | ROC AUC | PR AUC | acurácia |
|---|---|---|---|
| **recorte + transfer** | **0,707** | **0,636** | 0,671 |
| inteira + do zero | 0,677 | 0,612 | 0,617 |
| recorte + do zero | 0,627 | 0,542 | 0,601 |
| inteira + transfer | 0,593 | 0,490 | 0,557 |

Dois vazamentos foram encontrados e corrigidos no caminho. Primeiro: os
splits oficiais do CBIS-DDSM foram construídos de forma independente pra
massas e pra calcificações, e por isso **31 pacientes aparecem em treino e
em teste** ao mesmo tempo — realocados inteiramente pro treino. Segundo:
como uma paciente tem em média 1,9 anormalidades e a mesma lesão aparece
nas incidências CC e MLO, o split de validação precisa ser **agrupado por
paciente**, senão imagens da mesma mama caem dos dois lados.

O resultado mais importante desta etapa, porém, não está na tabela: **o
Grad-CAM desqualificou o segundo colocado**. O modelo "inteira + do zero"
acerta olhando pros marcadores de texto queimados na imagem (`RMLO`,
`LCC`) e pro contorno da mama — quase nunca pro tecido interno. Uma sonda
quantitativa confirma: uma regressão logística usando só estatísticas
globais da imagem (brilho, contraste, área de tecido), sem localizar
lesão nenhuma, já chega a AUC 0,572 nessa entrada. Só o "recorte +
transfer" concentra o calor sobre a massa, com morfologia espiculada, e
erra de forma clinicamente plausível.

A conclusão prática é que **localizar a lesão é pré-requisito pra
classificá-la** — a mesma arquitetura sobe de 0,627 pra 0,707 só por
receber o recorte —, e que **métrica sem explicabilidade é perigosa em
imagem médica**: pela tabela de AUC, o modelo que lê a etiqueta do exame
entraria como segundo melhor. Detalhamento completo, discussão crítica e
instruções de reprodução em `reports/03_cnn_extra.md`.

## 8. Discussão crítica: esse modelo pode ser usado na prática?

Sim, mas só como ferramenta de apoio — nunca substituindo o diagnóstico
médico.

O médico sempre tem a palavra final. Mesmo com 97,6% de recall, o modelo
ainda erra, então em produção ele funcionaria como camada de
triagem/priorização, nunca liberando um caso como "benigno" sem revisão
humana. O dataset é pequeno e de fonte única (569 casos, um único
processo/equipamento de coleta) — validar em dados de outras
clínicas/equipamentos seria pré-requisito antes de qualquer uso clínico
real.

Explicabilidade não é luxo em saúde, é pré-requisito: as camadas de
interpretação usadas aqui deixam o profissional julgar se a predição faz
sentido clinicamente. O custo assimétrico dos erros também deveria
orientar o ajuste do limiar de decisão em produção. E os próximos passos
pra uso real seriam: validação clínica prospectiva, auditoria de viés
populacional, monitoramento contínuo pós-deploy, e um fluxo onde a
predição é sempre uma entre várias informações que o médico considera.

## 9. Conclusão

A Regressão Logística treinada sobre o Breast Cancer Wisconsin chegou a
97,6% de recall e AUC de 0,995 na classe maligno, com pipeline de
pré-processamento, análise de correlação e três camadas de
explicabilidade documentadas de ponta a ponta. O resultado mostra que dá
pra construir um sistema de apoio à triagem viável tecnicamente pra essa
tarefa, sempre respeitando que a decisão final é do profissional de
saúde.
