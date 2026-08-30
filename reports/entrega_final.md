---
title: "Tech Challenge — Fase 1"
subtitle: "Sistema de Apoio ao Diagnóstico de Câncer de Mama"
author: "Pedro Bonacelli"
date: "Agosto de 2026"
geometry: margin=2.5cm
toc: true
toc-depth: 2
colorlinks: true
---

\newpage

# 1. Entregáveis

**Repositório:** [https://github.com/PedroBonacelli/feature-fase-1](https://github.com/PedroBonacelli/feature-fase-1)

Tem lá: código-fonte (`src/`), `README.md` com instruções de execução,
`Dockerfile`, instruções de obtenção do dataset (`data/raw/README.md`),
os resultados (`reports/figures/`) e o relatório técnico completo
(`reports/RELATORIO_TECNICO.md`).

**Vídeo de demonstração:** *link a ser adicionado após a gravação — ver
`reports/03_cnn_extra.md` e o README pra saber o estado atual do
projeto.*

\newpage

# 2. O Problema e o Dataset

Uma rede de hospitais e centros de saúde especializados no atendimento à
mulher quer um sistema de suporte ao diagnóstico e detecção de riscos,
que ajude profissionais de saúde a identificar mais cedo condições que
afetam a segurança e saúde feminina. Nesta Fase 1 o desafio é construir a
base desse sistema com foco em Machine Learning sobre dados médicos
estruturados.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — 569 exames, 30
features numéricas derivadas de imagens digitalizadas de biópsias por
agulha fina (FNA), em três grupos (`mean`, `error`, `worst`) pra 10
medidas de forma/textura do núcleo celular. Fonte:
[kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
(carreguei via `sklearn.datasets.load_breast_cancer`, mesma base, origem
UCI Machine Learning Repository).

Alvo: `target` (0 = maligno, 1 = benigno). Distribuição: 357 benignos
(62,7%) / 212 malignos (37,3%).

\newpage

# 3. Análise Exploratória de Dados (EDA)

Dataset limpo: zero valores ausentes, zero duplicatas. Dá pra ver
separação clara entre malignos e benignos em features de tamanho e
irregularidade de contorno, o que bate com o que se sabe clinicamente
sobre morfologia tumoral.

![Distribuição da variável alvo](figures/01_class_distribution.png){width=45%}

![Distribuição das principais features por diagnóstico — nota-se maior separação de classes em radius, perimeter, area, compactness, concavity e concave points](figures/02_feature_distributions.png){width=100%}

Há forte multicolinearidade entre medidas de tamanho (`radius`,
`perimeter`, `area` — correlação > 0,98 entre pares), esperado
matematicamente e que orientou decisões de modelagem.

![Matriz de correlação entre as 30 features](figures/04_correlation_heatmap.png){width=100%}

\newpage

# 4. Pré-processamento

1. Limpeza defensiva (duplicatas, imputação por mediana, descarte de
   medidas fisicamente inválidas) — 0 registros afetados neste dataset.
2. Remoção de features redundantes: de cada par com |correlação| ≥ 0,92,
   mantive só a mais correlacionada com o alvo — 8 das 30 features
   descartadas (`mean/worst radius`, `mean/worst area`, `mean perimeter`,
   `perimeter error`, `area error`, `mean concavity`), restando 22.
   `radius`, `perimeter` e `area` são três leituras do mesmo tamanho do
   núcleo celular; manter as três instabiliza os coeficientes da
   Logística e dilui a importância no Random Forest. Registro em
   `reports/features_removidas.csv`.
3. Split treino/teste estratificado (80/20 → 455/114 amostras),
   preservando a proporção de ~37% de malignos.
4. Escalonamento (`StandardScaler`) ajustado só no treino, evitando
   vazamento de dados.

\newpage

# 5. Modelagem

Três técnicas de classificação treinadas e comparadas (acima do mínimo
de duas):

| Modelo | Justificativa |
|---|---|
| Regressão Logística | Baseline linear, altamente interpretável |
| Árvore de Decisão (`max_depth=5`) | Não-linear, fácil de visualizar/explicar |
| Random Forest (300 árvores) | Ensemble mais robusto, comparação mais forte |

Os três usam `class_weight='balanced'`: maligno é a classe minoritária
(~37%) e é a que não se pode perder, então o modelo paga mais caro por
errá-la, priorizando recall sobre acurácia bruta.

**Métrica escolhida:** recall e F1 da classe maligno, não só accuracy —
com esse desbalanceamento (63%/37%), um modelo que sempre previsse
"benigno" já teria ~63% de acurácia sem detectar nenhum câncer. Falso
negativo (câncer classificado como benigno) é o erro mais grave
clinicamente.

\newpage

# 6. Resultados

Métricas no conjunto de teste (114 amostras nunca vistas no treino):

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | F1 (maligno) | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| **Regressão Logística** | 0,965 | 0,932 | **0,976** | **0,954** | 0,995 | **1** |
| Árvore de Decisão | 0,939 | 0,907 | 0,929 | 0,918 | 0,929 | 3 |
| Random Forest | **0,974** | **1,000** | 0,929 | 0,963 | **0,996** | 3 |

A validação cruzada repetida (10 folds × 3 repetições) confirma que
`class_weight='balanced'` eleva o recall da Logística e da Árvore de
forma consistente (efeito marginal no Random Forest, dentro do ruído
entre folds), enquanto a remoção das 8 features é neutra em desempenho
(melhora a Árvore, pros demais fica dentro do desvio entre folds) — o
ganho ali é de interpretabilidade. Detalhamento na seção 6.1 do
`RELATORIO_TECNICO.md`.

A Regressão Logística fica com o melhor resultado na métrica que mais
importa clinicamente — maior recall e F1 na classe maligno, e só 1 falso
negativo. O Random Forest, nesta reexecução, chegou na maior acurácia e
precisão perfeita (zero falsos positivos), mas ficou atrás no recall (3
falsos negativos) — por isso a Logística segue sendo o modelo recomendado
pra triagem.

*Nota: esta seção foi revalidada nesta revisão depois da remoção de
features redundantes e da adoção de class_weight='balanced'. Os números
da Logística e da Árvore reproduziram exatamente; os do Random Forest
mudaram um pouco em relação a uma execução anterior (accuracy/precisão
maiores, recall igual), efeito consistente com sensibilidade do
RandomForestClassifier a versões do scikit-learn — detalhes na seção 6.1
do RELATORIO_TECNICO.md.*

![Matrizes de confusão dos três modelos no conjunto de teste](figures/06_confusion_matrices.png){width=100%}

![Curvas ROC dos três modelos](figures/07_roc_curves.png){width=70%}

## Explicabilidade

Três camadas de interpretação: feature importance nativa, permutation
importance (model-agnostic) e uma explicação tipo-SHAP pra Regressão
Logística (fórmula analítica exata do SHAP pra modelos lineares —
`phi_i = coef_i · (x_i − média_treino_i)` — implementada na mão, já que o
pacote `shap` não instalou no ambiente sandbox usado neste
desenvolvimento).

![Feature importance / coeficientes dos três modelos](figures/08_feature_importance.png){width=100%}

As três abordagens convergem: as features mais relevantes são
consistentemente tamanho (`worst radius`, `worst area`, `worst
perimeter`) e irregularidade de contorno (`worst concave points`,
`concavity`) — essa convergência reforça que o modelo aprendeu um padrão
clinicamente plausível.

\newpage

# 7. [Extra] Visão Computacional (CNN)

Quatro modelos treinados e avaliados no CBIS-DDSM, em CPU (~56 min de
treino): duas entradas (recorte da lesão × mamografia inteira) por duas
arquiteturas (CNN de 4 blocos do zero × transfer learning com
MobileNetV2).

| modelo | ROC AUC | PR AUC | acurácia |
|---|---|---|---|
| **recorte + transfer** | **0,707** | **0,636** | 0,671 |
| inteira + do zero | 0,677 | 0,612 | 0,617 |
| recorte + do zero | 0,627 | 0,542 | 0,601 |
| inteira + transfer | 0,593 | 0,490 | 0,557 |

**Dois vazamentos corrigidos.** Os splits oficiais do CBIS-DDSM foram
construídos de forma independente pra massas e pra calcificações, o que
faz **31 pacientes aparecerem em treino e teste** ao mesmo tempo —
realocados inteiramente pro treino. E como uma paciente tem em média 1,9
anormalidades, com a mesma lesão nas incidências CC e MLO, o split de
validação é **agrupado por paciente**.

![Curvas ROC e Precisão-Recall dos quatro modelos no teste](figures/20_cnn_roc_pr.png){width=100%}

**O Grad-CAM desqualificou o segundo colocado.** O modelo "inteira + do
zero" acerta olhando pros marcadores de texto queimados na imagem
(`RMLO`, `LCC`) e pro contorno da mama, quase nunca pro tecido interno.
Uma sonda quantitativa confirma: regressão logística com apenas
estatísticas globais da imagem — sem localizar lesão alguma — já atinge
AUC 0,572 nessa entrada. Só o "recorte + transfer" concentra o calor
sobre a massa espiculada e erra de forma clinicamente plausível.

![Grad-CAM do melhor modelo (recorte + transfer): o calor cai sobre a massa espiculada, e mesmo os falsos positivos são erros clinicamente plausíveis](figures/21_cnn_gradcam_patch.png){width=82%}

![Grad-CAM do segundo colocado (inteira + do zero): o calor cai sobre os marcadores de texto queimados na imagem e sobre o fundo — na linha "VN", em cima das próprias letras](figures/22_cnn_gradcam_full.png){width=82%}

Duas lições: **localizar a lesão é pré-requisito pra classificá-la** (a
mesma arquitetura sobe de 0,627 pra 0,707 só por receber o recorte) e
**métrica sem explicabilidade é perigosa em imagem médica** — pela tabela
de AUC, o modelo que lê a etiqueta do exame entraria como segundo melhor.

Ajustado pra **recall ≥ 90%** na classe maligna, o melhor modelo entrega
recall 0,911 com precisão 0,445: pra não perder 9 em cada 10 cânceres,
mais da metade dos alarmes são falsos. AUC 0,707 é insuficiente pra uso
clínico, mesmo como triagem. Discussão completa em
`reports/03_cnn_extra.md`.

\newpage

# 8. Discussão Crítica: Esse Modelo Pode Ser Usado na Prática?

Sim, mas só como ferramenta de apoio à decisão — nunca substituindo o
diagnóstico médico.

1. O médico sempre tem a palavra final. Mesmo com 97,6% de recall, o
   modelo ainda erra. Em produção, funcionaria como camada de
   triagem/priorização, nunca liberando um caso como "benigno" sem
   revisão humana.
2. Dataset pequeno e de fonte única (569 casos, um único
   processo/equipamento de coleta). Validar em dados de outras
   clínicas/equipamentos seria pré-requisito pra uso clínico real.
3. Explicabilidade é pré-requisito, não luxo, em saúde — as camadas de
   interpretação usadas permitem que o profissional julgue se a predição
   faz sentido clinicamente.
4. Custo assimétrico dos erros deveria orientar o ajuste do limiar de
   decisão em produção.
5. Próximos passos pra uso real: validação clínica prospectiva, auditoria
   de viés populacional, monitoramento contínuo pós-deploy, e um fluxo de
   trabalho onde a predição é sempre uma entre várias informações
   consideradas pelo médico.

# 9. Conclusão

A Regressão Logística treinada sobre o Breast Cancer Wisconsin chegou a
97,6% de recall e AUC de 0,995 na classe maligno, com pipeline de
pré-processamento, análise de correlação e três camadas de
explicabilidade documentadas de ponta a ponta. O resultado valida a
viabilidade técnica de um sistema de apoio à triagem pra essa tarefa,
respeitando desde já o princípio de que a decisão final é sempre do
profissional de saúde.
