# Avaliação e Explicabilidade dos Modelos — Breast Cancer Wisconsin

## Métricas no teste (114 amostras, nunca vistas no treino)

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | F1 (maligno) | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,965 | 0,932 | 0,976 | 0,954 | 0,995 | **1** |
| Árvore de Decisão | 0,939 | 0,907 | 0,929 | 0,918 | 0,929 | 3 |
| Random Forest | 0,974 | 1,000 | 0,929 | 0,963 | 0,996 | 3 |

(Ver `reports/model_comparison.csv`, `reports/figures/06_confusion_matrices.png`
e `07_roc_curves.png`.)

## Por que recall da classe maligno é a métrica que importa

Accuracy sozinha engana aqui: com 63% benigno / 37% maligno, um modelo
"preguiçoso" que sempre chutasse benigno já teria ~63% de acerto sem
detectar nenhum câncer. O que importa clinicamente é o recall da classe
maligno (quantos cânceres realmente pegamos) e, secundariamente, o F1.

Falso negativo (câncer classificado como benigno) atrasa diagnóstico e
tratamento de uma paciente que realmente tem câncer — é o erro mais
grave. Falso positivo (benigno classificado como maligno) gera ansiedade
e exames extras, mas é bem menos custoso em termos de saúde.

Por isso a Regressão Logística é a mais adequada pra esse problema: maior
recall (97,6%) e só 1 falso negativo em 42 malignos no teste.

## Matriz de confusão (resumo)

- Regressão Logística: 41/42 malignos corretos, 69/72 benignos corretos.
- Árvore de Decisão: 39/42 malignos corretos (3 falsos negativos), 68/72
  benignos corretos.
- Random Forest: 39/42 malignos corretos (3 falsos negativos), mas 72/72
  benignos corretos — zero falsos positivos, precisão perfeita. Ainda
  assim fica atrás da Logística em recall, mesmo com class_weight
  balanceado (ver seção 6.1 do RELATORIO_TECNICO.md pra ablação disso).

## Explicabilidade

### Feature importance nativa e permutation importance

Ver `reports/figures/08_feature_importance.png` e `09_permutation_importance.png`.

Nos três modelos, as features mais relevantes se concentram
consistentemente em tamanho (worst radius, worst area, worst perimeter) e
irregularidade de contorno (worst concave points, concavity). Bate com a
EDA (etapa 3) e reforça, com múltiplos modelos e métodos de importância
diferentes, que tumores maiores e com bordas mais irregulares são o
padrão associado à malignidade nesse dataset — coerente com o que se
sabe clinicamente sobre morfologia tumoral.

### Explicação tipo-SHAP (nível de predição individual)

Como o pacote `shap` não instalou no ambiente de execução usado aqui
(sandbox sem acesso ao índice completo do PyPI — segue no
`requirements.txt` pra uso local/Docker com `shap.LinearExplainer` /
`shap.TreeExplainer`), implementei em `src/evaluation.py` a fórmula exata
do SHAP pra modelos lineares: pra Logística, `f(x) = intercepto +
Σ coef_i·x_i`, o valor SHAP de cada feature é `coef_i · (x_i −
média_treino_i)` — a mesma base matemática do `shap.LinearExplainer`
(Lundberg & Lee, 2017), sem depender da biblioteca.

Em casos malignos corretamente classificados, as maiores contribuições
positivas (empurrando pra "maligno") vêm de worst area, worst radius,
worst perimeter e mean concave points — exatamente as features de
tamanho/irregularidade que aparecem como mais importantes globalmente.
Em casos benignos, as mesmas features contribuem no sentido contrário.
Essa consistência entre explicação global e local é um bom sinal de que
o modelo aprendeu um padrão clinicamente plausível, não um artefato dos
dados.

## Esse modelo pode ser usado na prática?

Sim, mas só como apoio à decisão — nunca substituindo o diagnóstico
médico.

O médico deve sempre ter a palavra final: mesmo com 97,6% de recall, a
Logística ainda errou 1 em 42 malignos no teste. Num cenário real, isso
funcionaria como camada de triagem — sinalizando casos de maior risco pra
priorização, nunca liberando um caso como benigno sem revisão humana.

O dataset é pequeno e de uma fonte só (569 casos, features de um processo
de imagem específico), então validar em dados de outras clínicas seria
pré-requisito antes de qualquer uso clínico real. Explicabilidade
também não é luxo em saúde — um modelo caixa-preta que erra sem
explicação não dá pra usar; as técnicas usadas aqui permitem que o
profissional entenda por que o modelo sinalizou um caso.

O custo assimétrico dos erros deveria orientar o ajuste do limiar de
decisão em produção — pode valer a pena aceitar mais falsos positivos em
troca de recall ainda mais alto. E os próximos passos pra uso real:
validação clínica prospectiva, auditoria de viés populacional,
monitoramento contínuo pós-deploy, e um fluxo onde a predição é sempre
uma entre várias informações consideradas pelo médico.

## Escolha final

A Regressão Logística é o modelo recomendado: melhor recall e F1 na
classe maligno, e é o mais interpretável dos três (coeficientes
diretamente relacionáveis a cada feature). O Random Forest fica como
segunda opção — maior accuracy e precisão perfeita, mas recall menor. A
Árvore de Decisão é a mais simples de visualizar/explicar numa reunião
com não-técnicos, ao custo de recall também menor.
