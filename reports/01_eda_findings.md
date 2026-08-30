# Análise Exploratória de Dados (EDA) — Breast Cancer Wisconsin

## Visão geral

569 registros, 30 features numéricas (medidas tiradas de imagens
digitalizadas de biópsias por agulha fina — FNA — de massas mamárias) mais
o alvo. Nenhum valor ausente, nenhuma duplicata — dataset já curado, então
não precisou de imputação. As 30 features vêm em 3 grupos de 10 medidas
cada: `mean`, `error` e `worst` (pior/maior valor observado) para radius,
texture, perimeter, area, smoothness, compactness, concavity, concave
points, symmetry e fractal dimension.

## Distribuição do alvo

| Diagnóstico | Casos | % |
|---|---|---|
| Benigno | 357 | 62,7% |
| Maligno | 212 | 37,3% |

Desbalanceamento moderado (~63/37) — não é extremo a ponto de precisar de
SMOTE ou undersampling, mas já é relevante o bastante pra métrica de
avaliação não poder ser só accuracy (ver seção de avaliação: recall da
classe maligno é a métrica que importa, já que um falso negativo em
câncer é bem mais grave que um falso positivo).

## Distribuições por diagnóstico

Ver `reports/figures/02_feature_distributions.png` e `03_feature_boxplots.png`.

Pra várias features de forma/tamanho — mean radius, mean perimeter, mean
area, mean compactness, mean concavity, mean concave points — dá pra ver
separação visível entre malignos e benignos: tumor maligno tende a ser
maior e com contorno mais irregular (mais concavidade, mais pontos
côncavos). Faz sentido clinicamente — malignos costumam crescer de forma
mais desorganizada e com bordas irregulares, benignos tendem a ser mais
regulares e compactos.

`mean texture` e `mean smoothness` já se sobrepõem mais entre as classes,
então sozinhas discriminam menos — mas ainda podem ajudar em conjunto com
outras features.

## Correlação entre features

Ver `reports/figures/04_correlation_heatmap.png` e `05_target_correlation.png`.

As features de tamanho (radius, perimeter, area) são fortemente
correlacionadas entre si (> 0,98 em vários pares) — esperado
matematicamente, já que perímetro e área derivam do raio. Isso é
multicolinearidade, e pesou na etapa de modelagem: modelos lineares
sofrem mais com isso do que árvores.

concave points, perimeter e radius (tanto mean quanto worst) são as que
mais se correlacionam com o diagnóstico, reforçando que tamanho e
irregularidade de contorno são os sinais mais fortes de malignidade nesse
dataset. As features de erro (`*_error`) em geral têm correlação fraca
com o alvo — a variabilidade da medição em si parece menos informativa
que a medida em si.

## O que isso implica pras próximas etapas

Não precisa tratar ausentes, mas precisa escalonar (as escalas são bem
diferentes — `area` na casa de centenas/milhares vs `smoothness` na casa
de centésimos — o que pesa especialmente pra KNN e Regressão Logística).
A multicolinearidade forte entre medidas de tamanho justifica remover
algumas features redundantes (ou usar regularização nos modelos
lineares). O bom poder discriminativo de várias features sozinhas sugere
que até modelos simples devem performar bem — testado na etapa de
modelagem. E o desbalanceamento moderado reforça acompanhar recall/F1 da
classe maligno, não só accuracy.
