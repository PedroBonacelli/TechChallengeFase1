# Roteiro pra explicar as seções 6 e 7 pelo código

Ordem sugerida: primeiro mostra o resultado (a tabela/figura do relatório),
depois abre o código que gerou aquilo. Isso evita ficar lendo código na
frente de todo mundo sem contexto do "pra quê".

---

## SEÇÃO 6 — Resultados (modelo estruturado)

### Passo 1 — mostra a tabela final

Abra `reports/RELATORIO_TECNICO.md`, seção 6, ou a própria tabela em
`reports/model_comparison.csv`.

**Fale:** "Testei três modelos — Regressão Logística, Árvore de Decisão e
Random Forest — no conjunto de teste, que são 114 amostras que nenhum dos
três viu durante o treino. A métrica que mais importa aqui não é
accuracy, é o recall da classe maligno: de cada 100 cânceres reais, quantos
o modelo realmente pegou. Um falso negativo — dizer que é benigno quando é
câncer — é o erro mais caro clinicamente."

Aponte o número: Logística com 97,6% de recall e só 1 falso negativo em 42
casos malignos no teste. Random Forest teve a maior accuracy e precisão
perfeita, mas ficou com recall um pouco menor (92,9%) — por isso a
Logística é o modelo recomendado.

### Passo 2 — abre `src/evaluation.py`

Mostre estas partes, nessa ordem:

1. **`load_models()` e `load_test_data()`** (topo do arquivo) — carrega os
   3 modelos já treinados (`.joblib`) e os dados de teste que ficaram de
   fora do treino.

2. **`evaluate_models()`** — é aqui que a mágica acontece:
   ```python
   y_pred = model.predict(X_eval)
   y_proba = model.predict_proba(X_eval)[:, POS_LABEL]
   ```
   **Fale:** "`predict` dá a classe final (0 ou 1), `predict_proba` dá a
   probabilidade antes do corte — isso é o que uso depois na
   explicabilidade." Logo abaixo, mostre onde calcula cada métrica
   (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`,
   `roc_auc_score`) e por que passo `pos_label=POS_LABEL` (POS_LABEL = 0):
   "maligno é a classe 0, não a 1, que é o padrão do sklearn — se eu não
   dissesse isso explicitamente, ia calcular recall pra classe errada."

3. **`main()`** (final do arquivo) — mostra que o script roda tudo em
   sequência: avalia, salva a tabela em `model_comparison.csv`, gera as 4
   figuras (matrizes de confusão, ROC, feature importance, permutation
   importance) e por fim a explicação tipo-SHAP.

### Passo 3 — mostra as figuras

`reports/figures/06_confusion_matrices.png` — aponte que o Random Forest
teve zero falsos positivos (72/72 benignos certos), mas 3 falsos
negativos, contra só 1 da Logística.

`reports/figures/07_roc_curves.png` — os três modelos têm AUC alto
(0,93 a 1,00), mas isso já era esperado nesse dataset — o ponto forte da
Logística é justamente no recall, que o AUC sozinho não mostra tão bem.

### Passo 4 — se der tempo, mostra a explicabilidade

Ainda em `src/evaluation.py`, função `linear_shap_explanation()`. Explica
a fórmula:
```python
coef = -pd.Series(model.coef_[0], index=X_train_scaled.columns)
contributions = coef * (x - baseline)
```
**Fale:** "Pra Regressão Logística, dá pra calcular a contribuição exata
de cada feature pra uma predição — é basicamente o coeficiente vezes o
quanto aquele valor se afasta da média do treino. Isso é a mesma
matemática por trás do SHAP pra modelos lineares, só que fiz na mão
porque não consegui instalar a biblioteca `shap` no ambiente que usei."

Se quiser, mostre também `reports/figures/08_feature_importance.png`:
os três modelos convergem nas mesmas features — tamanho (`worst
perimeter`, `worst radius`) e irregularidade de contorno (`worst concave
points`, `concavity`).

### Se perguntarem sobre a ablação (seção 6.1)

Abra `src/ablation.py` e `reports/ablacao_preprocessamento.csv`.
**Fale:** "Com só 114 amostras de teste, um resultado pode ser sorte. Pra
não cair nessa, testei as duas decisões de pré-processamento — remover
features redundantes e usar class_weight balanceado — com validação
cruzada repetida (10 folds vezes 3 repetições, 30 medições por
configuração). Isso mostrou que balanced ajuda de verdade a Logística e a
Árvore, mas remover as features não muda o desempenho — o ganho ali é só
de interpretabilidade."

---

## SEÇÃO 7 — [Extra] CNN pra mamografia

✅ **Rodou.** 4 modelos treinados em CPU, ~56 min. Os números e figuras
abaixo são reais e estão em `reports/03_cnn_extra.md`.

**A melhor forma de contar essa seção é como uma reviravolta**, porque foi
o que aconteceu de verdade: a tabela de métricas aponta um vencedor, e a
explicabilidade revela que o segundo colocado estava trapaceando. Guarde
esse final pro Passo 4 — ele é o ponto alto da apresentação inteira.

### Passo 1 — a decisão de dados (`src/cnn_data_prep.py`)

**Fale:** "O CBIS-DDSM não vem organizado em pastas benigno/maligno. Vem
em CSVs de metadado e uma pasta de imagens cujos nomes não batem com nada
— o de-para tem que passar pelo `dicom_info.csv`."

Mostre `load_dicom_lookup()` e explique que o `SeriesDescription` separa
três tipos de imagem: mamografia inteira (2.857), **recorte da lesão**
(3.567) e máscara de ROI (3.247).

**Fale:** "Aqui está a primeira decisão importante: treinei com as duas
entradas — a mamografia inteira e o recorte da lesão — justamente pra
comparar. Na inteira, a lesão é uma fração minúscula de uma imagem de
5236×3016. No recorte, ela preenche o quadro."

Mostre `reports/figures/15_cnn_sample_grid.png` — é a figura que vende o
argumento sozinha: em cima os recortes, com a lesão nítida; embaixo as
mamografias inteiras, onde a 128×128 não dá pra ver lesão nenhuma.
**Guarde na memória os marcadores de texto nos cantos** (`RMLO`, `LCC`) —
eles voltam no Passo 4.

### Passo 2 — os dois vazamentos (o ponto de rigor metodológico)

Esse passo é o que mostra cuidado de verdade. São dois problemas, nenhum
óbvio.

Mostre `fix_split_leakage()` em `src/cnn_data_prep.py`.
**Fale:** "Os splits oficiais de treino e teste do CBIS-DDSM foram feitos
de forma independente pra massas e pra calcificações. O efeito colateral é
que **31 pacientes têm um caso de massa no treino e um de calcificação no
teste** — mesma mama, mesmo tecido, dos dois lados da avaliação. Realoquei
esses pacientes inteiros pro treino, pra manter o teste limpo sem jogar
dado fora."

Agora mostre `split_por_paciente()` em `src/cnn_mammography.py`:
```python
splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
idx_treino, idx_val = next(splitter.split(imagens, rotulos, groups=grupos))
vazamento = set(grupos[idx_treino]) & set(grupos[idx_val])
assert not vazamento, f"vazamento treino/validação: {vazamento}"
```
**Fale:** "Segundo vazamento, mais sutil: uma paciente tem em média 1,9
anormalidades, e a mesma lesão aparece nas incidências CC e MLO. Se eu
usasse um split de validação aleatório, imagens da mesma mama cairiam dos
dois lados e a métrica de validação ficaria inflada. Por isso agrupo por
paciente — e deixei um `assert` que quebra o treino se algum paciente
vazar, pra não depender de eu lembrar de conferir."

### Passo 3 — as arquiteturas e a tabela

Em `src/cnn_mammography.py`, mostre `build_scratch()`:
```python
for filtros in (32, 64, 128, 256):
    x = layers.Conv2D(filtros, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
```
**Fale:** "CNN de 4 blocos, cada um dobrando os filtros. Global Average
Pooling em vez de Flatten — menos parâmetros e menos overfitting, que é
risco real com dataset de imagem pequeno."

Depois `build_transfer()` e o treino em dois estágios.
**Fale:** "A segunda arquitetura é transfer learning com MobileNetV2
pré-treinada na ImageNet. Treino em dois estágios: primeiro congelo o
backbone e treino só a cabeça, senão os gradientes de uma cabeça aleatória
destroem os pesos pré-treinados. Depois libero só o último bloco com
learning rate baixo — fine-tune completo levaria 7,7 segundos por batch
nessa CPU, contra 0,4 do jeito que fiz."

Mostre `build_augmentation()` e conecte com a seção 6 via
`pesos_de_classe()`: "mesma lógica do `class_weight='balanced'` dos
modelos estruturados."

Agora a tabela (`reports/cnn_model_comparison.csv` ou a seção 7 do
relatório):

| modelo | ROC AUC |
|---|---|
| **recorte + transfer** | **0,707** |
| inteira + do zero | 0,677 |
| recorte + do zero | 0,627 |
| inteira + transfer | 0,593 |

**Fale:** "Melhor modelo: recorte com transfer learning, AUC 0,707. E
repare no segundo colocado — mamografia inteira com CNN do zero, 0,677.
Quase empatado. Isso me incomodou, porque a gente **acabou de ver** que na
imagem inteira a 128×128 não dá pra enxergar a lesão. Como é que ele
acerta?"

*(deixe a pergunta no ar e vá pro Passo 4)*

### Passo 4 — o Grad-CAM, e a reviravolta

Mostre `grad_cam()` em `src/cnn_evaluate.py`:
```python
grads = tape.gradient(score, mapa)
pesos = tf.reduce_mean(grads, axis=(1, 2))
cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", mapa, pesos))
```
**Fale:** "O Grad-CAM gera um mapa de calor de onde a CNN olhou pra
decidir. É o equivalente, em imagem, ao que o SHAP faz nos dados
estruturados."

Agora as duas figuras, **nesta ordem**:

`reports/figures/21_cnn_gradcam_patch.png` (o vencedor).
**Fale:** "No modelo do recorte, o calor cai em cima da massa, centrado na
morfologia espiculada — que é exatamente o que caracteriza malignidade. E
olha os falsos positivos: ele erra fixado em massas de aparência suspeita
que acabaram sendo benignas. É um erro que um radiologista reconhece."

`reports/figures/22_cnn_gradcam_full.png` (a reviravolta).
**Fale:** "Agora o segundo colocado. O calor cai em cima dos **marcadores
de texto queimados na imagem** — aquele `RMLO` no canto que vimos na
figura do Passo 1 — no fundo preto e no contorno da mama. Quase nunca no
tecido. Esse modelo não está lendo a mamografia, está lendo a etiqueta do
exame."

Aponte a linha "VN — benigno correto" da figura: o vermelho está
literalmente em cima das letras.

Feche com a sonda quantitativa (`sonda_de_atalho()` em
`src/cnn_evaluate.py`, resultado em `reports/cnn_shortcut_probe.csv`):
**Fale:** "Pra medir isso e não ficar só no 'olha a figura', treinei uma
regressão logística usando só cinco estatísticas globais da imagem —
brilho médio, contraste, quanto da imagem é tecido. Nada de localizar
lesão. Ela sozinha chega a AUC 0,572 na imagem inteira. Ou seja: boa parte
do que aquele modelo 'aprendeu' dá pra conseguir sem olhar lesão nenhuma."

### Passo 5 — o fechamento crítico

**Fale:** "Duas lições, e nenhuma delas é o número 0,707. A primeira:
**localizar a lesão é pré-requisito pra classificá-la** — a mesma
arquitetura sobe de 0,627 pra 0,707 só por receber o recorte em vez da
imagem inteira. Um sistema real precisaria de um estágio de detecção antes
do de classificação.

A segunda, mais importante: **métrica sem explicabilidade é perigosa em
imagem médica**. Se eu tivesse escolhido o modelo pela tabela de AUC, o
modelo que lê a etiqueta do exame entraria como segundo melhor. E ele
quebraria em produção no dia que o hospital trocasse o equipamento que
imprime esses marcadores — não na validação, em produção, com paciente."

E o encerramento honesto:
**Fale:** "AUC 0,707 não serve pra uso clínico, nem como triagem, e é
importante dizer isso em vez de vender o número. Ajustando o limiar pra
pegar 90% dos cânceres, a precisão cai pra 0,445 — mais da metade dos
alarmes seriam falsos. Na melhor das hipóteses esse modelo ordena a fila
de revisão. O laudo continua sendo do radiologista."

### Se perguntarem "por que a CNN é tão pior que o modelo da seção 6?"

Pergunta provável, e a resposta é boa:
**Fale:** "Porque não são o mesmo problema. O modelo da seção 6 recebe 30
features morfológicas que um especialista já mediu e extraiu do exame —
raio, perímetro, concavidade. A CNN recebe pixel cru e tem que descobrir
sozinha o que é lesão e o que é tecido normal. O 0,995 daquele e o 0,707
deste medem tarefas de dificuldade completamente diferente."

---

## Se alguém perguntar "por que vocês confiam nesse código"

Ponto rápido pra ter na manga: todo o pipeline reproduz os mesmos números
sempre que roda de novo, porque o `random_state=42` está fixado em todo
lugar que tem aleatoriedade (split treino/teste, inicialização dos
modelos, validação cruzada). Isso não é frescura acadêmica — é o que
permite alguém pegar seu repositório, rodar os mesmos scripts na ordem
do README, e chegar exatamente nas mesmas tabelas e figuras que estão no
relatório.

Na etapa da CNN vale acrescentar: além do seed fixo, o pipeline tem
verificações que **quebram** se a premissa for violada — o `assert` do
split agrupado por paciente e a checagem de interseção treino/teste no
`cnn_data_prep.py`. É a diferença entre "acho que não vazou" e "o código
não deixa vazar".
