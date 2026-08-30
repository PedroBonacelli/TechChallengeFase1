# [EXTRA] Visão Computacional — CNN para Mamografias (CBIS-DDSM)

## Resumo

Quatro modelos treinados e avaliados no CBIS-DDSM: duas entradas (recorte da
lesão × mamografia inteira) por duas arquiteturas (CNN do zero × transfer
learning com MobileNetV2). Tudo rodou localmente em CPU, em ~56 minutos de
treino no total.

O melhor modelo — **recorte + transfer learning, ROC AUC 0,707 no teste** — é
também o único cuja explicabilidade se sustenta. O segundo colocado na tabela de
métricas (mamografia inteira + CNN do zero, AUC 0,677) foi **reprovado na
auditoria de Grad-CAM**: ele acerta olhando para os marcadores de texto queimados
na imagem, não para a lesão. Esse contraste é o resultado mais importante desta
etapa, e está detalhado na seção *Explicabilidade*.

## Dataset e o de-para entre CSVs e imagens

O CBIS-DDSM baixado do Kaggle vem em duas pastas: `csv/` (metadados) e `jpeg/`
(uma pasta por `SeriesInstanceUID`). Os nomes de pasta não batem com nada nos
CSVs de descrição de caso — o de-para precisa passar por `dicom_info.csv`:

- `mass_case_description_*.csv` e `calc_case_description_*.csv`: cada linha é uma
  **anormalidade**, não uma imagem. Trazem `pathology`
  (`BENIGN` / `BENIGN_WITHOUT_CALLBACK` / `MALIGNANT`) e `image file path`, cujo
  primeiro pedaço (`Mass-Training_P_00001_LEFT_CC`) identifica a mamografia.
- `dicom_info.csv`: `PatientName` bate com esse identificador e `image_path` dá o
  caminho do jpeg. `SeriesDescription` separa os três tipos de imagem:
  `full mammogram images` (2.857), `cropped images` (3.567) e
  `ROI mask images` (3.247).

`BENIGN_WITHOUT_CALLBACK` é tratado como benigno (convenção padrão do dataset):
são achados considerados benignos que não exigiram retorno da paciente.

Em vez de copiar imagens (~3 GB duplicados), `src/cnn_data_prep.py` gera um
**manifesto** (`data/processed/cbis_manifest.csv`) apontando para os arquivos
originais e carregando junto os metadados clínicos que a EDA usa.

## As duas entradas

| entrada | n | resolução original | rótulo |
|---|---|---|---|
| `patch` — recorte da lesão | 3.567 | mediana 337×340 | direto da linha (1 anormalidade = 1 imagem) |
| `full` — mamografia inteira | 2.857 | mediana 5236×3016 | agregado: se **alguma** anormalidade é maligna, a imagem é maligna |

Para os recortes, `PatientName` é `<image_id>_<abnormality id>`, o que casa
3.567 das 3.568 linhas dos CSVs de caso (99,97%). Para a imagem inteira, 246 dos
3.103 `image_id` distintos não têm mamografia completa correspondente em
`dicom_info.csv` e ficam de fora.

Uma amostra de 60 recortes foi checada contra contaminação por máscara de
segmentação (um problema conhecido de algumas versões do CBIS-DDSM): nenhum caso,
e uma única imagem por pasta. O `SeriesDescription` é confiável aqui.

## Dois vazamentos encontrados e corrigidos

Este foi o achado metodológico da etapa, e vale registrar porque nenhum dos dois
é óbvio.

**1. Os splits oficiais compartilham pacientes.** Os conjuntos de treino e teste
do CBIS-DDSM foram construídos de forma independente para massas e para
calcificações. O efeito colateral: **31 pacientes têm um caso de massa no treino
e um de calcificação no teste** — a mesma mama, o mesmo tecido, dos dois lados da
avaliação. `src/cnn_data_prep.py` realoca esses pacientes inteiramente para o
treino (mantém o teste limpo sem descartar dado). Depois da correção a
interseção de pacientes entre treino e teste é zero.

**2. O split de validação não pode ser aleatório.** Uma paciente tem em média 1,9
anormalidades (máximo 14), e a mesma lesão aparece nas incidências CC e MLO. Um
`validation_split` aleatório — como fazia a versão anterior deste código — coloca
imagens da mesma mama nos dois lados e infla a métrica de validação. A versão
atual usa `GroupShuffleSplit` **agrupado por `patient_id`**, com uma asserção que
falha o treino se algum paciente vazar.

## Pré-processamento

`src/cnn_cache.py` decodifica e redimensiona as imagens **uma única vez** para
arrays `uint8` em disco (escala de cinza; os 3 canais que a MobileNetV2 espera
são replicados no pipeline de treino). Sem isso, cada época re-decodificaria
JPEGs de 5236×3016 e o treino em CPU viraria I/O puro. Usando `Image.draft()`
do PIL, que deixa o decodificador JPEG já devolver em escala reduzida, as 6.424
imagens são processadas em ~75 segundos.

Augmentation deliberadamente conservadora: só flip horizontal (equivale a trocar
mama esquerda/direita), rotação ≤5%, zoom ≤10% e leve variação de contraste
(imita diferença de equipamento). Espelhamento vertical ou rotações grandes
destruiriam a coerência anatômica da imagem.

Pesos de classe balanceados compensam o desbalanceamento (~60/40).

## EDA das imagens (figuras 13–17)

- **BI-RADS `assessment` é quase determinístico**: categoria 2 → 0,3% de
  malignos; categoria 5 → 97,7%. Mas `assessment` é o julgamento do próprio
  radiologista sobre o exame. Usá-lo como feature seria vazamento clínico: o
  modelo estaria copiando a resposta em vez de lê-la na imagem. A CNN não o vê —
  ela recebe só pixels.
- **Densidade mamária não separa nada**: taxa de malignidade praticamente
  achatada entre as categorias 1 a 4 (0,38 a 0,42).
- **Sutileza tem relação contraintuitiva**: os achados marcados como *mais
  sutis* (subtlety 1) têm a **maior** taxa de malignidade (55%), não a menor.
  Ou seja, o caso difícil de ver é justamente o mais perigoso de errar.
- **Intensidade média difere entre classes** nos recortes (129 nos malignos
  contra 114 nos benignos, ~0,45 desvio-padrão). É um atalho potencial, e
  motivou a sonda quantitativa descrita abaixo.
- A figura 15 mostra visualmente o problema da mamografia inteira: a 128×128 a
  lesão simplesmente desaparece, e o que sobra é a silhueta da mama — e os
  **marcadores de texto queimados na imagem** (`RMLO`, `LCC`…) nos cantos.

## Arquiteturas

| | `scratch` | `transfer` |
|---|---|---|
| entrada | 128×128, escala de cinza | 160×160, cinza replicado em RGB |
| rede | 4 blocos Conv-BN-MaxPool (32→64→128→256), GlobalAveragePooling, Dense 128, Dropout 0,4 | MobileNetV2 (ImageNet) + GlobalAveragePooling + Dropout 0,3 |
| treino | Adam 1e-4, até 40 épocas | 2 estágios: backbone congelado (Adam 1e-3) → fine-tune do último bloco (Adam 1e-5) |

O treino em dois estágios não é preferência estética: com o backbone destravado
desde o início, os gradientes de uma cabeça densa aleatória destroem os pesos
pré-treinados. E fine-tune completo é inviável nesta máquina — medido em
7,7 s/batch contra ~0,4 s/batch com só o último bloco liberado. As camadas de
BatchNorm ficam congeladas mesmo no bloco liberado (com batch pequeno, atualizar
as estatísticas de um backbone pré-treinado costuma piorar o resultado).

Todos monitoram `val_auc` — não accuracy, que com classe desbalanceada engana —
com early stopping e redução de learning rate.

## Resultados no teste

Conjuntos de teste: 644 recortes (387 benignos / 257 malignos) e 368 mamografias
inteiras (213 / 155). **Os dois conjuntos são diferentes**, então a comparação
entre entradas é qualitativa, não um empate técnico medido na mesma régua.

| modelo | ROC AUC | PR AUC | acurácia | precisão | recall | épocas | tempo |
|---|---|---|---|---|---|---|---|
| **patch_transfer** | **0,707** | **0,636** | 0,671 | 0,607 | 0,498 | 18 | 9,8 min |
| full_scratch | 0,677 | 0,612 | 0,617 | 0,536 | 0,677 | 26 | 18,0 min |
| patch_scratch | 0,627 | 0,542 | 0,601 | 0,500 | 0,518 | 20 | 18,1 min |
| full_transfer | 0,593 | 0,490 | 0,557 | 0,483 | 0,716 | 20 | 10,6 min |

(métricas de precisão/recall no limiar padrão de 0,5; tabela completa com os dois
critérios de limiar em `reports/cnn_model_comparison.csv`)

Transfer learning ajudou nos recortes (0,627 → 0,707) e **atrapalhou** na imagem
inteira (0,677 → 0,593). Faz sentido: as features do ImageNet são de objetos
naturais em close, o que se transfere razoavelmente para uma textura de lesão que
preenche o quadro, e muito mal para uma silhueta de mama sobre fundo preto.

### Escolha do limiar

Accuracy é a métrica errada aqui, e o limiar padrão de 0,5 também. Em rastreio de
câncer o custo dos erros é assimétrico: um falso negativo manda para casa uma
paciente com câncer, um falso positivo gera um exame a mais. É a mesma lógica já
aplicada ao modelo de dados estruturados na etapa principal.

Ajustando o `patch_transfer` para **recall ≥ 90%** na classe maligna, o limiar cai
de 0,50 para 0,26 e o resultado passa a ser **recall 0,911 com precisão 0,445** —
ou seja, para não perder 9 em cada 10 cânceres, mais da metade dos alarmes são
falsos. Esse é o preço real, e é ele que deveria ser negociado com a equipe
clínica, não a acurácia.

## Explicabilidade — e o resultado que inverte a tabela

O Grad-CAM (figuras 21 e 22) gera o mapa de calor das regiões que mais empurraram
a predição, cumprindo em imagem o papel que o SHAP cumpre nos dados estruturados.
Aqui ele não foi confirmação de rotina: ele **desqualificou um modelo**.

**`patch_transfer` (figura 21)** — o calor cai sobre a massa, centrado na
morfologia espiculada que caracteriza malignidade. Mesmo os falsos positivos são
erros clinicamente plausíveis: o modelo se fixa em massas de aparência suspeita
que acabaram sendo benignas. É o tipo de erro que um radiologista reconhece e
sabe descontar.

**`full_scratch` (figura 22)** — o calor cai sobre os **marcadores de texto
queimados na imagem** (`RMLO`, `LMLO` nos cantos), sobre o fundo preto e sobre o
contorno externo da mama. Quase nunca sobre tecido interno. O modelo com o
segundo melhor AUC da tabela está, em boa parte, **lendo a etiqueta do exame em
vez da mamografia**.

Para medir o tamanho desse efeito, `src/cnn_evaluate.py` roda uma **sonda de
atalho**: uma regressão logística treinada só sobre estatísticas globais da
imagem (brilho médio, contraste, fração ocupada por tecido, percentil 90,
máximo) — nada que envolva localizar a lesão.

| entrada | sonda (só estatísticas globais) | melhor CNN | ganho real da CNN |
|---|---|---|---|
| `patch` | 0,592 | 0,707 | **+0,115** |
| `full` | 0,572 | 0,677 | +0,105 |

A leitura honesta: nenhuma das CNNs é dramaticamente melhor que uma regressão
logística de cinco números. A `patch_transfer` pelo menos ganha suas margens
olhando para o lugar certo — o que a sonda não captura e o Grad-CAM confirma. A
`full_scratch` ganha margem parecida olhando para o lugar errado, e por isso sua
métrica não deve ser levada a sério.

## Discussão crítica: dá para usar na prática?

**Não neste estado.** AUC 0,707 é insuficiente para qualquer uso clínico, mesmo
como triagem — e é bom ser explícito sobre isso em vez de vender o número. Para
comparação, o modelo de dados estruturados da etapa principal chega a 0,995 de
AUC, mas resolve um problema muito mais fácil (features morfológicas já
extraídas e medidas por um especialista, em vez de pixels crus).

O que esta etapa demonstra de fato:

1. **Localizar a lesão é pré-requisito para classificá-la.** A mesma arquitetura
   sobe de 0,627 para 0,707 só por receber o recorte em vez da imagem inteira, e
   o Grad-CAM mostra que na imagem inteira o modelo nem tenta olhar para a lesão.
   Um sistema real precisaria de um estágio de **detecção** antes do de
   classificação — o que o recorte aqui simula usando anotação humana, e que em
   produção teria que ser automático.
2. **Métrica sem explicabilidade é perigosa em imagem médica.** Se a decisão
   fosse tomada pela tabela de AUC, o `full_scratch` entraria como segundo melhor
   modelo. O Grad-CAM mostrou que ele aprendeu a ler a etiqueta do exame. Um
   modelo assim quebra silenciosamente assim que o hospital troca o equipamento
   que imprime esses marcadores — e quebra em produção, não na validação.
3. **O médico tem a palavra final**, e aqui isso não é formalidade: com precisão
   de 0,445 no limiar de recall alto, mais da metade dos alarmes exige revisão
   humana para ser descartada. O modelo, na melhor das hipóteses, ordena a fila.

Limitações que precisariam ser endereçadas antes de qualquer uso: fonte única
(um dataset, um conjunto de protocolos de aquisição); resolução de 128–160px
descarta microcalcificações, que são justamente parte do que o radiologista
procura; ausência de validação prospectiva; e a remoção dos marcadores queimados
como passo obrigatório de pré-processamento.

Próximos passos naturais: treinar em resolução maior com GPU; usar as máscaras de
ROI (que o dataset traz e aqui não foram usadas) para um estágio de detecção;
recortar automaticamente a região da mama e apagar os marcadores; e validar em
uma coorte externa.

## Como reproduzir

```bash
python src/cnn_data_prep.py --cbis-root ../cnn   # manifesto + correção de vazamento
python src/cnn_cache.py                          # decodifica as imagens uma vez só
python src/cnn_eda.py                            # EDA das imagens -> figuras 13-17
python src/cnn_mammography.py --all              # treina as 4 combinações (~56 min em CPU)
python src/cnn_evaluate.py                       # avaliação + Grad-CAM -> figuras 18-22
```

`python src/cnn_mammography.py --all --quick` valida o pipeline inteiro em poucos
minutos antes de disparar o treino completo.

Ambiente usado: TensorFlow 2.18, CPU de 8 núcleos, sem GPU. Log completo do
treino em `reports/cnn_train_log.txt`.

## Arquivos gerados

| arquivo | conteúdo |
|---|---|
| `reports/cnn_model_comparison.csv` | métricas dos 4 modelos, nos dois critérios de limiar |
| `reports/cnn_shortcut_probe.csv` | linha de base da sonda de atalho |
| `reports/cnn_eda_stats.csv` | fatores clínicos e intensidade por classe |
| `reports/cnn_history_*.csv` | histórico de treino por modelo |
| `reports/figures/13-17_cnn_*` | EDA das imagens |
| `reports/figures/18-20_cnn_*` | curvas de treino, matrizes de confusão, ROC/PR |
| `reports/figures/21_cnn_gradcam_patch.png` | Grad-CAM do melhor modelo |
| `reports/figures/22_cnn_gradcam_full.png` | Grad-CAM que revelou o atalho |
