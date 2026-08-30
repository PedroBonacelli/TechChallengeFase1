# TechChallengeFase1

Projeto pro Tech Challenge da Fase 1 (IADT). Constrói a base de um sistema
de suporte ao diagnóstico e detecção de riscos pra saúde da mulher, usando
Machine Learning sobre dados médicos estruturados e, na etapa extra, Visão
Computacional (CNN) sobre imagens de mamografia.

## O desafio

Uma rede de hospitais quer identificar mais cedo condições que afetam a
segurança e a saúde feminina (câncer de mama, por exemplo). Esta fase
constrói a base de IA/ML que processa dados médicos pra apoiar — nunca
substituir — a decisão do profissional de saúde.

## Datasets

- **Breast Cancer Wisconsin (Diagnostic)** — dados estruturados de exames
  pra classificar tumores em malignos/benignos.
  Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **[EXTRA] CBIS-DDSM** — imagens de mamografia pra diagnóstico via CNN.
  Fonte: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

Os datasets não são versionados (ver `.gitignore`); instruções de download
em `data/raw/README.md`.

## Estrutura

```
tech-challenge-fase1/
├── data/
│   ├── raw/            # dados brutos (não versionados; ver data/raw/README.md)
│   └── processed/      # gerados por src/preprocessing.py
├── src/                # scripts do pipeline (ver "Como executar")
├── models/             # modelos treinados (.joblib)
├── reports/
│   ├── figures/
│   ├── RELATORIO_TECNICO.md   # leia primeiro
│   ├── 01_eda_findings.md
│   ├── 02_model_evaluation.md
│   ├── 03_cnn_extra.md         # [extra] visão computacional
│   ├── model_comparison.csv
│   └── cnn_model_comparison.csv
├── requirements.txt
├── Dockerfile
└── README.md
```

## Como executar

São scripts Python (não notebooks) — cada um roda uma etapa e salva o
resultado em `reports/`. Rode nesta ordem:

### venv local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/load_data.py       # baixa/gera o dataset em data/raw/
python src/eda.py             # EDA -> reports/figures/
python src/preprocessing.py   # limpeza, split, scaling -> data/processed/
python src/modeling.py        # treina os 3 modelos -> models/
python src/evaluation.py      # avaliação + explicabilidade
```

### [Extra] Visão computacional

Precisa do CBIS-DDSM baixado à parte (ver `data/raw/README.md`). Aponte
`--cbis-root` pra raiz do download:

```bash
python src/cnn_data_prep.py --cbis-root ../cnn   # manifesto + correção de vazamento
python src/cnn_cache.py                          # decodifica as imagens uma vez só
python src/cnn_eda.py                            # EDA das imagens -> figuras 13-17
python src/cnn_mammography.py --all              # treina as 4 combinações (~1h30 em CPU)
python src/cnn_evaluate.py                       # avaliação + Grad-CAM -> figuras 18-21
```

`python src/cnn_mammography.py --all --quick` valida o pipeline inteiro em
poucos minutos antes de disparar o treino completo. Discussão em
`reports/03_cnn_extra.md`.

### Docker

```bash
docker build -t tech-challenge-fase1 .
docker run -v $(pwd):/app tech-challenge-fase1 python src/load_data.py
# repete pros demais scripts, na mesma ordem
```

## Resultados

Modelo recomendado: **Regressão Logística** — 97,6% de recall e AUC de
0,995 na classe maligno, no teste (114 amostras). Comparação completa em
`reports/model_comparison.csv`, discussão em `reports/RELATORIO_TECNICO.md`.

### [Extra] CNN em mamografias

Melhor modelo: **recorte da lesão + transfer learning (MobileNetV2)** —
ROC AUC 0,707 no teste (644 imagens). Mas o achado principal é outro: o
Grad-CAM mostrou que o segundo colocado na tabela de AUC acerta olhando
pros marcadores de texto queimados na imagem, não pra lesão. Métrica sem
explicabilidade engana. Ver `reports/03_cnn_extra.md`.

## Roteiro

1. EDA — `src/eda.py`
2. Pré-processamento — `src/preprocessing.py`
3. Modelagem (3 algoritmos) — `src/modeling.py`
4. Avaliação e explicabilidade — `src/evaluation.py`
5. [Extra] CNN pra diagnóstico via imagem, com Grad-CAM — `src/cnn_*.py`, discussão em `reports/03_cnn_extra.md`
6. Relatório técnico com discussão crítica — `reports/RELATORIO_TECNICO.md`

## Aviso

Isso é uma prova de conceito de apoio à decisão. O diagnóstico final é
sempre responsabilidade do profissional de saúde.
