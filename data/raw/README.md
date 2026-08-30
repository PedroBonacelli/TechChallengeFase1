# Dados brutos

Este diretório recebe os datasets originais (não versionados no Git).

## Breast Cancer Wisconsin (Diagnostic)

- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Também disponível via `scikit-learn`:
  `from sklearn.datasets import load_breast_cancer`
- Salve o CSV baixado do Kaggle como `data/raw/breast_cancer_wisconsin.csv`.

## [EXTRA] CBIS-DDSM (mamografias)

- Fonte: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- Baixado via `kagglehub`. O download vem com duas pastas:

  ```
  <raiz-do-download>/
      csv/     dicom_info.csv, meta.csv e os 4 CSVs de descrição de caso
      jpeg/    uma pasta por SeriesInstanceUID, com os .jpg
  ```

- **As imagens não são copiadas pra dentro do projeto.** `src/cnn_data_prep.py`
  gera um manifesto (`data/processed/cbis_manifest.csv`) que aponta pros
  arquivos originais e carrega junto os metadados clínicos. Isso evita
  duplicar ~3 GB e preserva as colunas que a EDA usa (densidade mamária,
  BI-RADS, sutileza, tipo de anormalidade).

- Aponte o script pra raiz do download com `--cbis-root` (default: `../cnn`):

  ```bash
  python src/cnn_data_prep.py --cbis-root ../cnn
  python src/cnn_cache.py
  ```

- `src/cnn_cache.py` decodifica e redimensiona as imagens uma única vez pra
  arrays `.npy` em `data/processed/cnn_cache/` (~250 MB). Sem esse passo, cada
  época de treino re-decodificaria JPEGs de 5236x3016 e o treino em CPU viraria
  I/O puro.

Ver `reports/03_cnn_extra.md` para a descrição completa do pipeline e os
resultados.
