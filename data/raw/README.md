# Medical Abstracts Text Classification Corpus

CSVs originais, sem modificação, do **Medical Abstracts TC Corpus**, versionados aqui para que o treino, o CI e a DAG do Airflow não dependam de download.

| Arquivo | Linhas | Conteúdo |
|---|---|---|
| `medical_tc_train.csv` | 11.550 | `condition_label` (1–5), `medical_abstract` |
| `medical_tc_test.csv` | 2.888 | idem |
| `medical_tc_labels.csv` | 5 | `condition_label`, `condition_name` |

- **Fonte:** https://github.com/sebischair/Medical-Abstracts-TC-Corpus (o mesmo conteúdo está no [Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus))
- **Autores:** Tim Schopf, Daniel Braun e Florian Matthes, no artigo *Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches* (2022), https://doi.org/10.1145/3582768.3582795
- **Licença:** [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/). Estes arquivos continuam sob essa licença.

Para atualizar a partir da fonte: `python -m medical_triage.data.ingest --download`.
