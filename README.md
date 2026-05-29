# IEEE-CIS Fraud Detection: Exploration & Model Training

End-to-end fraud detection pipeline using the IEEE-CIS Kaggle dataset and the [Fraud Dataset Benchmark (FDB)](https://github.com/amazon-science/fraud-dataset-benchmark) data loader.

## Overview

This project explores multiple approaches to credit card fraud detection, from baseline models to advanced techniques like ensemble stacking, concept drift analysis, and semi-supervised learning.

## Pipeline

| Section | What it does |
|---------|-------------|
| **Data Loading** | Load standardized train/test splits via FDB |
| **EDA** | Class imbalance, missing values, distributions, correlations, temporal patterns |
| **Preprocessing** | Label encoding, imputation, data leakage prevention |
| **Baseline (RF)** | RandomForest with threshold analysis, confusion matrix |
| **Class Imbalance** | `class_weight='balanced'` comparison |
| **XGBoost** | Gradient boosting with `scale_pos_weight` |
| **Feature Engineering** | Log transforms, ratios, V-column aggregates, temporal features |
| **Optuna Tuning** | 50-trial Bayesian optimization with 3-fold CV |
| **Ensemble Stacking** | XGBoost meta-learner over RF + XGBoost + Tuned XGBoost |
| **Concept Drift** | Time-window evaluation, KS-test drift detection |
| **Semi-supervised** | SelfTrainingClassifier with 20% labeled data |
| **SHAP** | Feature importance and single-prediction explanations |

## Models Compared

| Model | AUC-ROC | TPR@1%FPR | Description |
|-------|---------|-----------|-------------|
| RF Baseline | — | — | Default RandomForest |
| RF Balanced | — | — | Class-weight adjusted |
| XGBoost | — | — | Default XGBoost |
| XGBoost + FE+Temp | — | — | Feature engineering + temporal features |
| XGBoost Tuned | — | — | Optuna-optimized hyperparameters |
| Stacked Ensemble | — | — | XGBoost meta-learner stacking 3 models |

> Run the notebook to fill in the results with your own metrics.

## Requirements

- Python 3.7+
- Kaggle account with [IEEE-CIS competition](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) rules accepted
- Kaggle API token at `~/.kaggle/kaggle.json`

### Python Dependencies

```bash
pip install kaggle scikit-learn requests python-dateutil numpy pandas Faker
pip install matplotlib seaborn xgboost shap optuna jupyterlab
pip install -e fraud-dataset-benchmark
```

### System Dependencies

```bash
brew install libomp  # Required for XGBoost on macOS
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/camiloavilacm/IEEE_CIS_Fraud_Detection_models_exploration.git
cd IEEE_CIS_Fraud_Detection_models_exploration

# Install FDB (from cloned submodule or pip)
pip install -e fraud-dataset-benchmark

# Launch Jupyter Lab
jupyter lab
```

Open `ieeecis_exploration.ipynb` and run all cells.

> **Note:** First run downloads ~500MB of data from Kaggle. Subsequent runs use cached data.

## Key Learnings

- **Class imbalance matters** — accuracy is misleading at 3.5% fraud rate. Use AUC-ROC, Precision-Recall, and TPR@1%FPR.
- **XGBoost > RandomForest** — gradient boosting consistently outperforms for tabular fraud data.
- **Temporal features help** — hour-of-day and day-of-week capture fraud patterns.
- **Hyperparameter tuning** — Optuna finds better params than defaults (especially `learning_rate` and `max_depth`).
- **Ensemble stacking** — combining diverse models captures different fraud signals.
- **Concept drift is real** — models trained on older data perform worse on recent transactions.
- **Semi-supervised works** — using only 20% of labels gets close to fully supervised performance.
- **SHAP reveals patterns** — transaction amount, V-column aggregates, and device info are top fraud indicators.

## Project Structure

```
IEEE_CIS_Fraud_Detection_models_exploration/
├── ieeecis_exploration.ipynb   # Main notebook (65 cells)
├── test_fdb.py                 # Quick FDB setup verification
├── .gitignore
└── README.md
```

## Dataset

The [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset contains ~590K anonymized transactions with 393 features (reduced to 67 by FDB). Provided by Vesta Corporation.

- **Train:** 561,013 rows, 3.5% fraud rate
- **Test:** 29,527 rows, time-based split
- **Features:** 6 categorical, 61 numerical (V, C, D, M prefixes)

## References

- [Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) — Grover et al., 2023
- [IEEE-CIS Fraud Detection Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## License

MIT
