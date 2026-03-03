# Fairness & Accuracy of AI in Criminal Justice

A research project investigating the relationship between model complexity, predictive accuracy, and algorithmic fairness across three criminal justice domains.

## Research Questions

1. How does feature complexity affect predictive performance across criminal justice domains?
2. How does feature reduction affect fairness metrics?
3. Can we identify "tipping points" where performance or fairness significantly degrade?

## Domains Analyzed

| Domain | Dataset | Source | Protected Attribute |
|--------|---------|--------|---------------------|
| Recidivism Prediction | ProPublica COMPAS | [ProPublica GitHub](https://github.com/propublica/compas-analysis) | Race (African-American vs Caucasian) |
| Predictive Policing | UCI Communities and Crime | [UCI ML Repository](https://archive.ics.uci.edu/dataset/183/communities+and+crime) | % Black population (binarised at median) |
| Victimisation Prediction | Chicago Strategic Subject List | [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Strategic-Subject-List-Historical/4aki-r3np) | Race (Black vs Non-Black) |

## Project Structure

```
thesis_fairness_cj/
├── config/
│   └── config.yaml                # Experiment configuration
├── src/
│   ├── data/
│   │   └── data_loader.py         # Dataset loaders (COMPAS, Communities, SSL)
│   ├── fairness/
│   │   ├── fairness_metrics.py    # SPD, EOD, DI, AOD, PED, calibration, etc.
│   │   ├── ml_metrics.py          # ML metrics + ROAR/ablation analysis
│   │   └── feature_selection.py   # FeatureSelector (MI, RF, LASSO, RFE, etc.)
│   ├── experiments/
│   │   └── run_experiments.py     # Main experiment runner (5 experiments)
│   └── analysis/
│       └── analyze_results.py     # Post-hoc analysis of saved results
├── results/
│   ├── compas_results.json        # COMPAS experiment output
│   ├── communities_crime_results.json
│   ├── ssl_results.json           # Chicago SSL experiment output
│   └── experiment_report.txt      # Unified TXT report (auto-generated)
├── docs/
│   ├── 01_introduction.md
│   └── ROADMAP.md
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone [repository-url]
cd thesis_fairness_cj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
sodapy>=2.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
```

## Usage

### Run all experiments (all 3 datasets)
```bash
python -m src.experiments.run_experiments --all --output results/
```

### Run single dataset
```bash
python -m src.experiments.run_experiments --dataset compas --output results/
python -m src.experiments.run_experiments --dataset communities_crime --output results/
python -m src.experiments.run_experiments --dataset ssl --output results/
```

### Test data loaders
```bash
python -m src.data.data_loader
```

### Output
Each run produces:
- **`{dataset}_results.json`** &mdash; Full experiment results (JSON)
- **`{dataset}_full_models.csv`** &mdash; Model comparison table
- **`{dataset}_reduction_{method}.csv`** &mdash; Feature reduction curves
- **`experiment_report.txt`** &mdash; Unified human-readable report (generated when using `--all`)

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Full Model Evaluation | Logistic Regression, Random Forest, Gradient Boosting on all features |
| 2 | Simple Baseline | 2-feature Logistic Regression (Dressel & Farid 2018 replication) |
| 3 | Feature Reduction | Progressive reduction via MI, RF importance, LASSO |
| 4 | ROAR Ablation | Remove-one-at-a-time feature ablation (LR + RF) |
| 5 | Cross-Validation | 5-fold stratified CV for robustness |

## Dataset Loading

All three datasets are loaded automatically:
- **COMPAS**: Downloaded from ProPublica GitHub
- **Communities & Crime**: Downloaded from UCI ML Repository
- **Chicago SSL**: Loaded from local CSV if available, otherwise downloaded via **Socrata API** (`sodapy`)

## ML Metrics (reported)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| AUC-ROC | Area Under the ROC Curve |
| Precision | Positive predictive value |
| Recall | True positive rate / Sensitivity |
| F1 | Harmonic mean of Precision and Recall |

## Fairness Metrics (reported)

| Metric | Description | Ideal | Threshold |
|--------|-------------|-------|-----------|
| Statistical Parity Difference (SPD) | Difference in positive prediction rates | 0 | \|SPD\| < 0.1 |
| Equal Opportunity Difference (EOD) | Difference in TPR between groups | 0 | \|EOD\| < 0.1 |
| Disparate Impact (DI) | Ratio of positive prediction rates | 1.0 | 0.8 <= DI <= 1.25 |
| Average Odds Difference (AOD) | Average of TPR and FPR differences | 0 | AOD < 0.1 |

## Feature Selection Methods

| Method | Type | Description |
|--------|------|-------------|
| Mutual Information | Filter | Non-linear dependency measure |
| Random Forest Importance | Embedded | Gini-based feature importance |
| LASSO (L1) | Embedded | L1-regularised logistic regression |

## Key References

1. Dressel, J., & Farid, H. (2018). The accuracy, fairness, and limits of predicting recidivism. *Science Advances*.
2. Berk, R., et al. (2021). Fairness in criminal justice risk assessments: The state of the art. *Sociological Methods & Research*.
3. Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*.
4. Ensign, D., et al. (2018). Runaway feedback loops in predictive policing. *FAT* 2018*.

## Author

Alessio Martucci

## License

This project is for academic research purposes.
