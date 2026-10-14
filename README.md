# Exploratory Data Analysis and Visualization of the Wisconsin Diagnostic Breast Cancer Dataset

## Project summary
This case study explores a real-world diagnostic dataset, performs statistical EDA, creates five core EDA visualizations, and trains a Multilayer Perceptron (MLP) neural network classifier. It also produces four model-evaluation visualizations.

**Educational use only:** this model is not a medical device and must not be used to diagnose, rule out, or make treatment decisions about cancer.

## Dataset
- Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
- Rows: 569
- Input features: 30
- Target: `target` (0 = malignant, 1 = benign)
- Dataset source/documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Original UCI repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Included CSV: `data/breast_cancer_wisconsin.csv`

The dataset contains measurements computed from digitized images of fine needle aspirate (FNA) samples. It is a diagnostic benchmark dataset, not a population screening sample.

## Project structure
- `data/` — included dataset CSV
- `src/run_analysis.py` — reproducible EDA, preprocessing, MLP training, evaluation, and plot generation
- `visualizations/` — generated EDA and model-evaluation charts
- `report/project_report.pdf` — assignment report
- `report/metrics.json` — machine-readable model results from the run
- `report/distribution_analysis.json` — summary statistics and Shapiro–Wilk normality test for mean radius
- Additional charts: `visualizations/10_mean_radius_distribution_kde.png` and `visualizations/11_mean_radius_qq_plot.png`
- `requirements.txt` — Python dependencies

## Run the project
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/run_analysis.py
```

The script writes charts to `visualizations/` and results to `report/metrics.json`.

## Methodology
1. Load and inspect the dataset.
2. Check missing values, duplicate rows, data types, class distribution and summary statistics.
3. Create stratified train/validation/test splits (70%/15%/15%).
4. Fit `StandardScaler` only on the training partition to avoid leakage.
5. Train an MLP classifier with hidden layers (64, 32), ReLU activation, Adam optimizer, L2 regularization and early stopping.
6. Evaluate on the held-out test set using accuracy, malignant-class precision/recall/F1 and ROC-AUC.

## Limitations
- The dataset is relatively small and comes from a specific diagnostic context.
- Results depend on the fixed random split and model settings.
- A held-out test score is not proof of clinical readiness or generalization to other hospitals/populations.
- Feature correlations do not establish causation.
