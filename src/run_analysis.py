"""
Assignment II: EDA and Visualization of Wisconsin Diagnostic Breast Cancer Dataset
Run from the project root:
    python src/run_analysis.py
The script uses the included CSV and writes charts/metrics to visualizations/ and report/.
This is an educational classification case study, not a clinical diagnostic tool.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "breast_cancer_wisconsin.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
if "target" not in df.columns:
    raise ValueError("Expected a 'target' column in the dataset.")
X, y = df.drop(columns="target"), df["target"].astype(int)
# Target convention: 0 = malignant, 1 = benign.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
# Fit scaler only on training data to prevent data leakage.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s, X_test_s = scaler.transform(X_val), scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                      alpha=0.001, learning_rate_init=0.001, max_iter=500,
                      early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=25, random_state=42)
model.fit(X_train_s, y_train)
val_pred, test_pred = model.predict(X_val_s), model.predict(X_test_s)
malignant_idx = list(model.classes_).index(0)
test_prob = model.predict_proba(X_test_s)[:, malignant_idx]
sns.set_theme()
# 1. Class distribution
fig, ax = plt.subplots(figsize=(7,4))
counts = df["target"].map({0:"Malignant",1:"Benign"}).value_counts().reindex(["Malignant","Benign"])
counts.plot(kind="bar", ax=ax); ax.set(title="Class Distribution in the Dataset", xlabel="Diagnosis", ylabel="Number of cases")
ax.tick_params(axis="x", rotation=0)
for c in ax.containers: ax.bar_label(c)
fig.tight_layout(); fig.savefig(OUT/"01_class_distribution.png", dpi=160); plt.close(fig)
# 2. Feature distribution
fig, ax = plt.subplots(figsize=(7,4))
sns.histplot(data=df, x="mean radius", hue="target", bins=25, ax=ax)
ax.set(title="Distribution of Mean Radius by Diagnosis", xlabel="Mean radius", ylabel="Count")
fig.tight_layout(); fig.savefig(OUT/"02_mean_radius_distribution.png", dpi=160); plt.close(fig)
# 3. Outlier/spread view
fig, ax = plt.subplots(figsize=(7,4))
sns.boxplot(data=df.assign(Diagnosis=df["target"].map({0:"Malignant",1:"Benign"})), x="Diagnosis", y="mean area", ax=ax)
ax.set(title="Mean Area by Diagnosis", xlabel="Diagnosis", ylabel="Mean area")
fig.tight_layout(); fig.savefig(OUT/"03_mean_area_boxplot.png", dpi=160); plt.close(fig)
# 4. Correlations
corr_cols = ["mean radius","mean texture","mean perimeter","mean area","mean smoothness",
             "mean compactness","mean concavity","mean concave points","target"]
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap of Selected Features")
fig.tight_layout(); fig.savefig(OUT/"04_correlation_heatmap.png", dpi=160); plt.close(fig)
# 5. Relationship
fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(data=df.assign(Diagnosis=df["target"].map({0:"Malignant",1:"Benign"})),
                x="mean radius", y="mean concave points", hue="Diagnosis", alpha=.75, ax=ax)
ax.set(title="Mean Radius vs Mean Concave Points", xlabel="Mean radius", ylabel="Mean concave points")
fig.tight_layout(); fig.savefig(OUT/"05_radius_concave_scatter.png", dpi=160); plt.close(fig)
# 6. Training loss
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(range(1, len(model.loss_curve_)+1), model.loss_curve_)
ax.set(title="Training Loss vs Iteration", xlabel="Iteration", ylabel="Loss")
fig.tight_layout(); fig.savefig(OUT/"06_loss_curve.png", dpi=160); plt.close(fig)
# 7. Internal early-stopping validation accuracy
fig, ax = plt.subplots(figsize=(7,4))
scores = getattr(model, "validation_scores_", None)
if scores:
    ax.plot(range(1, len(scores)+1), scores)
ax.set(title="Validation Accuracy During Training", xlabel="Iteration", ylabel="Internal validation accuracy")
fig.tight_layout(); fig.savefig(OUT/"07_validation_accuracy.png", dpi=160); plt.close(fig)
# 8. Test confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=[0,1])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant","Benign"], yticklabels=["Malignant","Benign"], ax=ax)
ax.set(title="Confusion Matrix — Test Set", xlabel="Predicted label", ylabel="Actual label")
fig.tight_layout(); fig.savefig(OUT/"08_confusion_matrix.png", dpi=160); plt.close(fig)
# 9. ROC/AUC for malignant class
fpr, tpr, _ = roc_curve((y_test==0).astype(int), test_prob)
auc = roc_auc_score((y_test==0).astype(int), test_prob)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax.plot([0,1],[0,1], "--", label="Chance")
ax.set(title="ROC Curve — Malignant Class", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"09_roc_curve.png", dpi=160); plt.close(fig)

# 10. Continuous distribution analysis for mean radius
from scipy import stats
radius = df["mean radius"].dropna().to_numpy()
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(radius, bins=25, density=True, alpha=.65, edgecolor="black")
grid = np.linspace(radius.min(), radius.max(), 300)
ax.plot(grid, stats.gaussian_kde(radius)(grid), linewidth=2)
ax.set(title="Distribution of Mean Radius with KDE", xlabel="Mean radius", ylabel="Density")
fig.tight_layout(); fig.savefig(OUT/"10_mean_radius_distribution_kde.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(6,5))
stats.probplot(radius, dist="norm", plot=ax)
ax.set(title="Normal Q–Q Plot: Mean Radius", xlabel="Theoretical normal quantiles", ylabel="Ordered sample quantiles")
fig.tight_layout(); fig.savefig(OUT/"11_mean_radius_qq_plot.png", dpi=160); plt.close(fig)
shapiro_result = stats.shapiro(radius)
distribution_results = {
    "feature": "mean radius", "count": int(len(radius)),
    "mean": float(np.mean(radius)), "median": float(np.median(radius)),
    "std": float(np.std(radius, ddof=1)), "skewness": float(stats.skew(radius, bias=False)),
    "excess_kurtosis": float(stats.kurtosis(radius, bias=False)),
    "shapiro_wilk_statistic": float(shapiro_result.statistic),
    "shapiro_wilk_p_value": float(shapiro_result.pvalue),
    "interpretation": ("Reject the normality null hypothesis at alpha=0.05." if shapiro_result.pvalue < 0.05 else "Do not reject the normality null hypothesis at alpha=0.05.")
}
(ROOT/"report"/"distribution_analysis.json").write_text(json.dumps(distribution_results, indent=2))

results = {
    "dataset_rows": int(len(df)), "feature_count": int(X.shape[1]),
    "train_rows": int(len(X_train)), "validation_rows": int(len(X_val)), "test_rows": int(len(X_test)),
    "missing_values": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
    "malignant_cases": int((y==0).sum()), "benign_cases": int((y==1).sum()),
    "validation_accuracy": float(accuracy_score(y_val, val_pred)),
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "malignant_precision": float(precision_score(y_test, test_pred, pos_label=0)),
    "malignant_recall": float(recall_score(y_test, test_pred, pos_label=0)),
    "malignant_f1": float(f1_score(y_test, test_pred, pos_label=0)),
    "malignant_roc_auc": float(auc), "training_iterations": int(model.n_iter_),
    "target_mapping": {"0":"malignant","1":"benign"}
}
(ROOT/"report"/"metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
