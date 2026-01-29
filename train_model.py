# ==========================================
# Iris Classification Pipeline (FULL CODE)
# - Fixes sklearn error: no multi_class arg
# - Uses RandomForestClassifier (not Regressor)
# - Correct accuracy + metrics
# - Saves confusion matrices + feature importance
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
    accuracy_score
)

sns.set(style="white")


# -------------------------------
# Load Data
# -------------------------------
dataset = pd.read_csv('iris.csv')

# Clean column names (handles " (cm)" and spaces)
dataset.columns = [col.strip().replace(" (cm)", "").replace(" ", "_") for col in dataset.columns]

# Ensure target exists
if "target" not in dataset.columns:
    raise ValueError("Column 'target' not found. Your iris.csv must contain a 'target' column (0/1/2).")


# -------------------------------
# Feature Engineering (safe divide)
# -------------------------------
eps = 1e-8
dataset["sepal_length_width_ratio"] = dataset["sepal_length"] / (dataset["sepal_width"] + eps)
dataset["petal_length_width_ratio"] = dataset["petal_length"] / (dataset["petal_width"] + eps)

# Select Features
dataset = dataset[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "sepal_length_width_ratio",
        "petal_length_width_ratio",
        "target",
    ]
]


# -------------------------------
# Split Data (stratify keeps class balance)
# -------------------------------
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset["target"]
)

X_train = train_data.drop("target", axis=1).astype("float32")
y_train = train_data["target"].astype("int32")

X_test = test_data.drop("target", axis=1).astype("float32")
y_test = test_data["target"].astype("int32")


# -------------------------------
# Logistic Regression (Version-safe)
# NOTE: multi_class removed to avoid old-sklearn error
# -------------------------------
logreg = LogisticRegression(
    C=1.0,          # better default than 0.0001 (0.0001 often underfits)
    solver="lbfgs",
    max_iter=500
)

logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, pred_lr)
acc_train_lr = accuracy_score(y_train, logreg.predict(X_train)) * 100
acc_test_lr = accuracy_score(y_test, pred_lr) * 100

f1_lr = f1_score(y_test, pred_lr, average="macro")
prec_lr = precision_score(y_test, pred_lr, average="macro")
recall_lr = recall_score(y_test, pred_lr, average="macro")


# -------------------------------
# Random Forest Classifier (Correct)
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=44
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, pred_rf)
acc_train_rf = accuracy_score(y_train, rf.predict(X_train)) * 100
acc_test_rf = accuracy_score(y_test, pred_rf) * 100

f1_rf = f1_score(y_test, pred_rf, average="macro")
prec_rf = precision_score(y_test, pred_rf, average="macro")
recall_rf = recall_score(y_test, pred_rf, average="macro")


# -------------------------------
# Confusion Matrix Plot Function
# -------------------------------
def plot_cm(cm, target_names, title="Confusion Matrix", normalize=True, save_path=None):
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("Blues")

    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    else:
        cm_plot = cm

    plt.imshow(cm_plot, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm_plot.max() / 1.5
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        text_val = f"{cm_plot[i, j]:0.4f}" if normalize else f"{cm[i, j]}"
        plt.text(
            j, i, text_val,
            horizontalalignment="center",
            color="white" if cm_plot[i, j] > thresh else "black"
        )

    acc = np.trace(cm) / np.sum(cm)
    plt.xlabel(f"Predicted Label (accuracy={acc:0.4f})")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=140)
    plt.show()


# Plot + Save confusion matrices
target_names = np.array(["setosa", "versicolor", "virginica"])

plot_cm(cm_lr, target_names, title="Confusion Matrix (Logistic Regression)", normalize=True,
        save_path="ConfusionMatrix_LogReg.png")

plot_cm(cm_rf, target_names, title="Confusion Matrix (Random Forest)", normalize=True,
        save_path="ConfusionMatrix_RF.png")


# -------------------------------
# Feature Importance Plot (RF)
# -------------------------------
importances = rf.feature_importances_
labels = X_train.columns

feature_df = pd.DataFrame({"feature": labels, "importance": importances}).sort_values(
    by="importance", ascending=False
)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=14)
ax.set_ylabel("Feature", fontsize=14)
ax.set_title("Random Forest Feature Importances", fontsize=14)
plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=140)  # save BEFORE show
plt.show()


# -------------------------------
# Print Results
# -------------------------------
print("\n========== Logistic Regression ==========")
print(f"Train Accuracy: {acc_train_lr:.2f}%")
print(f"Test  Accuracy: {acc_test_lr:.2f}%")
print(f"F1 (macro):     {f1_lr:.4f}")
print(f"Precision:      {prec_lr:.4f}")
print(f"Recall:         {recall_lr:.4f}")
print("Confusion Matrix:\n", cm_lr)

print("\n========== Random Forest ==========")
print(f"Train Accuracy: {acc_train_rf:.2f}%")
print(f"Test  Accuracy: {acc_test_rf:.2f}%")
print(f"F1 (macro):     {f1_rf:.4f}")
print(f"Precision:      {prec_rf:.4f}")
print(f"Recall:         {recall_rf:.4f}")
print("Confusion Matrix:\n", cm_rf)


# -------------------------------
# Save Scores to File
# -------------------------------
with open("scores.txt", "w") as score:
    score.write("Random Forest (Classifier)\n")
    score.write("Train Accuracy: %2.2f%%\n" % acc_train_rf)
    score.write("Test Accuracy:  %2.2f%%\n" % acc_test_rf)
    score.write("F1 (macro):     %0.4f\n" % f1_rf)
    score.write("Recall (macro): %0.4f\n" % recall_rf)
    score.write("Precision(macro): %0.4f\n" % prec_rf)

    score.write("\n\n")

    score.write("Logistic Regression\n")
    score.write("Train Accuracy: %2.2f%%\n" % acc_train_lr)
    score.write("Test Accuracy:  %2.2f%%\n" % acc_test_lr)
    score.write("F1 (macro):     %0.4f\n" % f1_lr)
    score.write("Recall (macro): %0.4f\n" % recall_lr)
    score.write("Precision(macro): %0.4f\n" % prec_lr)

print("\nSaved files: scores.txt, ConfusionMatrix_LogReg.png, ConfusionMatrix_RF.png, FeatureImportance.png")

