# src/evaluate.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_and_save(y_true, y_pred, name="results"):
    """
    Evaluate model predictions and save reports and plots.
    Automatically handles string vs numeric label mismatch.
    """
    # Convert to strings if mixed types
    y_true = pd.Series(y_true, dtype=str)
    y_pred = pd.Series(y_pred, dtype=str)

    # Print classification report
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    # Save report to file
    os.makedirs("results/eval", exist_ok=True)
    with open(f"results/eval/{name}_classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()))
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/eval/{name}_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"✅ Evaluation saved to results/eval/{name}_classification_report.txt and PNG")
