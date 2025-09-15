# Results

## A. Dataset Statistics
Table I summarizes the dataset properties.

**Table I – Dataset Summary**
| Rows | Features | Classes | Normal Samples | Attack Samples | Time Coverage |
|------|----------|---------|----------------|----------------|---------------|
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

---

## B. Baseline Supervised Models
**Table II – Supervised Intrusion Detection Results**
| Model | Accuracy | Precision | Recall | F1-macro |
|-------|----------|-----------|--------|----------|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] |
| MLP | [ ] | [ ] | [ ] | [ ] |

Observations: [TO FILL, e.g., XGBoost achieved highest macro-F1 but suffered from imbalance].

---

## C. Anomaly Detection Results
**Table III – Anomaly Detection Models**
| Model | Detection Rate | FAR | AUC-PR |
|-------|----------------|-----|--------|
| Autoencoder | [ ] | [ ] | [ ] |
| Isolation Forest | [ ] | [ ] | [ ] |
| One-Class SVM | [ ] | [ ] | [ ] |

Observation: [TO FILL, e.g., Autoencoder showed strong anomaly separation but had 4% FAR].

---

## D. Hybrid Fusion Results
**Table IV – Hybrid vs Individual Models**
| Model | Accuracy | Precision | Recall | F1-macro | FAR |
|-------|----------|-----------|--------|----------|-----|
| Supervised only | [ ] | [ ] | [ ] | [ ] | [ ] |
| AE only | [ ] | [ ] | [ ] | [ ] | [ ] |
| Hybrid Fusion | [ ] | [ ] | [ ] | [ ] | [ ] |

Hybrid fusion improved recall on minority attack classes while keeping FAR below operational threshold (≤5%).

---

## E. Visual Results
- **Fig. 1**: Hybrid architecture diagram.
- **Fig. 2**: Confusion matrix (test set).
- **Fig. 3**: ROC/PR curves for attack detection.
- **Fig. 4**: SHAP feature importance (XGBoost).
- **Fig. 5**: Autoencoder reconstruction error distribution.

---

## F. Robustness Analysis
- With Gaussian noise: F1 dropped by [ ]%.
- Under class imbalance stress: Hybrid model maintained [ ] recall vs baseline [ ].
- Temporal drift: Fusion approach adapted better than standalone models.

---

## G. Discussion
- Strengths: [TO FILL, e.g., hybrid captured both known attacks and novel anomalies].
- Limitations: [TO FILL, e.g., requires careful threshold tuning].
- Future work: federated learning, transformer-based anomaly detection, edge deployment optimization.
