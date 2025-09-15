# src/stacking.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class HybridFusion:
    def __init__(self):
        self.meta_model = LogisticRegression()

    def fit(self, sup_probs, anomaly_scores, y):
        X_meta = np.hstack([sup_probs, anomaly_scores.reshape(-1,1)])
        self.meta_model.fit(X_meta, y)

    def predict(self, sup_probs, anomaly_scores):
        X_meta = np.hstack([sup_probs, anomaly_scores.reshape(-1,1)])
        return self.meta_model.predict(X_meta)
