import numpy as np
import torch

def run_fusion(sup_pipeline, ae_model, X_val, y_val, preproc, alpha=0.7):
    """
    Fusion of supervised model + Autoencoder
    alpha: weight for supervised model (0.7 means 70% supervised, 30% AE)
    """
    # --- Supervised predictions ---
    sup_probs = sup_pipeline.predict_proba(X_val)
    max_sup_prob = np.max(sup_probs, axis=1)

    # --- Autoencoder reconstruction error ---
    X_val_proc = preproc.transform(X_val)
    with torch.no_grad():
        recon = ae_model(torch.tensor(X_val_proc).float()).numpy()
    ae_errors = ((X_val_proc - recon) ** 2).mean(axis=1)

    # Normalize AE errors between 0 and 1
    ae_errors_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-8)

    # --- Fusion score ---
    fusion_scores = alpha * max_sup_prob + (1 - alpha) * (1 - ae_errors_norm)

    # --- Final prediction ---
    final_preds = (fusion_scores > 0.5).astype(int)  # threshold can be tuned

    return final_preds, fusion_scores

