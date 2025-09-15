# src/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output folder exists
os.makedirs("results/plots", exist_ok=True)

def plot_training_curves(history, model_name="model"):
    """
    Generate accuracy and loss plots similar to reference graphs.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing lists for 'train_acc', 'val_acc', 'train_loss', 'val_loss'
    model_name : str
        Used to save the plots with model-specific names
    """

    # Validate the input dictionary
    required_keys = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing key '{key}' in history dictionary")

    # Generate epoch numbers
    epochs = np.arange(1, len(history['train_acc']) + 1)

    # ============ Accuracy Plot ============
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], 'bo-', label="Training acc")
    plt.plot(epochs, history['val_acc'], 'r-', label="Validation acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("(a) Variation of accuracy with epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_accuracy.png", dpi=300)
    plt.close()

    # ============ Loss Plot ============
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], 'bo-', label="Training loss")
    plt.plot(epochs, history['val_loss'], 'r-', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("(b) Categorical cross entropy loss vs number of epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_loss.png", dpi=300)
    plt.close()

    print(f"✅ Plots saved in results/plots/ as {model_name}_accuracy.png and {model_name}_loss.png")
