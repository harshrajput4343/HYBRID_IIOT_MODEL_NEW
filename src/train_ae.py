import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from src.constants import DATA_PATH, TARGET_COL, SEED

# -----------------------------
# Autoencoder Model
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# -----------------------------
# Smooth Curve Helper
# -----------------------------
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# -----------------------------
# Plot Training Curves
# -----------------------------
def plot_training_curves(history, model_name="autoencoder"):
    os.makedirs("results/plots", exist_ok=True)
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Smooth the loss curves
    train_loss_smooth = smooth_curve(history['train_loss'], factor=0.9)
    val_loss_smooth = smooth_curve(history['val_loss'], factor=0.9)

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_smooth, 'bo-', label="Training loss")
    plt.plot(epochs, val_loss_smooth, 'r-', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("(b) Categorical cross entropy loss vs number of epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_loss.png", dpi=300)
    plt.close()
    print(f"✅ Saved plot: results/plots/{model_name}_loss.png")


# -----------------------------
# Train Autoencoder
# -----------------------------
def train_autoencoder(num_epochs=50, batch_size=128, lr=1e-4):
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower()

    # Keep only normal traffic
    normal_df = df[df[TARGET_COL] == "normal"].drop(columns=[TARGET_COL])

    if normal_df.empty:
        raise ValueError("No normal class found for autoencoder training!")

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(normal_df)
    joblib.dump(scaler, "results/models/ae_preprocessor.joblib")

    # Split train-validation
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=SEED)

    # Convert to tensors
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float()), batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Track loss
    history = {'train_loss': [], 'val_loss': []}

    print("Training started...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch[0])
            loss = criterion(outputs, batch[0])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch[0].size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch[0])
                loss = criterion(outputs, batch[0])
                val_loss += loss.item() * batch[0].size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save model
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/autoencoder_state.pt")
    print("✅ Autoencoder saved at results/models/autoencoder_state.pt")

    # Plot curves
    plot_training_curves(history, model_name="autoencoder")

    return model, scaler
