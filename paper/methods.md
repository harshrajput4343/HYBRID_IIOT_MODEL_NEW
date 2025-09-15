# Methods

## A. Dataset
We use the **TON-IoT dataset** (`train_test_network.csv`) as our primary benchmark.  
- Number of rows: [TO FILL AFTER 00_quick_scan].  
- Number of features: [TO FILL].  
- Time coverage: [TO FILL if timestamp available].  
- Class distribution: Normal = [N], Attack = [N], detailed breakdown in Table I.

**Preprocessing applied:**
1. Removed duplicate rows.
2. Missing values imputed (median for numeric, most frequent for categorical).
3. Normalization: all categorical labels standardized (`BENIGN → Normal`).
4. Scaling: StandardScaler applied to numeric features.
5. One-hot encoding applied to categorical features.

## B. Exploratory Data Analysis
EDA revealed:
- Feature distributions were highly skewed [cite results from 01_EDA].
- Correlations showed [TO FILL — e.g., packets vs bytes highly correlated].
- Class imbalance: [TO FILL ratio].

## C. Preprocessing Pipeline
We implemented a reproducible **scikit-learn ColumnTransformer pipeline**:
- Numeric: Median imputation + StandardScaler.
- Categorical: Mode imputation + OneHotEncoder.
- Feature engineering: [TO FILL if added — rolling stats, ratios, etc.].

This pipeline was applied consistently to train/val/test splits to avoid data leakage.

## D. Supervised Intrusion Detection Models
We evaluated classical ML baselines and a simple deep MLP:
- Logistic Regression, Decision Tree, Random Forest.
- XGBoost (best-performing tabular model).
- MLP with [TO FILL layers/units/activations].

Training:
- 80/20 stratified split (train/validation).
- Random seed fixed to 42 for reproducibility.
- Optimizer & hyperparameters: [TO FILL].

## E. Anomaly Detection Models
To detect unknown or novel attacks, we trained unsupervised models:
- **Autoencoder (dense)**: architecture `Input → 128 → 64 → 16 (bottleneck) → 64 → 128 → Output`.
- Loss: Mean Squared Error (MSE).
- Trained only on "Normal" samples.
- Threshold for anomaly detection set at 95th percentile of reconstruction error.

Alternative detectors tested:
- Isolation Forest.
- One-Class SVM.

## F. Hybrid Fusion
We combine the strengths of supervised and unsupervised models:
- Input 1: Supervised classifier probabilities.
- Input 2: Autoencoder reconstruction error.
- Meta-classifier: Logistic Regression (stacking).
- Decision logic: Predict intrusion class if supervised confidence is high; otherwise, predict anomaly if AE error exceeds threshold.

Architecture diagram shown in Fig. 1.

## G. Hyperparameter Optimization
- Tool: Optuna.
- Objective: Maximize macro-F1 while penalizing false positives.
- Trials: 50.
- Search space: `n_estimators, max_depth, learning_rate, subsample`.

## H. Evaluation Protocol
Metrics:
- **Classification**: Accuracy, Precision, Recall, F1-score (per-class and macro).
- **Anomaly detection**: Detection rate, False Alarm Rate (FAR), AUC-PR.
- **Operational**: Inference latency, model size (MB).
- **Robustness tests**: Noise injection, class imbalance stress, temporal drift.

All experiments repeated 3 times with fixed random seeds; mean values reported.
