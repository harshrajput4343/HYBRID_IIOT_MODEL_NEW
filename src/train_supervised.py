# src/train_supervised.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

from .data_pipeline import build_preprocessing_pipeline
from .constants import DATA_PATH, TARGET_COL, SEED

def train_supervised():
    # --- Load dataset ---
    df = pd.read_csv(DATA_PATH)

    # --- Split ---
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(y)

    # --- Ensure save directory exists ---
    os.makedirs("results/models", exist_ok=True)

    # save label encoder
    joblib.dump(le, "results/models/label_encoder.joblib")

    # --- Train/validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # --- Preprocessor ---
    preproc, _, _ = build_preprocessing_pipeline(df, TARGET_COL)

    # --- Model ---
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist"   # CPU fast mode
    )

    pipeline = Pipeline([
        ("preproc", preproc),
        ("clf", model)
    ])

    # --- Train ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    preds = pipeline.predict(X_val)
    print(classification_report(y_val, preds, target_names=le.classes_))

    # save model
    joblib.dump(pipeline, "results/models/supervised_xgb.joblib")

    return pipeline
