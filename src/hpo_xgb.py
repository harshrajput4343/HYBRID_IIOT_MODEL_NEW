# src/hpo_xgb.py
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from .data_pipeline import build_preprocessing_pipeline
from .constants import DATA_PATH, TARGET_COL, SEED

def objective(trial):
    df = pd.read_csv(DATA_PATH)
    preproc, _, _ = build_preprocessing_pipeline(df, TARGET_COL)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0)
    }

    model = Pipeline([("preproc", preproc), ("clf", XGBClassifier(random_state=SEED, **params))])
    score = cross_val_score(model, X, y, scoring="f1_macro", cv=3).mean()
    return score

def run_hpo():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best Params:", study.best_params)
    return study.best_params
