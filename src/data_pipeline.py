# src/data_pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def build_preprocessing_pipeline(df, target_col: str = None):
    """
    Build ColumnTransformer for numeric and categorical features.
    Safely removes target_col if present.
    Returns: preproc, num_cols, cat_cols
    """
    # Identify numeric & categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Remove target if present
    if target_col is not None:
        if target_col in num_cols:
            num_cols.remove(target_col)
        if target_col in cat_cols:
            cat_cols.remove(target_col)

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    return preproc, num_cols, cat_cols
