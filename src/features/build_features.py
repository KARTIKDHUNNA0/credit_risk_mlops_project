# src/features/build_features.py
import pandas as pd
from pathlib import Path
from src.data.utils import load_interim

def build_features(df=None):
    """
    Minimal feature engineering so pipeline can run end-to-end.
    - If df is None, load train_FINAL.pkl (or fallback).
    - Fill simple missing values, encode categoricals via factorize.
    - Return dataframe with TARGET column preserved if present.
    """
    if df is None:
        # default will load train_FINAL.pkl or fallback synthetic
        df = load_interim("train_FINAL.pkl")

    # Work on a copy
    df = df.copy()

    # Basic cleaning and simple featurization:
    # - Drop unused columns if any obviously huge (none by default)
    # - Fill numeric na with median, categorical with 'MISSING'
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Keep target if present
    target_col = "TARGET"
    if target_col in df.columns:
        num_cols = [c for c in num_cols if c != target_col]

    # Fill numerics
    for c in num_cols:
        med = df[c].median() if df[c].notna().any() else 0
        df[c] = df[c].fillna(med)

    # Fill and encode categoricals with factorize (simple and robust)
    for c in cat_cols:
        df[c] = df[c].fillna("MISSING")
        df[c] = pd.factorize(df[c])[0].astype("int64")

    # If too many columns, optionally reduce — for now keep all
    # Return dataframe
    return df
