# src/models/evaluate.py
from sklearn.metrics import roc_auc_score, classification_report
from src.data.utils import load_interim
import pandas as pd

def evaluate_model(model):
    """
    Evaluate given model on test_FINAL.pkl if present; otherwise do a quick split.
    Returns a dict of metrics.
    """
    # Try to load test set
    try:
        df_test = load_interim("test_FINAL.pkl")
        if "TARGET" not in df_test.columns:
            raise KeyError("TARGET missing in test file; using train split instead")
        X_test = df_test.drop(columns=["TARGET"])
        y_test = df_test["TARGET"].astype(int)
    except Exception as e:
        # fallback: split train
        print("Could not load test_FINAL.pkl, falling back to train split for evaluation:", e)
        df = load_interim("train_FINAL.pkl")
        if "TARGET" not in df.columns:
            raise KeyError("TARGET column not found (train fallback)")
        X = df.drop(columns=["TARGET"])
        y = df["TARGET"].astype(int)
        # simple 80/20 split
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds) if len(y_test.unique()) > 1 else 0.0
    # optional thresholded metrics
    preds_bin = (preds >= 0.5).astype(int)
    report = classification_report(y_test, preds_bin, output_dict=True, zero_division=0)
    metrics = {
        "auc": float(auc),
        "classification_report": report
    }
    print("Eval AUC:", auc)
    return metrics
