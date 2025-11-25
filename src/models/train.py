# src/models/train.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.data.utils import load_interim

BEST_PARAMS = {
    "learning_rate": 0.0196748929443546,
    "num_leaves": 226,
    "feature_fraction": 0.6531519961265604,
    "bagging_fraction": 0.7796593941777206,
    "bagging_freq": 1,
    "min_data_in_leaf": 113,
}

def train_model():

    df = load_interim("train_FINAL.pkl")

    # 💥 FIX: LightGBM cannot handle special characters in feature names
    df.columns = (
        df.columns
        .str.replace("[^0-9a-zA-Z_]+", "_", regex=True)
    )

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=5000,
        metric="auc",
        random_state=42,
        **BEST_PARAMS
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200)],   # universal fix
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print("Validation AUC:", auc)

    return model
