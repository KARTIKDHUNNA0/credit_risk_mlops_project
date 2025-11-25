import joblib
import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_lgbm.pkl"

# sanitize feature names (LGBM hates special chars)
def clean_feature_names(columns):
    cleaned = []
    for c in columns:
        new = re.sub(r'[^A-Za-z0-9_]+', '_', c)
        cleaned.append(new)
    return cleaned

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# extract & clean feature names from training dataset
TRAIN_DF_PATH = PROJECT_ROOT / "data" / "processed" / "train_FINAL.pkl"
train_final=joblib.load(TRAIN_DF_PATH)
df_train = pd.DataFrame(train_final)

df_train.columns = clean_feature_names(df_train.columns)
feature_columns = [c for c in df_train.columns if c != "TARGET"]
