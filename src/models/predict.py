# src/models/predict.py
from src.data.utils import save_model_obj

def save_model(model, filename="best_model_lgbm.pkl"):
    """
    Save model to models/ directory using joblib via utils.
    """
    return save_model_obj(model, filename=filename)
