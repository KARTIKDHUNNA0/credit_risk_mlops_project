from fastapi.testclient import TestClient
from src.api.app import app
import numpy as np
import pandas as pd
import joblib
import os

# 1. Setup a Dummy Model for CI/CD (Since real model is on local D drive)
# We create this fixture to ensure the API starts up successfully in GitHub Actions
def setup_module(module):
    os.makedirs("models", exist_ok=True)
    dummy_model_path = "best_optuna_model.txt"
    
    # Create a dummy class that mimics the LightGBM/Sklearn model
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X)) # Return 0s
        def predict_proba(self, X):
            # Return [[0.9, 0.1]] format
            return np.array([[0.9, 0.1]] * len(X))

    # Save dummy model if real one doesn't exist (which is true in CI)
    if not os.path.exists(dummy_model_path):
        joblib.dump(DummyModel(), dummy_model_path)

# 2. Initialize Client
client = TestClient(app)

def test_health_check():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk Model is running!"}

def test_prediction_flow():
    """Test the predict endpoint with dummy data."""
    payload = {
        "data": {
            "AMT_INCOME_TOTAL": 50000,
            "DAYS_EMPLOYED": -1000,
            # Add other features your model expects if validation is strict
        }
    }
    response = client.post("/predict", json=payload)
    
    # Just check if we get a valid 200 response
    # The prediction value doesn't matter (it's dummy), just the code path.
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response