from fastapi.testclient import TestClient
from src.api.app import app
import os

# Initialize Client
# The CI pipeline guarantees the model file exists before this runs.
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
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response