from fastapi import FastAPI
import pandas as pd

from .model import model, feature_columns

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def home():
    return {"message": "Credit Risk Model is running!"}

@app.post("/predict")
def predict(payload: dict):
    data = payload["data"]

    # Put into DataFrame
    df = pd.DataFrame([data])

    # ensure all columns present
    for col in feature_columns:
        if col not in df:
            df[col] = 0  # default value if missing

    df = df[feature_columns]  # order cols

    # predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
