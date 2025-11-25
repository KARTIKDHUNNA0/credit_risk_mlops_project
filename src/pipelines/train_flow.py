# src/pipelines/train_flow.py
from prefect import flow, task
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.predict import save_model

@task
def train():
    model = train_model()
    return model

@task
def evaluate(model):
    metrics = evaluate_model(model)
    return metrics

@task
def persist(model):
    path = save_model(model)
    return path

@flow(name="train-flow")
def train_flow():
    model = train()
    metrics = evaluate(model)
    path = persist(model)
    print("Training completed. model saved to:", path)
    return metrics

if __name__ == "__main__":
    train_flow()
