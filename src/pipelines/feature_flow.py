# src/pipelines/feature_flow.py
from prefect import flow, task
from src.features.build_features import build_features
from src.data.utils import save_features, load_interim

@task
def create_features():
    # build_features will load train_FINAL.pkl when called without df
    df = build_features()
    return df

@task
def persist_features(df):
    path = save_features(df, filename="features.parquet")
    return path

@flow(name="feature-flow")
def feature_flow():
    df = create_features()
    p = persist_features(df)
    print("feature pipeline finished, saved to:", p)
    return p

if __name__ == "__main__":
    feature_flow()
