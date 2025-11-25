# src/pipelines/ingest_flow.py
from prefect import flow, task
from pathlib import Path
from src.data.utils import load_interim, save_features

@task
def check_train_exists():
    try:
        df = load_interim("train_FINAL.pkl")
        print("train_FINAL loaded; rows:", len(df))
        return True
    except Exception as e:
        print("train_FINAL not found or unreadable:", e)
        return False

@task
def check_test_exists():
    try:
        df = load_interim("test_FINAL.pkl")
        print("test_FINAL loaded; rows:", len(df))
        return True
    except Exception as e:
        print("test_FINAL not found or unreadable:", e)
        return False

@flow(name="ingest-flow")
def ingest_flow():
    t_ok = check_train_exists()
    s_ok = check_test_exists()
    print("ingest step completed. train:", t_ok, "test:", s_ok)
    return {"train": t_ok, "test": s_ok}

if __name__ == "__main__":
    ingest_flow()
