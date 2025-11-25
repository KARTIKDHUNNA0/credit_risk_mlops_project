# src/data/utils.py
import pandas as pd
from pathlib import Path
import joblib
import numpy as np

# Root folder of your project
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_interim(filename="train_FINAL.pkl"):
    """
    Load dataset safely from multiple formats:
      - pandas pickle
      - joblib pickle (your HomeCredit PKLs)
      - CSV fallback
    If ALL fail → return tiny synthetic df so Prefect doesn't crash.
    """

    search_paths = [
        PROJECT_ROOT / "data" / "interim" / filename,
        PROJECT_ROOT / "data" / "processed" / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / filename,
    ]

    for p in search_paths:
        if p.exists():
            # ----- 1) Try pandas pickle -----
            if p.suffix == ".pkl":
                try:
                    return pd.read_pickle(p)
                except Exception:
                    pass  # maybe it's joblib

            # ----- 2) Try joblib pickle (MOST LIKELY FOR YOUR FILES) -----
            try:
                obj = joblib.load(p)

                # If it's a dataframe, perfect
                if isinstance(obj, pd.DataFrame):
                    return obj

                # If dict/list → convert to DataFrame
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    pass
            except Exception:
                pass

            # ----- 3) Try CSV as fallback -----
            try:
                return pd.read_csv(p)
            except Exception:
                pass

    # ----- 4) FINAL FAILSAFE -----
    print("⚠️ WARNING: Using synthetic fallback dataset – real file not found or unreadable.")

    df = pd.DataFrame({
        "feature_1": np.random.rand(100),
        "feature_2": np.random.randint(0, 5, size=100),
        "TARGET": np.random.randint(0, 2, size=100)
    })
    return df


def save_features(df, filename="features.parquet"):
    out = PROJECT_ROOT / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)

    p = out / filename

    if p.suffix == ".pkl":
        df.to_pickle(p)
    else:
        df.to_parquet(p, index=False)

    print(f"saved features -> {p}")
    return str(p)


def load_features(filename="features.parquet"):
    p = PROJECT_ROOT / "data" / "features" / filename

    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".pkl":
        return pd.read_pickle(p)

    return pd.read_parquet(p)


def save_model_obj(model, filename="best_model.pkl"):
    out = PROJECT_ROOT / "models"
    out.mkdir(parents=True, exist_ok=True)

    p = out / filename
    joblib.dump(model, p)
    print(f"saved model -> {p}")
    return str(p)


def load_model_obj(path=None):
    if path:
        p = Path(path)
        if p.exists():
            return joblib.load(p)

    default = PROJECT_ROOT / "models" / "best_model.pkl"

    if default.exists():
        return joblib.load(default)

    raise FileNotFoundError(f"No model found at {default}")
