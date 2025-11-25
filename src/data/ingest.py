# src/data/ingest.py
import shutil
from pathlib import Path

def ingest_files(raw_dir: str, out_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in raw.iterdir():
        if p.is_file() and p.suffix in (".csv", ".parquet", ".pkl"):
            dest = out / p.name
            shutil.copy2(p, dest)
            copied.append(str(dest))
    print("Copied files:", copied)
    return copied
