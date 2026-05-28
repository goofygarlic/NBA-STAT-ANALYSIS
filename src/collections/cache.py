import os
import pandas as pd
from config import CACHE_DIR

def load_or_fetch(filename: str, fetch_fn) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{filename}.csv")

    if(os.path.exists(path)):
        print(f"[cache] Loading {filename} from disk")

    print(f"[cache] Fetching {filename} from API...")
    df = fetch_fn()
    df.to_csv(path, index=False)
    print(f"[cache] Saved -> {path}")
    return df

def bust(filename: str): # Forces a fresh fetch through deletion
    path = os.path.join(CACHE_DIR, f"{filename}.csv")
    if(os.path.exists):
        os.remove(path)
        print(f"[cache] Busted {filename}")