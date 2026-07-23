import os
import sys
import time
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, "../collections")
sys.path.insert(0, "../collections/historical")
sys.path.insert(0, "../features")

from config import RAW_DIR, SEASON, REQUEST_DELAY
from config_historical import MODELS_DIR, PROCESSED_DIR
from normalize import min_max

BASE_DIR     = r"C:\Users\karth\Desktop\Projects\NBA-STAT-ANALYSIS"
PLAYERS_PATH = os.path.join(BASE_DIR, "data", "raw", "players_master.csv")
TEAMS_PATH   = os.path.join(BASE_DIR, "data", "raw", "teams_master.csv")
ON_OFF_PATH  = os.path.join(BASE_DIR, "data", "raw", "on_off_splits.csv")
OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "mvp_scores_current.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "outputs", "models", "mvp_model.pkl")

MVP_FEATURES = [ # this should match the features used in train_mvp.py
    "PTS",
    "TS_PCT",
    "USG_PCT",
    "NET_RATING",
    "AST",
    "REB",
    "PIE",
    "W_PCT",
]

def load_current_data() -> pd.DataFrame:
    
    for path, label in [
        (PLAYERS_PATH, "players_master.csv — run src/collections/run_all.py"),
        (TEAMS_PATH,   "teams_master.csv   — run src/collections/run_all.py"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[score] Required file not found:\n  {path}\n"
                f"  Fix: {label}"
            )
 
    players = pd.read_csv(PLAYERS_PATH)
    teams   = pd.read_csv(TEAMS_PATH)
    
    team_wp = teams[["TEAM_ID", "W_PCT"]].copy()
    df = players.merge(team_wp, on="TEAM_ID", how="left", suffixes=("", "_team"))
    
    if "W_PCT_team" in df.columns:
        df["W_PCT"] = df["W_PCT_team"].fillna(df.get("W_PCT", np.nan))
        df = df.drop(columns=["W_PCT_team"])
        
    if os.path.exists(ON_OFF_PATH):
        on_off = pd.read_csv(ON_OFF_PATH)
 
        on  = on_off[on_off["GROUP_VALUE"] == "On"][["PLAYER_ID", "NET_RATING"]]\
                .rename(columns={"NET_RATING": "NET_ON"})
        off = on_off[on_off["GROUP_VALUE"] == "Off"][["PLAYER_ID", "NET_RATING"]]\
                .rename(columns={"NET_RATING": "NET_OFF"})
 
        on_off_diff = on.merge(off, on="PLAYER_ID", how="inner")
        on_off_diff["ON_OFF_DIFF"] = on_off_diff["NET_ON"] - on_off_diff["NET_OFF"]
 
        df = df.merge(
            on_off_diff[["PLAYER_ID", "ON_OFF_DIFF"]],
            on="PLAYER_ID",
            how="left"
        )
        print(f"[score] ON_OFF_DIFF attached for "
              f"{df['ON_OFF_DIFF'].notna().sum()} players")
    else:
        df["ON_OFF_DIFF"] = np.nan
        print("[score] WARNING: on_off_splits.csv not found — "
              "ON_OFF_DIFF will be NaN. Scores are still valid but "
              "less precise. Run collect_on_off.py to fix this.")
 
    return df

def score(df: pd.DataFrame, pipeline, features: list) -> pd.DataFrame:
    available = [f for f in features if f in df.columns]
    missing   = set(features) - set(available)
    if missing:
        print(f"[score] WARNING: Features missing from current data: {missing}")
        print(f"[score] Scoring with {len(available)}/{len(features)} features")
 
    X = df[available].fillna(0).values
    proba = pipeline.predict_proba(X)[:, 1]
 
    df = df.copy()
    df["MVP_PROB"]  = proba
    df["MVP_SCORE"] = min_max(pd.Series(proba, index=df.index)) * 100
    df["RANK"]      = df["MVP_SCORE"].rank(ascending=False, method="min").astype(int)
    
    base_cols = [
        "RANK", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP",
        "PTS", "TS_PCT", "USG_PCT", "NET_RATING",
        "AST", "REB", "PIE", "W_PCT",
    ]
    extra_cols = ["ON_OFF_DIFF"]
    score_cols = ["MVP_PROB", "MVP_SCORE"]
 
    output_cols = (
        base_cols
        + [c for c in extra_cols if c in df.columns]
        + score_cols
    )
    # Keep only cols that exist
    output_cols = [c for c in output_cols if c in df.columns]
 
    return df[output_cols].sort_values("RANK").reset_index(drop=True)

def score_current_season() -> pd.DataFrame:
    print("=" * 50)
    print(f"MVP Likelihood — Scoring {SEASON}")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n[score] Trained model not found at:\n  {MODEL_PATH}\n"
            f"  Fix: run src/models/train_mvp.py first"
        )
 
    artifact  = joblib.load(MODEL_PATH)
    pipeline  = artifact["pipeline"]
    features  = artifact["features"]
    print(f"[score] Model loaded — features: {features}")
    
    print(f"[score] Loading current season data...")
    df = load_current_data()
    print(f"[score] {len(df)} players loaded")
    
    print(f"[score] Scoring...")
    scored = score(df, pipeline, features)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scored.to_csv(OUTPUT_PATH, index=False)
 
    print(f"\n[score] Done. Top 10 MVP candidates — {SEASON}:")
    print("─" * 60)
    top10 = scored.head(10)[["RANK", "PLAYER_NAME", "TEAM_ABBREVIATION",
                               "PTS", "W_PCT", "MVP_SCORE"]]
    for _, row in top10.iterrows():
        print(f"  {int(row['RANK']):>2}. {row['PLAYER_NAME']:<26} "
              f"{row['TEAM_ABBREVIATION']:<4}  "
              f"{row['PTS']:.1f}pts  "
              f"W%:{row['W_PCT']:.3f}  "
              f"Score:{row['MVP_SCORE']:.1f}")
    print("─" * 60)
    print(f"\n[score] Full results -> {OUTPUT_PATH}")
    return scored
 
 
if __name__ == "__main__":
    score_current_season()