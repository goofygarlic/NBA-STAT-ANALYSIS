import os
import pandas as pd
from config_historical import (KAGGLE_VOTING_PATH, RAW_HIST_DIR, end_year_to_nba_api, START_YEAR, END_YEAR)

MVP_LABELS_PATH = os.path.join(RAW_HIST_DIR, "mvp_labels.csv")

def load_all_mvp_labels() -> pd.DataFrame:
    if not os.path.exists(KAGGLE_VOTING_PATH):
        raise FileNotFoundError(f"Kaggle voting file not found at {KAGGLE_VOTING_PATH}")
    
    print("[mvp_labels] Loading Kaggle voting data...")
    df = pd.read_csv(KAGGLE_VOTING_PATH)
    
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()
    
    df["SEASON"] = df["year"].apply(end_year_to_nba_api)
    
    df["PLAYER_NAME"] = df["Player"].str.strip()
    df["PLAYER_NORM"] = df["PLAYER_NAME"].str.lower().str.strip()
    
    df = df.rename(columns={
        "Share": "SHARE",
        "First": "FIRST_PLACE_VOTES",
        "Rank": "RANK",
    })
    
    df["MVP_WINNER"] = df["RANK"].astype(str).str.strip() == "1"
    
    df = df[df["SHARE"] > 0].copy()
    
    winners_per_season = df.groupby("SEASON")["MVP_WINNER"].sum()
    bad = winners_per_season[winners_per_season != 1].index.tolist()
    if bad:
        print(f"Warning: unexpected winner count for seasons: {bad}")
        
    
    
    
    keep = ["PLAYER_NAME", "PLAYER_NORM", "SEASON", "SHARE", "FIRST_PLACE_VOTES", "MVP_WINNER", "RANK"]
    df = df[keep].drop_duplicates(subset=["PLAYER_NORM", "SEASON"])
    
    os.makedirs(RAW_HIST_DIR, exist_ok=True)
    df.to_csv(MVP_LABELS_PATH, index=False)
    
    print(f"[mvp_labels] {len(df)} vote rows across" f" {df['SEASON'].nunique()} seasons")
    print(f" Winners : {df['MVP_WINNER'].sum()}")
    print(f" Seasons : {sorted(df['SEASON'].unique())}")
    print(f" -> {MVP_LABELS_PATH}")
    return df