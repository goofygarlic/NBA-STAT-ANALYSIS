import os
import pandas as pd
from config_historical import (KAGGLE_VOTING_PATH, RAW_HIST_DIR, end_year_to_nba_api, START_YEAR, END_YEAR)

MVP_LABELS_PATH = os.path.join(RAW_HIST_DIR, "mvp_labels.csv")

def load_mvp_labels() -> pd.DataFrame:
    if not os.path.exists(KAGGLE_VOTING_PATH):
        raise FileNotFoundError(
            f"\n[mvp_labels] Voting file not found at:\n"
            f"  {KAGGLE_VOTING_PATH}\n"
            f"Place mvp_award_shares.csv in data/raw/historical/"
        )
        
    print("[mvp_labels] Loading voting file...")
    df = pd.read_csv(KAGGLE_VOTING_PATH)
    
    # filter for only mvp (this can change easily for other awards :D)
    df = df[df["award"] == "nba mvp"].copy()
    
    df = df[(df["season"] >= START_YEAR) & (df["season"] <= END_YEAR)].copy()
    
    df["SEASON"] = df["season"].apply(end_year_to_nba_api)

    df["PLAYER_NAME"] = df["player"].str.strip()
    df["PLAYER_NORM"] = df["PLAYER_NAME"].str.lower().str.strip()
    
    df["PLAYER_ID_BBREF"] = df["player_id"]
    
    df = df.rename(columns={
        "share": "SHARE",
        "first": "FIRST_PLACE_VOTES",
        "winner": "MVP_WINNER",
    })
    
    df["MVP_WINNER"] = df["MVP_WINNER"].astype(bool)
    
    df = df[df["SHARE"] > 0].copy()
    
    winners_per_season = df.groupby("SEASON")["MVP_WINNER"].sum()
    bad = winners_per_season[winners_per_season != 1].index.tolist()
    if bad:
        print(f"[mvp_labels] WARNING: Unexpected winner count for: {bad}")
        
    keep = ["PLAYER_NAME", "PLAYER_NORM", "PLAYER_ID_BBREF", "SEASON", "SHARE", "FIRST_PLACE_VOTES", "MVP_WINNER"]
    df = df[keep].drop_duplicates(subset=["PLAYER_NORM", "SEASON"])
    
    os.makedirs(RAW_HIST_DIR, exist_ok=True)
    df.to_csv(MVP_LABELS_PATH, index=False)
    
    print(f"[mvp_labels] {len(df)} vote rows across {df['SEASON'].nunique()} seasons")
    print(f"  Winners : {df['MVP_WINNER'].sum()}")
    print(f"  Range   : {df['SEASON'].min()} -> {df['SEASON'].max()}")
    print(f"  -> {MVP_LABELS_PATH}")
    return df