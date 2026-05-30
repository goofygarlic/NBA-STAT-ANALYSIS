import os
import sys
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from config_historical import (all_nba_api_seasons, NBA_API_DELAY, RAW_HIST_DIR, CACHE_HIST_DIR, PROCESSED_DIR)

HIST_STATS_PATH = os.path.join(RAW_HIST_DIR, "historical_player_stats.csv")
MVP_LABELS_PATH = os.path.join(RAW_HIST_DIR, "mvp_labels.csv")
TRAINING_PATH = os.path.join(PROCESSED_DIR, "training_mvp.csv")

MVP_FEATURES = [ # this can change so easily lol but i'll have it at this for now, depending on what I determine is important from actual mvp votes
    "PTS",
    "TS_PCT",
    "USG_PCT",
    "NET_RATING",
    "AST",
    "REB",
    "PIE",
    "W_PCT",
]

# ALL HISTORICAL STATS

def fetch_season_stats(season: str) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_HIST_DIR, f"mvp_stats_{season}.csv")
    os.makedirs(CACHE_HIST_DIR, exist_ok=True)
    
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    
    print(f"[mvp_data] Fetching {season}...")
    time.sleep(NBA_API_DELAY)
    
    try:
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_simple_nullable="Advanced",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]
        
        time.sleep(NBA_API_DELAY)
        
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_simple_nullable="Base",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]
        
        # FOR WIN PERCENTAGE
        wp = base[["PLAYER_ID", "W_PCT"]].copy()
        df = adv.merge(wp, on="PLAYER_ID", how="left")
        df["SEASON"] = season
        
        df["PLAYER_NORM"] = df["PLAYER_NAME"].str.lower().str.strip()
        
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"[mvp_data] Error fetching {season}: {e}")
        return pd.DataFrame()
    
def collect_historical_stats() -> pd.DataFrame:
    frames = []
    
    for season in all_nba_api_seasons():
        df = fetch_season_stats(season)
        if not df.empty:
            frames.append(df)
            
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    
    os.makedirs(RAW_HIST_DIR, exist_ok=True)
    combined.to_csv(HIST_STATS_PATH, index=False)
    print(f"[mvp_data] Historical stats: {len(combined)} player-seasons " f"across {combined['SEASON'].nunique()} seasons")
    return combined



# Join stats + labels for training data :D

def build_training_dataset(stats_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    df = stats_df.merge(labels_df[["PLAYER_NORM", "SEASON", "SHARE", "MVP_WINNER"]], on=["PLAYER_NORM", "SEASON"], how="left")
    
    df["SHARE"] = df["SHARE"].fillna(0.0)
    df["MVP_LABEL"] = (df["SHARE"] > 0.0).astype(int)
    df["MVP_WINNER"] = df["MVP_WINNER"].fillna(False).astype(bool)
    
    # in case no mvp winner found (should never happen)
    seasons_with_winners = df[df["MVP_WINNER"]]["SEASON"].unique()
    all_seasons = df["SEASON"].unique()
    missing = set(all_seasons) - set(seasons_with_winners)
    if missing: 
        print(f"[mvp_data] WARNING: no MVP winner found for seasons: {sorted(missing)}")
        print(" Check if kaggle file covers these seasons")
        
    # drop rows missing model features
    available_features = [f for f in MVP_FEATURES if f in df.columns]
    df = df.dropna(subset=available_features)
    
    pos_rate = df["MVP_LABEL"].mean() * 100
    print(f"\n[mvp_data] Training dataset summary:")
    print(f"  Total player-seasons : {len(df)}")
    print(f"  MVP vote recipients  : {df['MVP_LABEL'].sum()}")
    print(f"  Positive rate        : {pos_rate:.2f}%")
    print(f"  Seasons covered      : {df['SEASON'].nunique()}")
    print(f"  Features available   : {available_features}")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(TRAINING_PATH, index=False)
    print(f"\n[mvp_data] Training data -> {TRAINING_PATH}")
    return df

def build_mvp_training_data():
    if os.path.exists(HIST_STATS_PATH):
        print(f"[mvp_data] Loading cached historical stats...")
        stats_df = pd.read_csv(HIST_STATS_PATH)
    else:
        stats_df = collect_historical_stats()
    
    labels_df = pd.read_csv(MVP_LABELS_PATH)
    
    return build_training_dataset(stats_df, labels_df)