import time
import os
import pandas as pd
from nba_api.stats.endpoints import playerdashbordbygeneralsplits
from config import SEASON, REQUEST_DELAY, CACHE_DIR, RAW_DIR

def fetch_player_on_off(player_id: int) -> pd.DataFrame:
    time.sleep(REQUEST_DELAY)
    splits = playerdashbordbygeneralsplits.PlayerDashboardByGeneralSplits(
        player_id = player_id,
        season = SEASON,
        measure_type_simple_nullable="Advanced",
    ).get_data_frames()

    return splits[1] if len(splits) > 1 else pd.DataFrame() # splits[1] contains on/off data

def collect_on_off(player_ids: list) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = []

    for i, pid in enumerate(player_ids):
        cache_file = os.path.join(CACHE_DIR, f"on_off{pid}.csv")


        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
        else:
            print(f"[on_off] Fetching player id {pid} ({i+1}/{len(player_ids)})")
            df = fetch_player_on_off(pid)
            if not df.empty:
                df.to_csv(cache_file, index=False)

        if not df.empty:
            results.append(df)
    
    if not results:
        return pd.DataFrame()
    
    combined = pd.concat(results, ignore_index=True)
    out = os.path.join(RAW_DIR, "on_off_splits.csv")
    combined.to_csv(out, index=False)
    print(f"[on_off] Written -> {out}")
    return combined            