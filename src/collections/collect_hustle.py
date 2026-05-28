import time
import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashpthustle
from config import SEASON, REQUEST_DELAY, RAW_DIR
from cache import load_or_fetch

def fetch_hustle():
    time.sleep(REQUEST_DELAY)
    return leaguedashpthustle.LeagueDashPtHustle(
        season=SEASON,
        league_id_nullable="00",
        per_mode_time_nullable="PerGame",
    ).get_data_frames()[0]

def collect_hustle_stats() -> pd.DataFrame:
    df = load_or_fetch("players_hustle", fetch_hustle)

    os.makedirs(RAW_DIR, exist_ok=True)
    out = os.path.join(RAW_DIR, "hustle_stats.csv")
    df.to_csv(out, index=False)
    print(f"[hustle] Written -> {out} ({len(df)} players)")
    return df

if __name__ == "__main__":
    collect_hustle_stats()