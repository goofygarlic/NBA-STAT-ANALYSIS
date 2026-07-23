import time
import os
import pandas as pd
from nba_api.stats.endpoints import(
    leaguedashplayerstats,
    leaguedashplayerclutch,
)
from config import SEASON, SEASON_TYPE, REQUEST_DELAY, MIN_GAMES, RAW_DIR
from cache import load_or_fetch

def fetch_with_delay(endpoint_cls, **kwargs) -> pd.DataFrame:
    time.sleep(REQUEST_DELAY)
    return endpoint_cls(**kwargs).get_data_frames()[0]



def fetch_standard(): # fetches regular stats
    return fetch_with_delay(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    )

def fetch_advanced(): # fetches advanced stats
    return fetch_with_delay(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )

def fetch_clutch(): # fetches clutch stats (the final 5 minutes of the 4th quarter or overtime when the score is within 5 points)
    return fetch_with_delay(
        leaguedashplayerclutch.LeagueDashPlayerClutch,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )



def collect_player_stats() -> pd.DataFrame:
    standard = load_or_fetch("players_standard", fetch_standard)
    advanced = load_or_fetch("players_advanced", fetch_advanced)
    clutch = load_or_fetch("players_clutch", fetch_clutch)

    df = standard.merge(advanced, on="PLAYER_ID", suffixes=("", "_adv"))
    df = df.merge(clutch, on="PLAYER_ID", suffixes=("", "_clutch"))
    df = df[df["GP"] >= MIN_GAMES].copy()

    os.makedirs(RAW_DIR, exist_ok=True)
    out = os.path.join(RAW_DIR, "players_master.csv")
    df.to_csv(out, index=False)
    print(f"[collect] Master player file written -> {out} ({len(df)} players)")
    return df

if __name__ == "__main__":
    collect_player_stats()