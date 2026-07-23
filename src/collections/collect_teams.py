import time
import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

from config import SEASON, SEASON_TYPE, REQUEST_DELAY, MIN_GAMES, RAW_DIR
from cache import load_or_fetch


def fetch_team_standard(): # fetches regular stats for team
    time.sleep(REQUEST_DELAY)
    return leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

def fetch_team_advanced(): # fetches advanced stats for team
    time.sleep(REQUEST_DELAY)
    return leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]



def collect_team_stats() -> pd.DataFrame:
    standard = load_or_fetch("teams_standard", fetch_team_standard)
    advanced = load_or_fetch("teams_advanced", fetch_team_advanced)

    df = standard.merge(advanced, on="TEAM_ID", suffixes=("", "_adv"))

    os.makedirs(RAW_DIR, exist_ok=True)
    out = os.path.join(RAW_DIR, "teams_master.csv")
    df.to_csv(out, index=False)
    print(f"[collect] Master team file written -> {out} ({len(df)} teams)")
    return df

if __name__ == "__main__":
    collect_team_stats()