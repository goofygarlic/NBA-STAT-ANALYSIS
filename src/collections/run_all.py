from collect_players import collect_player_stats
from collect_teams import collect_team_stats
from collect_on_off import collect_on_off
from collect_hustle import collect_hustle_stats

def main():
    print("=" * 50)
    print("NBA ANALYTICS: DATA COLLECTION STAGE")
    print("=" * 50)

    players_df = collect_player_stats
    teams_df = collect_team_stats
    hustle_df = collect_hustle_stats

    player_ids = players_df["PLAYER_ID"].tolist()
    print(f"\n[run_all] Fetching on/off splits for {len(player_ids)} players...")
    print("[run_all] This will take several minutes due to rate limiting.\n")
    on_off_df = collect_on_off(player_ids)

    print("\n" + "=" * 50)
    print("Collection complete.")
    print(f"  Players : {len(players_df)}")
    print(f"  Teams   : {len(teams_df)}")
    print(f"  Hustle  : {len(hustle_df)}")
    print(f"  On/Off  : {len(on_off_df)} rows")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()
 