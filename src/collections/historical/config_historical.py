import os

# Season ranges, API delay
START_YEAR = 2001
END_YEAR = 2025
NBA_API_DELAY = 0.65

# Target accuracy of models
TARGET_ROC_AUC = 0.90

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
RAW_HIST_DIR = os.path.join(BASE_DIR, "data/raw/historical")
CACHE_HIST_DIR = os.path.join(BASE_DIR, "data/cache/historical")
PROCESSED_DIR = os.path.join(BASE_DIR, "/data/processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs/models")

# Kaggle Voting file (need to combine all of these)
KAGGLE_VOTING_PATH = os.path.join(RAW_HIST_DIR, "mvp_award_shares.csv")


def end_year_to_nba_api(year: int) -> str:
    start = year - 1
    return f"{start}-{str(year)[2:]}"

def all_nba_api_seasons() -> list[str]:
    return [end_year_to_nba_api(y) for y in range(START_YEAR, END_YEAR + 1)]