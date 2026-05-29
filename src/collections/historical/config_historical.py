import os

# Season ranges, API delay
START_YEAR = 2000
END_YEAR = 2024
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


