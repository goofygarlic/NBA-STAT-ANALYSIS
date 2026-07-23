import os
## CONFIG FOR DATA

SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
REQUEST_DELAY = 0.65
MIN_GAMES = 20
BASE_DIR      = r"C:\Users\karth\Desktop\Projects\NBA-STAT-ANALYSIS"
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
CACHE_DIR     = os.path.join(BASE_DIR, "data", "cache")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
