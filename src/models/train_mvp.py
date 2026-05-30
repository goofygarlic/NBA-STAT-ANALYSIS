import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "../collections/historical")
from collections.historical.config_historical import PROCESSED_DIR, MODELS_DIR, TARGET_ROC_AUC

TRAINING_PATH = os.path.join(PROCESSED_DIR, "training_mvp.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "mvp_model.pkl")
VALIDATION_SEASONS = 5 # keep the last 5 seasons for validation

MVP_FEATURES = [ # this should match the features used in build_mvp_training_data.py
    "PTS",
    "TS_PCT",
    "USG_PCT",
    "NET_RATING",
    "AST",
    "REB",
    "PIE",
    "W_PCT",
]