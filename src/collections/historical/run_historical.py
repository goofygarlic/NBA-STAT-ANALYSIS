import sys
sys.path.insert(0, "../")

from load_mvp_labels import load_all_mvp_labels
from build_mvp_training_data import collect_historical_stats, build_training_dataset
from config_historical import RAW_HIST_DIR, MVP_LABELS_PATH

import os
import pandas as pd

def main():
    print("=" * 50)
    print("NBA ANALYTICS: HISTORICAL DATA COLLECTION STAGE")
    print("=" * 50)
    
    print("\n[historical] Step 1: Loading MVP labels...")
    labels_df = load_all_mvp_labels()
    
    print("\n[historical] Step 2: Collecting historical stats...")
    stats_df = collect_historical_stats()
    
    print("\n[historical] Step 3: Building training dataset...")
    training_df = build_training_dataset(stats_df, labels_df)
    
    print("\n" + "=" * 50)
    print("Historical data collection complete.")
    print(f"  Player-seasons : {len(training_df)}")
    print(f"  MVP candidates : {training_df['MVP_LABEL'].sum()}")
    print(f"  Seasons        : {training_df['SEASON'].nunique()}")
    print("\nNext step:")
    print("  cd ../../models")
    print("  python train_mvp.py")
    print("=" * 50)
    
if __name__ == "__main__":
    main()