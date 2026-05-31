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

def get_model_configs() -> list:
    return [
        {
            "name": "Logistic Regression (C=1.0)",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ))
            ])
        },
        {
            "name": "Logistic Regression (C=0.1, stronger regularization)",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    C=0.1,
                    max_iter=1000,
                    random_state=42,
                ))
            ])
        },
        {
            "name": "XGBoost",
            "pipeline": _build_xgboost_pipeline(),
        },
    ]


def _build_xgboost_pipeline():
    try:
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=20,  # handles class imbalance like class_weight="balanced"
                eval_metric="auc",
                random_state=42,
                verbosity=0,
            ))
        ])
    except ImportError:
        print(f"[train_mvp] xgboost not installed, skipping XGBoost config")
        return None
    

# TRAINING TIME :DDD

def train(df: pd.DataFrame) -> tuple:
    available = [f for f in MVP_FEATURES if f in df.columns]
    if len(available) < len(MVP_FEATURES):
        missing = set(MVP_FEATURES) - set(available)
        print(f"[train_mvp] WARNING: Missing Features: {missing}")
        print(f"[train_mvp] Training with available features: {available}")

    X = df[available].fillna(0).values
    Y = df["MVP_LABEL"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_pipeline = None
    best_auc = 0.0

    for config in get_model_configs():
        if config["pipeline"] is None:
            continue

        print(f"\n[train_mvp] Trying: {config['name']}")
        scores = cross_val_score(
            config["pipeline"], X, Y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        mean_auc = scores.mean()
        std_auc = scores.std()
        print(f" Cross-val ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_pipeline = config["pipeline"]
            best_name = config["name"]

        if mean_auc >= TARGET_ROC_AUC:
            print(f"\n[train_mvp] ✓ Accuracy gate cleared ({mean_auc:.4f} >= {TARGET_ROC_AUC})")
            print(f"[train_mvp] Using: {config['name']}")
            break
        else:
            print(f"\n[train_mvp] Accuracy gate not reached. Best: {best_auc:.4f}")
            print(f"[train_mvp] Proceeding with best model: {best_name}")

        best_pipeline.fit(X, Y)

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"pipeline": best_pipeline, "features": available}, MODEL_PATH)
        print(f"[train_mvp] Model saved -> {MODEL_PATH}")

        return best_pipeline, best_auc, available
    
# evaluating with historical data

def validate(df: pd.DataFrame, pipeline, features: list, n_seasons: int = VALIDATION_SEASONS) -> pd.DataFrame:
    all_seasons = sorted(df["SEASON"].unique())
    val_seasons = all_seasons[-n_seasons:]

    print(f"\n[validate] Validation seasons: {val_seasons}")
    print(f"[validate] (These seasons were NOT used in training)\n")

    records = []

    for season in val_seasons:
        season_df = df[df["SEASON"] == season].copy()
        X = season_df[features].fillna(0).values

        proba = pipeline.predict_proba(X)[:, 1]
        season_df["MVP_PROB"] = proba

        # model's top pick !!
        model_pick = season_df.sort_values("MVP_PROB", ascending=False).iloc[0]

        # real mvp winner
        actual_winner = season_df[season_df["MVP_WINNER"] == True]
        if actual_winner.empty:
            actual_name = "NOT FOUND"
            actual_share = None
        else:
            actual_winner = actual_winner.iloc[0]
            acutal_name = actual_winner["PLAYER_NAME"]
            actual_share = actual_winner["SHARE"]
        
        correct = model_pick["PLAYER_NAME"] == actual_name

        # Model's ranking system
        season_df["RANK"] = season_df["MVP_PROB"].rank(ascending=False)
        winner_row = season_df[season_df["PLAYER_NAME"] == actual_name]
        winner_rank = int(winner_row["RANK"].values[0]) if not winner_row.empty else None
        winner_prob  = float(winner_row["MVP_PROB"].values[0]) if not winner_row.empty else None

        records.append({
            "SEASON":            season,
            "ACTUAL_MVP":        actual_name,
            "ACTUAL_SHARE":      round(actual_share, 3) if actual_share else None,
            "MODEL_PICK":        model_pick["PLAYER_NAME"],
            "MODEL_PICK_PROB":   round(float(model_pick["MVP_PROB"]), 3),
            "ACTUAL_WINNER_RANK": winner_rank,
            "ACTUAL_WINNER_PROB": round(winner_prob, 3) if winner_prob else None,
            "CORRECT":           correct,
        })

    results_df = pd.DataFrame(records)
    accuracy = results_df["CORRECT"].mean()

    print("-" * 72)
    print(f"{'SEASON':<10} {'ACTUAL MVP':<22} {'MODEL PICK':<22} {'RANK':>4}  {'OK?'}")
    print("─" * 72)
    for _, row in results_df.iterrows():
        ok = "✓" if row["CORRECT"] else "✗"
        print(f"{row['SEASON']:<10} {row['ACTUAL_MVP']:<22} "
              f"{row['MODEL_PICK']:<22} {str(row['ACTUAL_WINNER_RANK']):>4}   {ok}")
    print("─" * 72)
    print(f"Top-1 accuracy over {n_seasons} seasons: {accuracy:.0%}  "
          f"({int(results_df['CORRECT'].sum())}/{n_seasons} correct)\n")
 
    out_path = os.path.join(PROCESSED_DIR, "mvp_validation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"[validate] Full results -> {out_path}")
    return results_df

def main():
    print("=" * 50)
    print("MVP Likelihood — Training & Validation")
    print("=" * 50)
 
    df = pd.read_csv(TRAINING_PATH)
    print(f"[train_mvp] Loaded {len(df)} player-seasons, "
          f"{df['MVP_LABEL'].sum()} positive examples\n")
 
    all_seasons  = sorted(df["SEASON"].unique())
    train_seasons = all_seasons[:-VALIDATION_SEASONS]
    val_seasons   = all_seasons[-VALIDATION_SEASONS:]
 
    train_df = df[df["SEASON"].isin(train_seasons)]
    val_df   = df[df["SEASON"].isin(val_seasons)]
 
    print(f"[train_mvp] Training seasons : {train_seasons[0]} → {train_seasons[-1]} "
          f"({len(train_seasons)} seasons)")
    print(f"[train_mvp] Validation seasons: {val_seasons[0]} → {val_seasons[-1]} "
          f"({len(val_seasons)} seasons, held out)")
 
    pipeline, auc, features = train(train_df)
    validate(val_df, pipeline, features)
 
 
if __name__ == "__main__":
    main()