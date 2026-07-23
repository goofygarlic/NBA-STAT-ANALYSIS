# NBA-STAT-ANALYSIS
An advanced stat tracker for the NBA giving statistics like MVP candidates a real, data-backed answer.

## Setup
Install all dependencies. `pip install nba_api pandas numpy scikit-learn xgboost joblib matplotlib seaborn plotly dash requests`

Download this file from Kaggle. `https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats`. Place into `data/raw/historical/`

## How to run

### Collecting current season data:
`cd src/collections`, run `run_all.py`

### Historical data and training set
`cd historical`, run `run_historical.py`

### Model Training
`cd ../../models`, run `train_mvp.py`

### Current season scoring
`score_current_season.py`

## Why was this built?
I was having a friendly discussion with a friend of mine about NBA awards distributions. I wanted to prove a point to him so I made this project. After looking at the results, I realized that my own intution was incorrect from a statistical standpoint. My friend doesn't need to know that though :D

## Limitations
One part that is not easily trackable is the narritive perspective of MVP voting. Sometimes, there are narritives which outweigh even reasonable stats. This was shown within the validation set of training, where the model achieved only a 40% success rate (2/5 correct). This can be shown from a narritive perspective, as one of the failed years was in 2022-2023, where Joel Embiid won his first MVP. Many people state that this was more of a narritive than having solely the best stats in the league.

## Notes
Not fully operational on other machines, file saving issues are present. Output paths and input paths are hardcoded for now, will be fixing them.