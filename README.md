# NBA-STAT-ANALYSIS
An advanced stat tracker for the NBA giving statistics like MVP candidates a real, data-backed answer.

## Setup
Install all dependencies. `pip install nba_api pandas numpy scikit-learn xgboost joblib matplotlib seaborn plotly dash requests`

Download this file from Kaggle. `https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats`. Place into `data/raw/historical/`

### Collecting current season data:
`cd src/collections`, run `run_all.py`

### Historical data and training set
`cd historical`, run `run_historical.py`

### Model Training
`cd ../../models`, run `train_mvp.py`

### Current season scoring
`score_current_season.py`

### Notes
Not fully operational on other machines, file saving issues are present. Output paths and input paths are hardcoded for now, will be fixing them.