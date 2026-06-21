import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})
ages = pd.read_csv("player_birthdates.csv")

# Merge ages
df = df.merge(ages, on='player', how='left')
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

# Calculate age AS OF the start of each season (Oct 1)
def season_start_date(season):
    year = int(season[:4])
    return pd.Timestamp(year=year, month=10, day=1)

df['season_start'] = df['season'].apply(season_start_date)
df['age'] = (df['season_start'] - df['birth_date']).dt.days / 365.25

print(f"Players with age data: {df['age'].notna().sum()} / {len(df)}")

# ─── BUILD PRIOR-SEASON FEATURES ──────────────────────────────────────────────
# For each player-season, we need their PRIOR season's stats as predictors

df = df.sort_values(['player', 'season'])

# Calculate basic per-60 rates we need historically
df['xga_per60'] = pd.to_numeric(df['xga_per60'], errors='coerce')
df['ga_per60'] = pd.to_numeric(df['ga_per60'], errors='coerce')
df['TOI'] = pd.to_numeric(df['TOI'], errors='coerce')

# Shift to get PRIOR season stats for same player
df['prior_xga_per60'] = df.groupby('player')['xga_per60'].shift(1)
df['prior_ga_per60'] = df.groupby('player')['ga_per60'].shift(1)
df['prior_TOI'] = df.groupby('player')['TOI'].shift(1)
df['prior_season'] = df.groupby('player')['season'].shift(1)

# Two seasons back
df['prior2_xga_per60'] = df.groupby('player')['xga_per60'].shift(2)
df['prior2_TOI'] = df.groupby('player')['TOI'].shift(2)

# TOI trend — are they being used more or less?
df['toi_trend'] = df['TOI'] - df['prior_TOI']

# Performance trend — improving or declining?
df['xga_trend'] = df['prior_xga_per60'] - df['prior2_xga_per60']  # negative = improving

# Team change flag
df['prior_team'] = df.groupby('player')['team_clean'].shift(1)
df['team_changed'] = (df['team_clean'] != df['prior_team']).astype(int)

# ─── ISOLATE 2025-26 FOR PREDICTION ───────────────────────────────────────────
predict_set = df[df['season'] == '20252026'].copy()
predict_set = predict_set.dropna(subset=['prior_xga_per60'])  # must have prior season data

print(f"\n2025-26 players with prior season data: {len(predict_set)}")

# ─── BUILD TRAINING SET (predict season N using season N-1 info) ─────────────
train_set = df[df['season'].isin(['20222023', '20232024', '20242025'])].copy()
train_set = train_set.dropna(subset=['prior_xga_per60', 'xga_per60'])

print(f"Training set (predicting using prior season): {len(train_set)}")

PREDICT_FEATURES = [
    'age',
    'prior_xga_per60',
    'prior_TOI',
    'toi_trend',
    'xga_trend',
    'team_changed',
    'dz_start_pct',  # current season's actual deployment (known going in from coaching decisions)
]

# Clean
for col in PREDICT_FEATURES:
    train_set[col] = pd.to_numeric(train_set[col], errors='coerce')
    predict_set[col] = pd.to_numeric(predict_set[col], errors='coerce')

train_set[PREDICT_FEATURES] = train_set[PREDICT_FEATURES].fillna(train_set[PREDICT_FEATURES].median())
predict_set[PREDICT_FEATURES] = predict_set[PREDICT_FEATURES].fillna(train_set[PREDICT_FEATURES].median())

train_set = train_set.dropna(subset=['xga_per60'])

# ─── TRAIN PREDICTION MODEL ───────────────────────────────────────────────────
print("\nTraining prediction model...")

pred_model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0
)

X_train = train_set[PREDICT_FEATURES].values
y_train = train_set['xga_per60'].values

pred_model.fit(X_train, y_train)

# ─── PREDICT 2025-26 ──────────────────────────────────────────────────────────
predict_set['predicted_xga_per60'] = pred_model.predict(predict_set[PREDICT_FEATURES].values)

# ─── COMPARE TO ACTUAL ────────────────────────────────────────────────────────
predict_set['actual_xga_per60'] = predict_set['xga_per60']
predict_set['prediction_error'] = predict_set['actual_xga_per60'] - predict_set['predicted_xga_per60']

corr = np.corrcoef(predict_set['actual_xga_per60'], predict_set['predicted_xga_per60'])[0,1]
mae = np.abs(predict_set['prediction_error']).mean()

print(f"\n--- PREDICTION RESULTS (2025-26) ---")
print(f"Correlation (predicted vs actual): {corr:.3f}")
print(f"Mean Absolute Error: {mae:.3f} xGA/60")

results = predict_set[['player', 'team_clean', 'age', 'TOI', 
                        'predicted_xga_per60', 'actual_xga_per60', 'prediction_error']].copy()
results = results.sort_values('predicted_xga_per60')

print("\n--- PREDICTED TOP 15 (best expected PK performance) ---")
print(results.head(15).to_string(index=False))

print("\n--- BIGGEST OVERPERFORMERS (actual much better than predicted) ---")
overperform = results.nsmallest(10, 'prediction_error')
print(overperform.to_string(index=False))

print("\n--- BIGGEST UNDERPERFORMERS (actual much worse than predicted) ---")
underperform = results.nlargest(10, 'prediction_error')
print(underperform.to_string(index=False))

print("\n--- FEATURE IMPORTANCE ---")
imp = pd.DataFrame({
    'feature': PREDICT_FEATURES,
    'importance': pred_model.feature_importances_
}).sort_values('importance', ascending=False)
print(imp.to_string(index=False))

results.to_csv("pk_predictions_2025_26.csv", index=False)
print("\nSaved to pk_predictions_2025_26.csv")