import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})
ages = pd.read_csv("player_birthdates.csv")
df = df.merge(ages, on='player', how='left')
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

def season_start_date(season):
    year = int(season[:4])
    return pd.Timestamp(year=year, month=10, day=1)

df['season_start'] = df['season'].apply(season_start_date)
df['age'] = (df['season_start'] - df['birth_date']).dt.days / 365.25

df = df.sort_values(['player', 'season'])
df['xga_per60'] = pd.to_numeric(df['xga_per60'], errors='coerce')
df['TOI'] = pd.to_numeric(df['TOI'], errors='coerce')

df['prior_xga_per60'] = df.groupby('player')['xga_per60'].shift(1)
df['prior_TOI'] = df.groupby('player')['TOI'].shift(1)
df['prior2_xga_per60'] = df.groupby('player')['xga_per60'].shift(2)
df['toi_trend'] = df['TOI'] - df['prior_TOI']
df['xga_trend'] = df['prior_xga_per60'] - df['prior2_xga_per60']
df['prior_team'] = df.groupby('player')['team_clean'].shift(1)
df['team_changed'] = (df['team_clean'] != df['prior_team']).astype(int)

PREDICT_FEATURES = [
    'age', 'prior_xga_per60', 'prior_TOI', 'toi_trend',
    'xga_trend', 'team_changed', 'dz_start_pct',
]

for col in PREDICT_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ─── TEST ACROSS MULTIPLE SEASON PAIRS ────────────────────────────────────────
test_seasons = ['20232024', '20242025', '20252026']

print("\n" + "="*70)
print("VALIDATION: PREDICTING EACH SEASON FROM PRIOR SEASON DATA")
print("="*70)

all_results = []

for test_season in test_seasons:
    # Train on everything except the test season and anything after it
    train_seasons = [s for s in ['20212022','20222023','20232024','20242025','20252026'] 
                      if s < test_season]
    
    train_set = df[df['season'].isin(train_seasons)].copy()
    train_set = train_set.dropna(subset=PREDICT_FEATURES + ['xga_per60'])
    
    test_set = df[df['season'] == test_season].copy()
    test_set = test_set.dropna(subset=PREDICT_FEATURES + ['xga_per60'])
    
    if len(train_set) < 50 or len(test_set) < 10:
        print(f"\n{test_season}: insufficient data, skipping")
        continue
    
    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0
    )
    model.fit(train_set[PREDICT_FEATURES], train_set['xga_per60'])
    
    test_set['predicted'] = model.predict(test_set[PREDICT_FEATURES])
    test_set['error'] = test_set['xga_per60'] - test_set['predicted']
    
    # Our model accuracy
    corr = np.corrcoef(test_set['xga_per60'], test_set['predicted'])[0,1]
    mae = np.abs(test_set['error']).mean()
    rmse = np.sqrt((test_set['error']**2).mean())
    
    # BASELINE: just predict last season repeats exactly
    baseline_error = test_set['xga_per60'] - test_set['prior_xga_per60']
    baseline_mae = np.abs(baseline_error).mean()
    baseline_corr = np.corrcoef(test_set['xga_per60'], test_set['prior_xga_per60'])[0,1]
    
    print(f"\n--- {test_season} (train: {len(train_set)}, test: {len(test_set)}) ---")
    print(f"  Our Model    — Correlation: {corr:.3f} | MAE: {mae:.3f}")
    print(f"  Baseline     — Correlation: {baseline_corr:.3f} | MAE: {baseline_mae:.3f}")
    print(f"  Improvement  — MAE reduced by: {baseline_mae - mae:.3f} ({((baseline_mae-mae)/baseline_mae*100):.1f}%)")
    
    all_results.append({
        'season': test_season,
        'our_corr': corr, 'our_mae': mae, 'our_rmse': rmse,
        'baseline_corr': baseline_corr, 'baseline_mae': baseline_mae,
        'n_test': len(test_set)
    })

# ─── ERROR DISTRIBUTION ANALYSIS (most recent test) ───────────────────────────
print("\n" + "="*70)
print("ERROR DISTRIBUTION (2025-26 predictions)")
print("="*70)

errors = test_set['error'].values
print(f"Median absolute error: {np.median(np.abs(errors)):.3f}")
print(f"75th percentile error: {np.percentile(np.abs(errors), 75):.3f}")
print(f"90th percentile error: {np.percentile(np.abs(errors), 90):.3f}")
print(f"Max error: {np.abs(errors).max():.3f}")
print(f"% of predictions within 1.0 xGA/60: {(np.abs(errors) <= 1.0).mean()*100:.1f}%")
print(f"% of predictions within 2.0 xGA/60: {(np.abs(errors) <= 2.0).mean()*100:.1f}%")

# ─── SUMMARY TABLE ─────────────────────────────────────────────────────────────
summary = pd.DataFrame(all_results)
print("\n" + "="*70)
print("SUMMARY ACROSS ALL TEST SEASONS")
print("="*70)
print(summary.to_string(index=False))

summary.to_csv("pk_validation_summary.csv", index=False)
print("\nSaved to pk_validation_summary.csv")