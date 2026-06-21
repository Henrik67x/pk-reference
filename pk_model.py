import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("pk_merged.csv", dtype={'season': str})

# ─── CLEAN ────────────────────────────────────────────────────────────────────
numeric_cols = [
    'TOI', 'GP', 'Goals', 'Total Assists', 'Shots', 'ixG', 'iCF', 'iFF',
    'iSCF', 'iHDCF', 'Giveaways', 'Takeaways', 'Hits', 'Shots Blocked',
    'Faceoffs Won', 'Faceoffs Lost', 'Faceoffs %', 'PIM',
    'Penalties Drawn', 'Total Penalties',
    'oi_CF', 'oi_CA', 'oi_CF%', 'oi_SF', 'oi_SA', 'oi_SF%',
    'oi_GF', 'oi_GA', 'oi_GF%', 'oi_xGF', 'oi_xGA', 'oi_xGF%',
    'oi_SCF', 'oi_SCA', 'oi_SCF%', 'oi_HDCF', 'oi_HDCA', 'oi_HDCF%',
    'oi_HDGF', 'oi_HDGA', 'oi_HDGF%',
    'oi_PDO', 'oi_Off.\xa0Zone Start %'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fix TOI
df['TOI'] = pd.to_numeric(df['TOI'], errors='coerce')

# Remove duplicates — keep highest TOI record per player/season
df = df.sort_values('TOI', ascending=False)
df = df.drop_duplicates(subset=['player', 'season'], keep='first')

# Minimum TOI filter
df = df[df['TOI'] >= 30].copy()

print(f"After dedup + TOI filter: {len(df)} records")

# Split
train = df[df['season'] != '20242025'].copy()
test = df[df['season'] == '20242025'].copy()

print(f"Training records: {len(train)}")
print(f"Test records (2024-25): {len(test)}")

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    toi = d['TOI'].replace(0, np.nan)

    # Individual per-60 actions
    d['blocks_per60'] = (d['Shots Blocked'] / toi) * 60
    d['takeaways_per60'] = (d['Takeaways'] / toi) * 60
    d['giveaways_per60'] = (d['Giveaways'] / toi) * 60
    d['hits_per60'] = (d['Hits'] / toi) * 60
    d['penalties_taken_per60'] = (d['Total Penalties'] / toi) * 60
    d['penalties_drawn_per60'] = (d['Penalties Drawn'] / toi) * 60
    d['ixg_per60'] = (d['ixG'] / toi) * 60
    d['ihdcf_per60'] = (d['iHDCF'] / toi) * 60
    d['sh_goals_per60'] = (d['Goals'] / toi) * 60
    d['icf_per60'] = (d['iCF'] / toi) * 60

    # Ratio stats
    d['ta_ga_ratio'] = d['takeaways_per60'] / (d['giveaways_per60'] + 0.1)
    d['block_rate'] = d['blocks_per60'] / (d['sa_per60'] + 0.1) if 'sa_per60' in d.columns else 0

    # On-ice per-60
    d['xga_per60'] = (d['oi_xGA'] / toi) * 60
    d['ga_per60'] = (d['oi_GA'] / toi) * 60
    d['hdca_per60'] = (d['oi_HDCA'] / toi) * 60
    d['sa_per60'] = (d['oi_SA'] / toi) * 60
    d['sca_per60'] = (d['oi_SCA'] / toi) * 60
    d['ca_per60'] = (d['oi_CA'] / toi) * 60

    # Block rate now that sa_per60 exists
    d['block_rate'] = d['blocks_per60'] / (d['sa_per60'] + 0.1)

    # Deployment
    d['dz_start_pct'] = 100 - d['oi_Off.\xa0Zone Start %'].fillna(50)
    d['toi_pg'] = d['TOI'] / d['GP'].replace(0, np.nan)

    # Faceoffs
    d['faceoff_pct'] = pd.to_numeric(d['Faceoffs %'], errors='coerce').fillna(0)

    # Position
    d['is_defense'] = (d['position'] == 'D').astype(int)
    d['is_center'] = (d['position'] == 'C').astype(int)

    # Luck
    d['pdo'] = pd.to_numeric(d['oi_PDO'], errors='coerce').fillna(1.0)

    # High danger suppression rate
    d['hdca_pct_of_sa'] = d['oi_HDCA'] / (d['oi_SA'].replace(0, np.nan))

    # Individual defensive impact score
    # Combines blocks + takeaways - giveaways - penalties
    d['individual_defensive_score'] = (
        d['blocks_per60'] * 1.0 +
        d['takeaways_per60'] * 1.5 -
        d['giveaways_per60'] * 1.2 -
        d['penalties_taken_per60'] * 2.0 +
        d['penalties_drawn_per60'] * 0.5 +
        d['sh_goals_per60'] * 3.0
    )

    return d

train = engineer_features(train)
test = engineer_features(test)

# ─── FEATURES ─────────────────────────────────────────────────────────────────
# Split into team context vs individual
# This lets us isolate individual contribution

TEAM_CONTEXT_FEATURES = [
    'hdca_per60',
    'sca_per60',
    'sa_per60',
    'ca_per60',
    'hdca_pct_of_sa',
    'dz_start_pct',
    'pdo',
]

INDIVIDUAL_FEATURES = [
    'blocks_per60',
    'block_rate',
    'takeaways_per60',
    'giveaways_per60',
    'hits_per60',
    'penalties_taken_per60',
    'penalties_drawn_per60',
    'sh_goals_per60',
    'ixg_per60',
    'ihdcf_per60',
    'icf_per60',
    'ta_ga_ratio',
    'individual_defensive_score',
    'faceoff_pct',
    'toi_pg',
    'is_defense',
    'is_center',
]

FEATURES = TEAM_CONTEXT_FEATURES + INDIVIDUAL_FEATURES

TARGET_xPKS = 'xga_per60'
TARGET_PKS = 'ga_per60'

def clean_features(df, features, target):
    subset = df[features + [target, 'player', 'team', 'season', 'TOI', 'position']].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna(subset=features + [target])
    return subset

train_xpks = clean_features(train, FEATURES, TARGET_xPKS)
train_pks = clean_features(train, FEATURES, TARGET_PKS)
test_xpks = clean_features(test, FEATURES, TARGET_xPKS)
test_pks = clean_features(test, FEATURES, TARGET_PKS)

print(f"\nTraining xPKS: {len(train_xpks)} | Test: {len(test_xpks)}")
print(f"Training PKS: {len(train_pks)} | Test: {len(test_pks)}")

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nTraining models...")

def make_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

xpks_model = make_model()
pks_model = make_model()

xpks_model.fit(train_xpks[FEATURES], train_xpks[TARGET_xPKS])
pks_model.fit(train_pks[FEATURES], train_pks[TARGET_PKS])

print("Done.")

# ─── VALIDATE ─────────────────────────────────────────────────────────────────
xpks_preds = xpks_model.predict(test_xpks[FEATURES])
pks_preds = pks_model.predict(test_pks[FEATURES])

xpks_rmse = np.sqrt(mean_squared_error(test_xpks[TARGET_xPKS], xpks_preds))
pks_rmse = np.sqrt(mean_squared_error(test_pks[TARGET_PKS], pks_preds))
xpks_corr = np.corrcoef(test_xpks[TARGET_xPKS], xpks_preds)[0,1]
pks_corr = np.corrcoef(test_pks[TARGET_PKS], pks_preds)[0,1]

print(f"\n--- VALIDATION ---")
print(f"xPKS RMSE: {xpks_rmse:.4f} | Correlation: {xpks_corr:.4f}")
print(f"PKS RMSE:  {pks_rmse:.4f} | Correlation: {pks_corr:.4f}")

# ─── NORMALIZE TO -20 to +20 ──────────────────────────────────────────────────
def normalize_scores(predictions, percentile_clip=95):
    """
    Lower predicted xGA/GA = better PK player = higher score
    Use percentile-based normalization to avoid extreme clipping
    """
    flipped = -predictions
    
    low = np.percentile(flipped, 100 - percentile_clip)
    high = np.percentile(flipped, percentile_clip)
    
    scores = ((flipped - low) / (high - low)) * 40 - 20
    scores = np.clip(scores, -20, 20)
    return np.round(scores, 1)

test_xpks = test_xpks.copy()
test_pks = test_pks.copy()

test_xpks['xPKS'] = normalize_scores(xpks_preds)
test_pks['PKS'] = normalize_scores(pks_preds)

# Merge
results = test_xpks[['player', 'team', 'season', 'TOI', 'position', 'xPKS']].merge(
    test_pks[['player', 'team', 'season', 'PKS']],
    on=['player', 'team', 'season']
)

results = results.sort_values('xPKS', ascending=False).reset_index(drop=True)
results['Rank'] = results.index + 1

# ─── RESULTS ──────────────────────────────────────────────────────────────────
print("\n--- TOP 20 PK FORWARDS (2024-25) ---")
fwd = results[results['position'].isin(['C', 'L', 'R'])].head(20)
print(fwd[['Rank', 'player', 'team', 'TOI', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- TOP 20 PK DEFENSEMEN (2024-25) ---")
dmen = results[results['position'] == 'D'].head(20)
print(dmen[['Rank', 'player', 'team', 'TOI', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- BOTTOM 10 FORWARDS (2024-25) ---")
fwd_bottom = results[results['position'].isin(['C', 'L', 'R'])].tail(10)
print(fwd_bottom[['Rank', 'player', 'team', 'TOI', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- SCORE DISTRIBUTION ---")
print(f"xPKS mean: {results['xPKS'].mean():.1f}")
print(f"xPKS std:  {results['xPKS'].std():.1f}")
print(f"xPKS min:  {results['xPKS'].min():.1f}")
print(f"xPKS max:  {results['xPKS'].max():.1f}")
print(f"Players at exactly +20: {(results['xPKS'] == 20).sum()}")
print(f"Players at exactly -20: {(results['xPKS'] == -20).sum()}")

print("\n--- FEATURE IMPORTANCE ---")
imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': xpks_model.feature_importances_
}).sort_values('importance', ascending=False)
print(imp.head(15).to_string(index=False))

results.to_csv("pk_scores_2024_25.csv", index=False)
print("\nSaved to pk_scores_2024_25.csv")