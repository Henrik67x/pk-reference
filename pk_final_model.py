import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})

# ─── CONTEXT FEATURES (things outside player control) ─────────────────────────
CONTEXT_FEATURES = [
    'dz_start_pct',      # deployment
    'toi_pg',            # usage/trust
    'is_defense',        # position
    'is_center',         # position
    'sv_pct',            # goalie quality
    'ca_per60',          # team defensive structure
    'pdo',               # luck
]

# ─── INDIVIDUAL FEATURES (things the player controls) ─────────────────────────
INDIVIDUAL_FEATURES = [
    'pk_blocks_per60',
    'block_rate',
    'pk_takeaways_per60',
    'pk_giveaways_per60',
    'pk_ta_ga_ratio',
    'pk_hits_per60',
    'pk_penalties_taken_per60',
    'sh_goals_per60',
    'pk_fo_pct',
    'ihdcf_per60',
    'individual_pk_score',
]

ALL_FEATURES = CONTEXT_FEATURES + INDIVIDUAL_FEATURES

TARGET_xPKS = 'xga_per60'
TARGET_PKS = 'ga_per60'

# ─── CLEAN ────────────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan)

# Fix sv_pct — should be around 85-95, not 77
# If values look like percentages already leave them, if decimal multiply by 100
if df['sv_pct'].median() < 1:
    df['sv_pct'] = df['sv_pct'] * 100

# Fill missing context with medians
for col in CONTEXT_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

for col in INDIVIDUAL_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

for col in [TARGET_xPKS, TARGET_PKS]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove extreme outliers in targets (>99th percentile)
xga_99 = df[TARGET_xPKS].quantile(0.99)
ga_99 = df[TARGET_PKS].quantile(0.99)
df = df[df[TARGET_xPKS] <= xga_99].copy()
df = df[df[TARGET_PKS] <= ga_99].copy()

# Drop rows missing targets
df = df.dropna(subset=[TARGET_xPKS, TARGET_PKS])

print(f"Clean dataset: {len(df)} records")

# ─── SPLIT ────────────────────────────────────────────────────────────────────
train = df[df['season'] != '20242025'].copy()
test = df[df['season'] == '20242025'].copy()

print(f"Training: {len(train)} | Test: {len(test)}")

# ─── STAGE 1: CONTEXT MODEL ───────────────────────────────────────────────────
# What xGA/60 would an average player allow in this exact situation?
print("\nTraining Stage 1 — Context model...")

context_model_xpks = Ridge(alpha=1.0)
context_model_pks = Ridge(alpha=1.0)

X_train_ctx = train[CONTEXT_FEATURES].values
context_model_xpks.fit(X_train_ctx, train[TARGET_xPKS])
context_model_pks.fit(X_train_ctx, train[TARGET_PKS])

# Calculate residuals — actual minus expected given context
train['expected_xga'] = context_model_xpks.predict(X_train_ctx)
train['expected_ga'] = context_model_pks.predict(train[CONTEXT_FEATURES].values)

train['residual_xga'] = train[TARGET_xPKS] - train['expected_xga']
train['residual_ga'] = train[TARGET_PKS] - train['expected_ga']

test['expected_xga'] = context_model_xpks.predict(test[CONTEXT_FEATURES].values)
test['expected_ga'] = context_model_pks.predict(test[CONTEXT_FEATURES].values)

test['residual_xga'] = test[TARGET_xPKS] - test['expected_xga']
test['residual_ga'] = test[TARGET_PKS] - test['expected_ga']

print(f"Context model R² (xGA): {context_model_xpks.score(X_train_ctx, train[TARGET_xPKS]):.3f}")
print(f"Context model R² (GA):  {context_model_pks.score(train[CONTEXT_FEATURES].values, train[TARGET_PKS]):.3f}")

# ─── STAGE 2: INDIVIDUAL MODEL ────────────────────────────────────────────────
# Can individual actions explain the residual?
print("\nTraining Stage 2 — Individual action model...")

def make_xgb():
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=8,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0
    )

ind_model_xpks = make_xgb()
ind_model_pks = make_xgb()

X_train_ind = train[INDIVIDUAL_FEATURES].values
ind_model_xpks.fit(X_train_ind, train['residual_xga'])
ind_model_pks.fit(X_train_ind, train['residual_ga'])

# Predict individual contribution on test set
test['ind_contribution_xga'] = ind_model_xpks.predict(test[INDIVIDUAL_FEATURES].values)
test['ind_contribution_ga'] = ind_model_pks.predict(test[INDIVIDUAL_FEATURES].values)

# ─── STAGE 3: FINAL SCORE ─────────────────────────────────────────────────────
# xPKS = based on residual explained by individual actions
# PKS = based on actual outcome residual
# Both adjusted for context

# The score = -(residual) because lower xGA = better
# We use the individual contribution to explain the residual
test['raw_xPKS'] = -(test['expected_xga'] + test['ind_contribution_xga'])
test['raw_PKS'] = -(test['expected_ga'] + test['ind_contribution_ga'])

# ─── NORMALIZE TO SCALE ───────────────────────────────────────────────────────
def normalize_to_pkscore(raw_scores, avg=10, elite=18.5, max_val=20):
    """
    Normalize so that:
    - Average player = 10
    - Elite player (95th percentile) = 18-19
    - 20 is essentially perfect
    - Negative values are genuinely bad
    """
    median = np.median(raw_scores)
    p95 = np.percentile(raw_scores, 95)
    p5 = np.percentile(raw_scores, 5)
    
    scores = np.zeros(len(raw_scores))
    
    for i, val in enumerate(raw_scores):
        if val >= median:
            # Above average: scale from 10 to 20
            normalized = (val - median) / (p95 - median)
            scores[i] = avg + normalized * (elite - avg)
        else:
            # Below average: scale from negative to 10
            normalized = (val - median) / (median - p5)
            scores[i] = avg + normalized * avg  # can go negative
    
    scores = np.clip(scores, -20, max_val)
    return np.round(scores, 1)

test['xPKS'] = normalize_to_pkscore(test['raw_xPKS'].values)
test['PKS'] = normalize_to_pkscore(test['raw_PKS'].values)

# ─── VALIDATION ───────────────────────────────────────────────────────────────
print("\n--- VALIDATION ---")
xpks_corr = np.corrcoef(test[TARGET_xPKS], test['xPKS'])[0,1]
pks_corr = np.corrcoef(test[TARGET_PKS], test['PKS'])[0,1]
print(f"xPKS correlation with actual xGA/60: {xpks_corr:.3f}")
print(f"PKS correlation with actual GA/60:   {pks_corr:.3f}")

print("\n--- SCORE DISTRIBUTION ---")
print(f"xPKS — mean: {test['xPKS'].mean():.1f} | std: {test['xPKS'].std():.1f} | min: {test['xPKS'].min():.1f} | max: {test['xPKS'].max():.1f}")
print(f"PKS  — mean: {test['PKS'].mean():.1f} | std: {test['PKS'].std():.1f} | min: {test['PKS'].min():.1f} | max: {test['PKS'].max():.1f}")
print(f"Players at exactly 20: {(test['xPKS'] == 20).sum()}")
print(f"Players below 0: {(test['xPKS'] < 0).sum()}")

# ─── RESULTS ──────────────────────────────────────────────────────────────────
results = test[['player', 'team', 'season', 'TOI', 'position', 
                'xPKS', 'PKS', 'xga_per60', 'ga_per60',
                'pk_blocks_per60', 'pk_takeaways_per60', 
                'pk_giveaways_per60', 'sh_goals_per60']].copy()

results = results.sort_values('xPKS', ascending=False).reset_index(drop=True)
results['Rank'] = results.index + 1

print("\n--- TOP 20 PK FORWARDS (2024-25) ---")
fwd = results[results['position'].isin(['C', 'L', 'R'])].head(20)
print(fwd[['Rank', 'player', 'team', 'TOI', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- TOP 20 PK DEFENSEMEN (2024-25) ---")
dmen = results[results['position'] == 'D'].head(20)
print(dmen[['Rank', 'player', 'team', 'TOI', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- BOTTOM 10 (2024-25) ---")
print(results.tail(10)[['Rank', 'player', 'team', 'TOI', 'position', 'xPKS', 'PKS']].to_string(index=False))

print("\n--- xPKS vs PKS GAP (most bailed out by goalie/teammates) ---")
results['gap'] = results['PKS'] - results['xPKS']
bailed = results.nlargest(10, 'gap')
print(bailed[['player', 'team', 'xPKS', 'PKS', 'gap']].to_string(index=False))

print("\n--- xPKS vs PKS GAP (let down by goalie/teammates) ---")
letdown = results.nsmallest(10, 'gap')
print(letdown[['player', 'team', 'xPKS', 'PKS', 'gap']].to_string(index=False))

print("\n--- FEATURE IMPORTANCE ---")
imp = pd.DataFrame({
    'feature': INDIVIDUAL_FEATURES,
    'importance': ind_model_xpks.feature_importances_
}).sort_values('importance', ascending=False)
print(imp.to_string(index=False))

results.to_csv("pk_scores_final_2024_25.csv", index=False)
print("\nSaved to pk_scores_final_2024_25.csv")