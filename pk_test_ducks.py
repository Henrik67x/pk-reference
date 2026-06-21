import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

CONTEXT_FEATURES = [
    'dz_start_pct', 'toi_pg', 'is_defense', 'is_center',
    'sv_pct', 'ca_per60', 'pdo',
]

INDIVIDUAL_FEATURES = [
    'pk_blocks_per60', 'block_rate', 'pk_takeaways_per60',
    'pk_giveaways_per60', 'pk_ta_ga_ratio', 'pk_hits_per60',
    'pk_penalties_taken_per60', 'sh_goals_per60', 'pk_fo_pct',
    'ihdcf_per60', 'individual_pk_score',
]

TARGET_xPKS = 'xga_per60'
TARGET_PKS = 'ga_per60'

print("Loading data...")
df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})
df = df.replace([np.inf, -np.inf], np.nan)

if df['sv_pct'].median() < 1:
    df['sv_pct'] = df['sv_pct'] * 100

for col in CONTEXT_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in INDIVIDUAL_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)

for col in [TARGET_xPKS, TARGET_PKS]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

xga_99 = df[TARGET_xPKS].quantile(0.99)
ga_99 = df[TARGET_PKS].quantile(0.99)
df = df[df[TARGET_xPKS] <= xga_99].copy()
df = df[df[TARGET_PKS] <= ga_99].copy()
df = df.dropna(subset=[TARGET_xPKS, TARGET_PKS])

# Train on EVERYTHING except 2025-26 (since it's incomplete/current season)
train = df[df['season'] != '20252026'].copy()
ducks_2526 = df[(df['season'] == '20252026') & (df['team_clean'] == 'ANA')].copy()

print(f"Training records: {len(train)}")
print(f"Ducks 2025-26 records: {len(ducks_2526)}")

if len(ducks_2526) == 0:
    print("No 2025-26 Ducks data found. Checking what's available...")
    print(df[df['season'] == '20252026']['team_clean'].unique())

# Stage 1 — context
context_model_xpks = Ridge(alpha=1.0)
context_model_pks = Ridge(alpha=1.0)

X_train_ctx = train[CONTEXT_FEATURES].values
context_model_xpks.fit(X_train_ctx, train[TARGET_xPKS])
context_model_pks.fit(X_train_ctx, train[TARGET_PKS])

train['expected_xga'] = context_model_xpks.predict(X_train_ctx)
train['expected_ga'] = context_model_pks.predict(X_train_ctx)
train['residual_xga'] = train[TARGET_xPKS] - train['expected_xga']
train['residual_ga'] = train[TARGET_PKS] - train['expected_ga']

# Stage 2 — individual
def make_xgb():
    return XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=8,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, verbosity=0
    )

ind_model_xpks = make_xgb()
ind_model_pks = make_xgb()

X_train_ind = train[INDIVIDUAL_FEATURES].values
ind_model_xpks.fit(X_train_ind, train['residual_xga'])
ind_model_pks.fit(X_train_ind, train['residual_ga'])

# Apply to Ducks 2025-26
if len(ducks_2526) > 0:
    ducks_2526['expected_xga'] = context_model_xpks.predict(ducks_2526[CONTEXT_FEATURES].values)
    ducks_2526['expected_ga'] = context_model_pks.predict(ducks_2526[CONTEXT_FEATURES].values)
    ducks_2526['ind_contribution_xga'] = ind_model_xpks.predict(ducks_2526[INDIVIDUAL_FEATURES].values)
    ducks_2526['ind_contribution_ga'] = ind_model_pks.predict(ducks_2526[INDIVIDUAL_FEATURES].values)

    ducks_2526['raw_xPKS'] = -(ducks_2526['expected_xga'] + ducks_2526['ind_contribution_xga'])
    ducks_2526['raw_PKS'] = -(ducks_2526['expected_ga'] + ducks_2526['ind_contribution_ga'])

    # Use the full training set raw scores to calibrate normalization
    train['expected_xga_self'] = train['expected_xga']
    train['ind_contribution_xga_self'] = ind_model_xpks.predict(X_train_ind)
    train['raw_xPKS'] = -(train['expected_xga'] + train['ind_contribution_xga_self'])

    train['ind_contribution_ga_self'] = ind_model_pks.predict(X_train_ind)
    train['raw_PKS'] = -(train['expected_ga'] + train['ind_contribution_ga_self'])

    def normalize_to_pkscore(raw_scores, reference_scores, avg=10, elite=18.5, max_val=20):
        median = np.median(reference_scores)
        p95 = np.percentile(reference_scores, 95)
        p5 = np.percentile(reference_scores, 5)

        scores = np.zeros(len(raw_scores))
        for i, val in enumerate(raw_scores):
            if val >= median:
                normalized = (val - median) / (p95 - median)
                scores[i] = avg + normalized * (elite - avg)
            else:
                normalized = (val - median) / (median - p5)
                scores[i] = avg + normalized * avg
        return np.clip(scores, -20, max_val).round(1)

    ducks_2526['xPKS'] = normalize_to_pkscore(ducks_2526['raw_xPKS'].values, train['raw_xPKS'].values)
    ducks_2526['PKS'] = normalize_to_pkscore(ducks_2526['raw_PKS'].values, train['raw_PKS'].values)

    print("\n--- ANAHEIM DUCKS 2025-26 PK SCORES ---")
    result_cols = ['player', 'TOI', 'position', 'xPKS', 'PKS', 
                    'pk_blocks_per60', 'pk_takeaways_per60', 'pk_giveaways_per60']
    print(ducks_2526.sort_values('xPKS', ascending=False)[result_cols].to_string(index=False))

    ducks_2526.to_csv("ducks_pk_2025_26.csv", index=False)
    print("\nSaved to ducks_pk_2025_26.csv")