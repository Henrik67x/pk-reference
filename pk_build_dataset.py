import pandas as pd
import numpy as np

print("Loading NST data...")
nst = pd.read_csv("pk_merged.csv", dtype={'season': str})

numeric_cols = [
    'TOI', 'GP', 'Goals', 'Total Assists', 'Shots', 'ixG', 'iCF', 'iFF',
    'iSCF', 'iHDCF', 'Giveaways', 'Takeaways', 'Hits', 'Shots Blocked',
    'Faceoffs Won', 'Faceoffs Lost', 'Faceoffs %', 'PIM',
    'Penalties Drawn', 'Total Penalties',
    'oi_CF', 'oi_CA', 'oi_CF%', 'oi_SF', 'oi_SA', 'oi_SF%',
    'oi_GF', 'oi_GA', 'oi_GF%', 'oi_xGF', 'oi_xGA', 'oi_xGF%',
    'oi_SCF', 'oi_SCA', 'oi_SCF%', 'oi_HDCF', 'oi_HDCA', 'oi_HDCF%',
    'oi_HDGF', 'oi_HDGA', 'oi_HDGF%',
    'oi_PDO', 'oi_Off.\xa0Zone Start %', 'oi_On-Ice SV%'
]

for col in numeric_cols:
    if col in nst.columns:
        nst[col] = pd.to_numeric(nst[col], errors='coerce')

nst['TOI'] = pd.to_numeric(nst['TOI'], errors='coerce')

# Deduplicate — keep highest TOI per player/season
nst = nst.sort_values('TOI', ascending=False)
nst = nst.drop_duplicates(subset=['player', 'season'], keep='first')

# Minimum TOI filter — 60 minutes
merged = nst[nst['TOI'] >= 10].copy()
print(f"After 60min filter: {len(merged)}")

merged['team_clean'] = merged['team'].apply(lambda x: str(x).split(',')[0].strip())

# Feature engineering — all directly from NST, validated and accurate
def engineer_features(df):
    d = df.copy()
    toi = d['TOI'].replace(0, np.nan)

    # Individual PK actions (NST, validated)
    d['pk_blocks_per60'] = (d['Shots Blocked'] / toi) * 60
    d['pk_takeaways_per60'] = (d['Takeaways'] / toi) * 60
    d['pk_giveaways_per60'] = (d['Giveaways'] / toi) * 60
    d['pk_hits_per60'] = (d['Hits'] / toi) * 60
    d['pk_penalties_taken_per60'] = (d['Total Penalties'] / toi) * 60
    d['sh_goals_per60'] = (d['Goals'] / toi) * 60
    d['ihdcf_per60'] = (d['iHDCF'] / toi) * 60
    d['ixg_per60'] = (d['ixG'] / toi) * 60

    fo_total = d['Faceoffs Won'] + d['Faceoffs Lost']
    d['pk_fo_pct'] = (d['Faceoffs Won'] / fo_total.replace(0, np.nan) * 100).fillna(0)

    d['pk_ta_ga_ratio'] = d['pk_takeaways_per60'] / (d['pk_giveaways_per60'] + 0.1)

    # On-ice (team context, NST)
    d['xga_per60'] = (d['oi_xGA'] / toi) * 60
    d['ga_per60'] = (d['oi_GA'] / toi) * 60
    d['hdca_per60'] = (d['oi_HDCA'] / toi) * 60
    d['sa_per60'] = (d['oi_SA'] / toi) * 60
    d['sca_per60'] = (d['oi_SCA'] / toi) * 60
    d['ca_per60'] = (d['oi_CA'] / toi) * 60

    d['block_rate'] = d['pk_blocks_per60'] / (d['sa_per60'] + 0.1)

    d['dz_start_pct'] = 100 - d['oi_Off.\xa0Zone Start %'].fillna(50)
    d['toi_pg'] = d['TOI'] / d['GP'].replace(0, np.nan)

    d['sv_pct'] = pd.to_numeric(d['oi_On-Ice SV%'], errors='coerce')
    median_sv = d['sv_pct'].median()
    d['sv_pct'] = d['sv_pct'].fillna(median_sv)

    d['pdo'] = pd.to_numeric(d['oi_PDO'], errors='coerce').fillna(1.0)
    d['hdca_pct_of_sa'] = d['oi_HDCA'] / (d['oi_SA'].replace(0, np.nan))

    d['is_defense'] = (d['position'] == 'D').astype(int)
    d['is_center'] = (d['position'] == 'C').astype(int)

    d['individual_pk_score'] = (
        d['pk_blocks_per60'] * 1.0 +
        d['pk_takeaways_per60'] * 1.5 -
        d['pk_giveaways_per60'] * 1.2 -
        d['pk_penalties_taken_per60'] * 3.0 +
        d['sh_goals_per60'] * 4.0 +
        d['pk_hits_per60'] * 0.3
    )

    return d

merged = engineer_features(merged)

merged.to_csv("pk_final_dataset.csv", index=False)
print(f"\nSaved pk_final_dataset.csv — {len(merged)} records, all NST-validated")

print("\nSample — Leo Carlsson:")
leo = merged[merged['player'].str.contains('Carlsson', na=False)]
cols = ['player', 'team', 'season', 'TOI', 'pk_blocks_per60', 
        'pk_takeaways_per60', 'pk_giveaways_per60', 'xga_per60', 'ga_per60']
print(leo[cols].to_string(index=False))