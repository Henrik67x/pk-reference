import pandas as pd

df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})
season_2526 = df[df['season'] == '20252026']
print(f"Total 2025-26 players in dataset: {len(season_2526)}")
print(f"Max TOI in 2025-26: {season_2526['TOI'].max()}")
print(f"Min TOI in 2025-26: {season_2526['TOI'].min()}")
print(f"\nSample of 2025-26 players:")
print(season_2526[['player', 'team', 'TOI']].sort_values('TOI', ascending=False).head(20).to_string(index=False))