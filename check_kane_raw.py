import pandas as pd
df = pd.read_csv("pk_merged.csv", dtype={'season': str})
matches = df[df['player'].str.strip().str.lower() == 'patrick kane']
print(f"Exact matches: {len(matches)}")
print(matches[['player', 'team', 'season', 'TOI']].to_string(index=False))