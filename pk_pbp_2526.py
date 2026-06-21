import sys
sys.path.append('.')
from pk_pbp_collector import collect_season_pbp
import pandas as pd

print("Collecting 2025-26 PK play-by-play data...")
df = collect_season_pbp("20252026")
df.to_csv("pk_pbp_20252026.csv", index=False)
print(f"\nSaved pk_pbp_20252026.csv with {len(df)} player records")

# Show Ducks players specifically
ducks = df[df['team'] == 'ANA']
print(f"\nAnaheim Ducks PK stats ({len(ducks)} players):")
print(ducks.to_string(index=False))