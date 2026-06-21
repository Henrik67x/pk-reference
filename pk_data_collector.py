import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

import os
from dotenv import load_dotenv

load_dotenv()
NST_KEY = os.getenv("NST_KEY")

SEASONS = [
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026"
]

def get_nst_table(url, season, label):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    table = soup.find("table")
    
    if not table:
        print(f"  No table for {label} {season}")
        return pd.DataFrame()
    
    headers = [th.get_text(strip=True) for th in table.find("tr").find_all(["th", "td"])]
    rows = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) > 5:
            rows.append([col.get_text(strip=True) for col in cols])
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    if headers and len(headers) == df.shape[1]:
        df.columns = headers
    
    df["season"] = season
    return df

def get_individual_pk(season):
    url = f"https://data.naturalstattrick.com/playerteams.php?fromseason={season}&thruseason={season}&stype=2&sit=4v5&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgfilt=false&lines=single&draftteam=ALL&key={NST_KEY}"
    return get_nst_table(url, season, "individual")

def get_onice_pk(season):
    url = f"https://data.naturalstattrick.com/playerteams.php?fromseason={season}&thruseason={season}&stype=2&sit=4v5&score=all&stdoi=oi&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgfilt=false&lines=single&draftteam=ALL&key={NST_KEY}"
    return get_nst_table(url, season, "on-ice")

def collect_all_data():
    all_individual = []
    all_onice = []
    
    for season in SEASONS:
        print(f"Collecting {season}...")
        
        ind = get_individual_pk(season)
        oi = get_onice_pk(season)
        
        print(f"  Individual: {len(ind)} players")
        print(f"  On-ice: {len(oi)} players")
        
        all_individual.append(ind)
        all_onice.append(oi)
        time.sleep(1)
    
    ind_combined = pd.concat(all_individual, ignore_index=True)
    oi_combined = pd.concat(all_onice, ignore_index=True)
    
    # Clean up column names for merging
    ind_combined = ind_combined.rename(columns={"Player": "player", "Team": "team", "Position": "position"})
    oi_combined = oi_combined.rename(columns={"Player": "player", "Team": "team", "Position": "position"})
    
    # Add suffixes to avoid column name conflicts
    oi_cols = {col: f"oi_{col}" for col in oi_combined.columns if col not in ["player", "team", "position", "season", "GP", "TOI", ""]}
    oi_combined = oi_combined.rename(columns=oi_cols)
    
    # Merge on player + team + season
    merged = pd.merge(
        ind_combined,
        oi_combined,
        on=["player", "team", "season"],
        how="inner",
        suffixes=("", "_oi")
    )
    
    print(f"\nMerged dataset: {len(merged)} records, {len(merged.columns)} columns")
    
    # Save both
    ind_combined.to_csv("pk_individual.csv", index=False)
    oi_combined.to_csv("pk_onice.csv", index=False)
    merged.to_csv("pk_merged.csv", index=False)
    
    print("Saved: pk_individual.csv, pk_onice.csv, pk_merged.csv")
    return merged

if __name__ == "__main__":
    df = collect_all_data()
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample row:")
    print(df.iloc[0])