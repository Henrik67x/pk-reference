import pandas as pd
import requests
import time

print("Loading dataset...")
df = pd.read_csv("pk_final_dataset.csv", dtype={'season': str})

unique_players = df['player'].unique()
print(f"Unique players: {len(unique_players)}")

player_ages = {}

for i, player_name in enumerate(unique_players):
    try:
        search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=3&q={player_name.replace(' ', '%20')}&active=true"
        r = requests.get(search_url, timeout=5)
        data = r.json()
        
        if data:
            player_id = data[0]["playerId"]
            player_url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
            p_r = requests.get(player_url, timeout=5)
            p_data = p_r.json()
            birth_date = p_data.get("birthDate")
            player_ages[player_name] = birth_date
        else:
            player_ages[player_name] = None
            
    except:
        player_ages[player_name] = None
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(unique_players)}")
    
    time.sleep(0.1)

age_df = pd.DataFrame(list(player_ages.items()), columns=['player', 'birth_date'])
age_df.to_csv("player_birthdates.csv", index=False)
print(f"\nSaved {age_df['birth_date'].notna().sum()} birthdates out of {len(age_df)} players")
print("Saved to player_birthdates.csv")