import requests
import pandas as pd
import time
from collections import defaultdict

def get_season_game_ids(season):
    """Get all regular season game IDs for a given season"""
    url = f"https://api-web.nhle.com/v1/schedule/calendar/{season[:4]}-10-01"
    
    # Use the standings to get team list, then pull schedule
    schedule_url = f"https://api-web.nhle.com/v1/club-schedule-season/ANA/{season}"
    r = requests.get(schedule_url)
    data = r.json()
    
    game_ids = []
    for game in data.get('games', []):
        if game.get('gameType') == 2:  # Regular season only
            game_ids.append(game['id'])
    
    return game_ids

def get_all_game_ids_for_season(season):
    """Get game IDs by pulling from multiple teams to ensure complete coverage"""
    teams = ['ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 
             'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NJD',
             'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS',
             'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WPG', 'WSH']
    
    all_ids = set()
    for team in teams[:4]:  # Only need a few teams to get all games
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        r = requests.get(url)
        data = r.json()
        for game in data.get('games', []):
            if game.get('gameType') == 2:
                all_ids.add(game['id'])
        time.sleep(0.2)
    
    return list(all_ids)

def parse_pbp_for_pk_events(game_id):
    """Parse play by play and extract all PK events with player attribution"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except:
        return []
    
    plays = data.get('plays', [])
    roster = data.get('rosterSpots', [])
    
    # Build player ID to name mapping
    player_map = {}
    for player in roster:
        pid = player.get('playerId')
        fname = player.get('firstName', {}).get('default', '')
        lname = player.get('lastName', {}).get('default', '')
        team = player.get('teamId')
        player_map[pid] = {
            'name': f"{fname} {lname}".strip(),
            'team': team
        }
    
    # Build team ID to abbrev mapping
    home_team_id = data.get('homeTeam', {}).get('id')
    away_team_id = data.get('awayTeam', {}).get('id')
    home_abbrev = data.get('homeTeam', {}).get('abbrev', '')
    away_abbrev = data.get('awayTeam', {}).get('abbrev', '')
    
    team_map = {
        home_team_id: home_abbrev,
        away_team_id: away_abbrev
    }
    
    pk_events = []
    
    for play in plays:
        situation = play.get('situationCode', '1551')
        type_key = play.get('typeDescKey', '')
        details = play.get('details', {})
        period = play.get('periodDescriptor', {}).get('number', 0)
        time_in = play.get('timeInPeriod', '0:00')
        
        # Parse situation code: [home_goalie][home_skaters][away_skaters][away_goalie]
        # e.g. 1451 = home has goalie + 4 skaters, away has 5 skaters + goalie
        # PK situation: one team has 4 skaters (or 3)
        if len(situation) == 4:
            home_goalie = int(situation[0])
            home_skaters = int(situation[1])
            away_skaters = int(situation[2])
            away_goalie = int(situation[3])
            
            home_pk = home_skaters < away_skaters  # home team is shorthanded
            away_pk = away_skaters < home_skaters  # away team is shorthanded
            
            is_pk = home_pk or away_pk
        else:
            is_pk = False
        
        if not is_pk:
            continue
        
        # Determine which team is killing and which is on PP
        if home_pk:
            pk_team_id = home_team_id
            pk_team = home_abbrev
            pp_team = away_abbrev
        else:
            pk_team_id = away_team_id
            pk_team = away_abbrev
            pp_team = home_abbrev
        
        event = {
            'game_id': game_id,
            'period': period,
            'time': time_in,
            'situation': situation,
            'pk_team': pk_team,
            'pp_team': pp_team,
            'event_type': type_key,
        }
        
        # Extract player info based on event type
        if type_key == 'blocked-shot':
            # Blocker is the PK player we care about
            blocker_id = details.get('blockingPlayerId')
            shooter_id = details.get('shootingPlayerId')
            if blocker_id and blocker_id in player_map:
                blocker = player_map[blocker_id]
                event['player_id'] = blocker_id
                event['player_name'] = blocker['name']
                event['player_team'] = team_map.get(blocker['team'], '')
                # Only count if blocker is on PK team
                if event['player_team'] == pk_team:
                    event['stat'] = 'pk_block'
                    pk_events.append(event.copy())
        
        elif type_key == 'takeaway':
            player_id = details.get('playerId')
            if player_id and player_id in player_map:
                p = player_map[player_id]
                event['player_id'] = player_id
                event['player_name'] = p['name']
                event['player_team'] = team_map.get(p['team'], '')
                if event['player_team'] == pk_team:
                    event['stat'] = 'pk_takeaway'
                    pk_events.append(event.copy())
        
        elif type_key == 'giveaway':
            player_id = details.get('playerId')
            if player_id and player_id in player_map:
                p = player_map[player_id]
                event['player_id'] = player_id
                event['player_name'] = p['name']
                event['player_team'] = team_map.get(p['team'], '')
                if event['player_team'] == pk_team:
                    event['stat'] = 'pk_giveaway'
                    pk_events.append(event.copy())
        
        elif type_key == 'hit':
            hitter_id = details.get('hittingPlayerId')
            if hitter_id and hitter_id in player_map:
                p = player_map[hitter_id]
                event['player_id'] = hitter_id
                event['player_name'] = p['name']
                event['player_team'] = team_map.get(p['team'], '')
                if event['player_team'] == pk_team:
                    event['stat'] = 'pk_hit'
                    pk_events.append(event.copy())
        
        elif type_key == 'faceoff':
            winner_id = details.get('winningPlayerId')
            loser_id = details.get('losingPlayerId')
            
            if winner_id and winner_id in player_map:
                p = player_map[winner_id]
                event_copy = event.copy()
                event_copy['player_id'] = winner_id
                event_copy['player_name'] = p['name']
                event_copy['player_team'] = team_map.get(p['team'], '')
                if event_copy['player_team'] == pk_team:
                    event_copy['stat'] = 'pk_faceoff_win'
                    pk_events.append(event_copy)
            
            if loser_id and loser_id in player_map:
                p = player_map[loser_id]
                event_copy = event.copy()
                event_copy['player_id'] = loser_id
                event_copy['player_name'] = p['name']
                event_copy['player_team'] = team_map.get(p['team'], '')
                if event_copy['player_team'] == pk_team:
                    event_copy['stat'] = 'pk_faceoff_loss'
                    pk_events.append(event_copy)
        
        elif type_key == 'goal':
            # Goal against — attribute to all PK players on ice
            event['stat'] = 'pk_goal_against'
            event['player_id'] = None
            event['player_name'] = 'TEAM'
            event['player_team'] = pk_team
            pk_events.append(event.copy())
        
        elif type_key == 'penalty':
            # Penalty taken while shorthanded — catastrophic
            committer_id = details.get('committedByPlayerId')
            if committer_id and committer_id in player_map:
                p = player_map[committer_id]
                event['player_id'] = committer_id
                event['player_name'] = p['name']
                event['player_team'] = team_map.get(p['team'], '')
                if event['player_team'] == pk_team:
                    event['stat'] = 'pk_penalty_taken'
                    pk_events.append(event.copy())
        
        elif type_key in ['shot-on-goal', 'missed-shot', 'blocked-shot']:
            # Track shots against (for the PK team)
            shooter_id = details.get('shootingPlayerId') or details.get('playerId')
            if shooter_id and shooter_id in player_map:
                p = player_map[shooter_id]
                shooter_team = team_map.get(p['team'], '')
                if shooter_team == pp_team:  # shot by PP team = against PK
                    event['player_id'] = None
                    event['player_name'] = 'SHOT_AGAINST'
                    event['player_team'] = pk_team
                    event['stat'] = 'pk_shot_against'
                    pk_events.append(event.copy())
    
    return pk_events

def aggregate_player_pk_stats(pk_events):
    """Aggregate raw events into per-player PK stats"""
    player_stats = defaultdict(lambda: {
        'pk_blocks': 0,
        'pk_takeaways': 0,
        'pk_giveaways': 0,
        'pk_hits': 0,
        'pk_faceoff_wins': 0,
        'pk_faceoff_losses': 0,
        'pk_penalties_taken': 0,
    })
    
    for event in pk_events:
        if not event.get('player_name') or event['player_name'] in ['TEAM', 'SHOT_AGAINST']:
            continue
        
        key = (event['player_name'], event['player_team'])
        stat = event['stat']
        
        if stat == 'pk_block':
            player_stats[key]['pk_blocks'] += 1
        elif stat == 'pk_takeaway':
            player_stats[key]['pk_takeaways'] += 1
        elif stat == 'pk_giveaway':
            player_stats[key]['pk_giveaways'] += 1
        elif stat == 'pk_hit':
            player_stats[key]['pk_hits'] += 1
        elif stat == 'pk_faceoff_win':
            player_stats[key]['pk_faceoff_wins'] += 1
        elif stat == 'pk_faceoff_loss':
            player_stats[key]['pk_faceoff_losses'] += 1
        elif stat == 'pk_penalty_taken':
            player_stats[key]['pk_penalties_taken'] += 1
    
    rows = []
    for (name, team), stats in player_stats.items():
        row = {'player': name, 'team': team}
        row.update(stats)
        fo_total = stats['pk_faceoff_wins'] + stats['pk_faceoff_losses']
        row['pk_faceoff_pct'] = (stats['pk_faceoff_wins'] / fo_total * 100) if fo_total > 0 else None
        rows.append(row)
    
    return pd.DataFrame(rows)

def collect_season_pbp(season, max_games=None):
    """Collect PBP data for a full season"""
    print(f"\nCollecting game IDs for {season}...")
    game_ids = get_all_game_ids_for_season(season)
    print(f"Found {len(game_ids)} games")
    
    if max_games:
        game_ids = game_ids[:max_games]
        print(f"Limited to {max_games} games for testing")
    
    all_events = []
    
    for i, game_id in enumerate(game_ids):
        events = parse_pbp_for_pk_events(game_id)
        all_events.extend(events)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(game_ids)} games, {len(all_events)} PK events so far")
        
        time.sleep(0.1)
    
    print(f"Total PK events: {len(all_events)}")
    df_stats = aggregate_player_pk_stats(all_events)
    df_stats['season'] = season
    return df_stats

if __name__ == "__main__":
    SEASONS = ["20212022", "20222023", "20232024", "20242025"]
    
    all_seasons = []
    
    for season in SEASONS:
        df = collect_season_pbp(season)
        all_seasons.append(df)
        df.to_csv(f"pk_pbp_{season}.csv", index=False)
        print(f"Saved pk_pbp_{season}.csv")
        time.sleep(2)
    
    combined = pd.concat(all_seasons, ignore_index=True)
    combined.to_csv("pk_pbp_all_seasons.csv", index=False)
    print(f"\nDone. Total records: {len(combined)}")
    print(combined.head(5).to_string(index=False))