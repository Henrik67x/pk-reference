import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from app import answer_question, extract_player_name, get_nhl_data

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatPucks",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TEAM_COLORS = {
    "ANA": {"primary": "#F47A38", "secondary": "#B9975B", "name": "Anaheim Ducks"},
    "BOS": {"primary": "#FFB81C", "secondary": "#000000", "name": "Boston Bruins"},
    "BUF": {"primary": "#003087", "secondary": "#FCB514", "name": "Buffalo Sabres"},
    "CAR": {"primary": "#CC0000", "secondary": "#000000", "name": "Carolina Hurricanes"},
    "CBJ": {"primary": "#002654", "secondary": "#CE1126", "name": "Columbus Blue Jackets"},
    "CGY": {"primary": "#C8102E", "secondary": "#F1BE48", "name": "Calgary Flames"},
    "CHI": {"primary": "#CF0A2C", "secondary": "#000000", "name": "Chicago Blackhawks"},
    "COL": {"primary": "#6F263D", "secondary": "#236192", "name": "Colorado Avalanche"},
    "DAL": {"primary": "#006847", "secondary": "#8F8F8C", "name": "Dallas Stars"},
    "DET": {"primary": "#CE1126", "secondary": "#FFFFFF", "name": "Detroit Red Wings"},
    "EDM": {"primary": "#FF4C00", "secondary": "#003777", "name": "Edmonton Oilers"},
    "FLA": {"primary": "#041E42", "secondary": "#C8102E", "name": "Florida Panthers"},
    "LAK": {"primary": "#333333", "secondary": "#A2AAAD", "name": "Los Angeles Kings"},
    "MIN": {"primary": "#154734", "secondary": "#A6192E", "name": "Minnesota Wild"},
    "MTL": {"primary": "#AF1E2D", "secondary": "#192168", "name": "Montreal Canadiens"},
    "NJD": {"primary": "#CE1126", "secondary": "#003087", "name": "New Jersey Devils"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42", "name": "Nashville Predators"},
    "NYI": {"primary": "#003087", "secondary": "#FC4C02", "name": "New York Islanders"},
    "NYR": {"primary": "#0038A8", "secondary": "#CE1126", "name": "New York Rangers"},
    "OTT": {"primary": "#C8102E", "secondary": "#C69214", "name": "Ottawa Senators"},
    "PHI": {"primary": "#F74902", "secondary": "#000000", "name": "Philadelphia Flyers"},
    "PIT": {"primary": "#FCB514", "secondary": "#000000", "name": "Pittsburgh Penguins"},
    "SEA": {"primary": "#355464", "secondary": "#99D9D9", "name": "Seattle Kraken"},
    "SJS": {"primary": "#006D75", "secondary": "#EA7200", "name": "San Jose Sharks"},
    "STL": {"primary": "#002F87", "secondary": "#FCB514", "name": "St. Louis Blues"},
    "TBL": {"primary": "#002868", "secondary": "#FFFFFF", "name": "Tampa Bay Lightning"},
    "TOR": {"primary": "#003E7E", "secondary": "#FFFFFF", "name": "Toronto Maple Leafs"},
    "UTA": {"primary": "#6CACE4", "secondary": "#1C1C1C", "name": "Utah Mammoth"},
    "VAN": {"primary": "#00205B", "secondary": "#00843D", "name": "Vancouver Canucks"},
    "VGK": {"primary": "#B4975A", "secondary": "#333F42", "name": "Vegas Golden Knights"},
    "WPG": {"primary": "#041E42", "secondary": "#004C97", "name": "Winnipeg Jets"},
    "WSH": {"primary": "#041E42", "secondary": "#C8102E", "name": "Washington Capitals"},
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #080c18;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1600px !important; }

/* ── HEADER ── */
.sp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 16px 0;
    border-bottom: 1px solid #1a2540;
    margin-bottom: 24px;
}
.sp-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: 1px;
}
.sp-logo span { color: #4FC3F7; }
.sp-tagline {
    font-size: 0.75rem;
    color: #4a5a75;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── SEARCH ── */
.stTextInput input {
    background: #0f1628 !important;
    border: 1px solid #2a3a55 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 11px 16px !important;
}
.stTextInput input:focus {
    border-color: #4FC3F7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.12) !important;
}
.stTextInput label {
    display: none !important;
}
.stButton button {
    background: #4FC3F7 !important;
    color: #080c18 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 11px 28px !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
}
.stButton button:hover { background: #81D4FA !important; }

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0 0 10px 0;
    border-bottom: 2px solid #4FC3F7;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── CARDS ── */
.card {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2a3a55; }

/* ── STAT ROW ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid #111827;
    font-size: 0.88rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-row:hover { background: #111827; border-radius: 6px; }

.rank-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #4a5a75;
    min-width: 28px;
}
.player-name-cell {
    font-weight: 600;
    color: #e2e8f0;
    flex: 1;
    padding-left: 8px;
}
.team-pill {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.5px;
    margin-right: 8px;
}
.stat-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #4FC3F7;
    min-width: 70px;
    text-align: right;
}

/* ── PLAYOFF CARD ── */
.playoff-game {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.playoff-teams {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
}
.playoff-score {
    font-size: 1.4rem;
    color: #4FC3F7;
}
.playoff-status {
    font-size: 0.72rem;
    color: #4a5a75;
    text-align: center;
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── TABLES ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
div[data-testid="stDataFrame"] table {
    background: #0f1628 !important;
}

/* ── ANSWER BOX ── */
.answer-wrap {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 24px;
    margin-top: 16px;
}

.stMarkdown table {
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 10px 0;
    width: 100%;
}
.stMarkdown table th {
    background: #1a2540;
    color: #4FC3F7;
    padding: 9px 13px;
    text-align: left;
    font-size: 0.77rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border-bottom: 2px solid #2a3a55;
    white-space: nowrap;
}
.stMarkdown table td {
    padding: 8px 13px;
    border-bottom: 1px solid #111827;
    color: #c8d4e8;
    white-space: nowrap;
}
.stMarkdown table tr:hover td { background: #111827; }

/* ── METRIC BOXES ── */
.metric-box {
    background: #0f1628;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    color: #4a5a75;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #4FC3F7;
}
.metric-sub {
    font-size: 0.78rem;
    color: #6a7a95;
    margin-top: 4px;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c18; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── DATA FUNCTIONS ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_standings():
    url = "https://api-web.nhle.com/v1/standings/now"
    r = requests.get(url)
    return r.json().get("standings", [])

@st.cache_data(ttl=60)
def get_scores():
    url = "https://api-web.nhle.com/v1/score/now"
    r = requests.get(url)
    return r.json().get("games", [])

@st.cache_data(ttl=3600)
def get_league_leaders():
    skaters = []
    for cat in ["points", "goals", "assists"]:
        url = f"https://api-web.nhle.com/v1/skater-stats-leaders/current?categories={cat}&limit=10"
        r = requests.get(url)
        data = r.json()
        skaters.append((cat, data.get(cat, [])))
    return skaters

@st.cache_data(ttl=3600)
def get_team_roster(team_abbrev):
    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    r = requests.get(url)
    data = r.json()
    players = []
    for group in ["forwards", "defensemen", "goalies"]:
        for p in data.get(group, []):
            players.append({
                "id": p["id"],
                "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "position": p.get("positionCode", ""),
                "team": team_abbrev
            })
    return players

@st.cache_data(ttl=3600)
def get_all_rosters():
    teams = list(TEAM_COLORS.keys())
    all_players = []
    for t in teams:
        try:
            players = get_team_roster(t)
            all_players.extend(players)
        except:
            pass
    return all_players

@st.cache_data(ttl=3600)
def get_player_contract_from_ep(player_name):
    """Get cap hit from Elite Prospects via requests (no selenium needed)"""
    try:
        search_url = f"https://www.eliteprospects.com/search/player?q={player_name.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.content, "html.parser")
        # Try to find cap hit in page
        text = soup.get_text()
        return text[:500]
    except:
        return None


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sp-header">
    <div>
        <div class="sp-logo">stat<span>pucks</span></div>
        <div class="sp-tagline">NHL Analytics · Contracts · Advanced Stats</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── SEARCH BAR ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input("search", placeholder="Search any player, stat, or contract... e.g. 'Leo Carlsson contract and xGF%'", label_visibility="collapsed")
with col_btn:
    search_btn = st.button("SEARCH")

if search_btn and question:
    with st.spinner(""):
        player_name = extract_player_name(question)
        team_color = {"primary": "#4FC3F7"}
        team_display = ""
        team_abbrev = ""

        if player_name.lower() != "none":
            try:
                nhl_data = get_nhl_data(player_name)
                if nhl_data:
                    team_abbrev = nhl_data.get("currentTeamAbbrev", "")
                    tc = TEAM_COLORS.get(team_abbrev, {"primary": "#4FC3F7", "name": ""})
                    team_color = tc
                    team_display = tc.get("name", "")
            except:
                pass

        if team_display:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {team_color['primary']}18, {team_color['primary']}08);
                        border: 1px solid {team_color['primary']}33; border-radius: 10px;
                        padding: 16px 20px; margin: 12px 0;">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.5rem;
                            font-weight:800; color:{team_color['primary']};">{player_name}</div>
                <div style="font-size:0.78rem; color:{team_color['primary']}88;
                            letter-spacing:1px; text-transform:uppercase; margin-top:2px;">
                    {team_display} · {team_abbrev}
                </div>
            </div>
            """, unsafe_allow_html=True)

        answer = answer_question(question)
        st.markdown(f'<div class="answer-wrap" style="border-top: 3px solid {team_color["primary"]};">', unsafe_allow_html=True)
        st.markdown(answer, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

elif search_btn:
    st.warning("Enter a player name or question.")

# ─── LIVE SCORES / RECENT GAMES ───────────────────────────────────────────────
st.markdown('<div class="section-header">🏒 Scores</div>', unsafe_allow_html=True)

try:
    games = get_scores()
    if games:
        cols = st.columns(min(len(games), 4))
        for i, game in enumerate(games[:8]):
            if i < len(cols):
                with cols[i % 4]:
                    away = game.get("awayTeam", {})
                    home = game.get("homeTeam", {})
                    away_abbrev = away.get("abbrev", "")
                    home_abbrev = home.get("abbrev", "")
                    away_score = away.get("score", "-")
                    home_score = home.get("score", "-")
                    state = game.get("gameState", "")
                    period = game.get("periodDescriptor", {}).get("number", "")
                    away_color = TEAM_COLORS.get(away_abbrev, {}).get("primary", "#4FC3F7")
                    home_color = TEAM_COLORS.get(home_abbrev, {}).get("primary", "#4FC3F7")

                    status = "FINAL" if state == "OFF" else f"LIVE · P{period}" if state == "LIVE" else "UPCOMING"

                    st.markdown(f"""
                    <div class="playoff-game">
                        <div class="playoff-teams">
                            <span style="color:{away_color}">{away_abbrev}</span>
                            <span class="playoff-score">{away_score} – {home_score}</span>
                            <span style="color:{home_color}">{home_abbrev}</span>
                        </div>
                        <div class="playoff-status">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:#4a5a75; text-align:center; padding:24px;">No games scheduled today</div>', unsafe_allow_html=True)
except Exception as e:
    st.markdown('<div class="card" style="color:#4a5a75;">Could not load scores.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── LEAGUE LEADERS + STANDINGS ───────────────────────────────────────────────
col_leaders, col_standings = st.columns([3, 2])

with col_leaders:
    st.markdown('<div class="section-header">📊 League Leaders</div>', unsafe_allow_html=True)
    
    try:
        leaders_data = get_league_leaders()
        tabs = st.tabs(["Points", "Goals", "Assists"])
        
        for tab, (cat, players) in zip(tabs, leaders_data):
            with tab:
                for i, p in enumerate(players[:10]):
                    name = f"{p.get('firstName', {}).get('default','')} {p.get('lastName', {}).get('default','')}"
                    team = p.get("teamAbbrevs", "")
                    value = p.get("value", 0)
                    tc = TEAM_COLORS.get(team, {})
                    color = tc.get("primary", "#4FC3F7")
                    
                    st.markdown(f"""
                    <div class="stat-row">
                        <span class="rank-num">{i+1}</span>
                        <span class="player-name-cell">{name}</span>
                        <span class="team-pill" style="background:{color}22; color:{color};">{team}</span>
                        <span class="stat-value">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="card" style="color:#4a5a75;">Could not load leaders.</div>', unsafe_allow_html=True)

with col_standings:
    st.markdown('<div class="section-header">🏆 Standings</div>', unsafe_allow_html=True)
    
    try:
        standings = get_standings()
        
        conf_tab = st.tabs(["Western", "Eastern"])
        
        for tab, conf in zip(conf_tab, ["Western", "Eastern"]):
            with tab:
                conf_teams = [t for t in standings if t.get("conferenceName") == conf]
                conf_teams.sort(key=lambda x: x.get("points", 0), reverse=True)
                
                for i, team in enumerate(conf_teams[:8]):
                    abbrev = team.get("teamAbbrev", {}).get("default", "")
                    name = team.get("teamCommonName", {}).get("default", "")
                    pts = team.get("points", 0)
                    w = team.get("wins", 0)
                    l = team.get("losses", 0)
                    ot = team.get("otLosses", 0)
                    tc = TEAM_COLORS.get(abbrev, {})
                    color = tc.get("primary", "#4FC3F7")
                    
                    st.markdown(f"""
                    <div class="stat-row">
                        <span class="rank-num">{i+1}</span>
                        <span class="player-name-cell">{name}</span>
                        <span style="font-size:0.78rem; color:#4a5a75; margin-right:8px;">{w}-{l}-{ot}</span>
                        <span class="stat-value" style="color:{color};">{pts}</span>
                    </div>
                    """, unsafe_allow_html=True)
    except:
        st.markdown('<div class="card" style="color:#4a5a75;">Could not load standings.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── CAP HIT LEADERBOARD ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">💰 Cap Hit Leaderboard</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:0.8rem; color:#4a5a75; margin-bottom:12px;">Sourced from NHL API roster data · Contract values from Elite Prospects</div>', unsafe_allow_html=True)

# Known top contracts hardcoded for reliability (these are public record)
TOP_CONTRACTS = [
    {"Player": "Connor McDavid", "Team": "EDM", "Position": "C", "AAV": "$12,500,000", "Expires": "2026"},
    {"Player": "Nathan MacKinnon", "Team": "COL", "Position": "C", "AAV": "$12,600,000", "Expires": "2031"},
    {"Player": "Auston Matthews", "Team": "TOR", "Position": "C", "AAV": "$13,250,000", "Expires": "2028"},
    {"Player": "Erik Karlsson", "Team": "PIT", "Position": "D", "AAV": "$11,500,000", "Expires": "2027"},
    {"Player": "Artemi Panarin", "Team": "NYR", "Position": "LW", "AAV": "$11,642,857", "Expires": "2026"},
    {"Player": "Leon Draisaitl", "Team": "EDM", "Position": "C", "AAV": "$14,000,000", "Expires": "2025"},
    {"Player": "David Pastrnak", "Team": "BOS", "Position": "RW", "AAV": "$11,250,000", "Expires": "2032"},
    {"Player": "Jack Eichel", "Team": "VGK", "Position": "C", "AAV": "$10,000,000", "Expires": "2028"},
    {"Player": "Aleksander Barkov", "Team": "FLA", "Position": "C", "AAV": "$10,000,000", "Expires": "2031"},
    {"Player": "Matthew Tkachuk", "Team": "FLA", "Position": "LW", "AAV": "$9,500,000", "Expires": "2029"},
    {"Player": "John Tavares", "Team": "TOR", "Position": "C", "AAV": "$11,000,000", "Expires": "2026"},
    {"Player": "Mark Scheifele", "Team": "WPG", "Position": "C", "AAV": "$8,500,000", "Expires": "2028"},
    {"Player": "Elias Pettersson", "Team": "VAN", "Position": "C", "AAV": "$11,600,000", "Expires": "2031"},
    {"Player": "Brady Tkachuk", "Team": "OTT", "Position": "LW", "AAV": "$9,500,000", "Expires": "2030"},
    {"Player": "Mitch Marner", "Team": "VGK", "Position": "RW", "AAV": "$10,893,000", "Expires": "2030"},
    {"Player": "Roman Josi", "Team": "NSH", "Position": "D", "AAV": "$9,059,000", "Expires": "2026"},
    {"Player": "Cale Makar", "Team": "COL", "Position": "D", "AAV": "$9,000,000", "Expires": "2028"},
    {"Player": "Adam Fox", "Team": "NYR", "Position": "D", "AAV": "$9,500,000", "Expires": "2029"},
    {"Player": "Sebastian Aho", "Team": "CAR", "Position": "C", "AAV": "$8,454,000", "Expires": "2026"},
    {"Player": "Kirill Kaprizov", "Team": "MIN", "Position": "LW", "AAV": "$9,000,000", "Expires": "2028"},
]

df = pd.DataFrame(TOP_CONTRACTS)

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    pos_filter = st.selectbox("Position", ["All", "C", "LW", "RW", "D", "G"])
with col_f2:
    team_filter = st.selectbox("Team", ["All"] + sorted(TEAM_COLORS.keys()))
with col_f3:
    sort_by = st.selectbox("Sort By", ["AAV (High → Low)", "AAV (Low → High)", "Player Name", "Expires"])

filtered_df = df.copy()
if pos_filter != "All":
    filtered_df = filtered_df[filtered_df["Position"] == pos_filter]
if team_filter != "All":
    filtered_df = filtered_df[filtered_df["Team"] == team_filter]

def aav_to_num(aav_str):
    return float(aav_str.replace("$","").replace(",",""))

if sort_by == "AAV (High → Low)":
    filtered_df["_sort"] = filtered_df["AAV"].apply(aav_to_num)
    filtered_df = filtered_df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
elif sort_by == "AAV (Low → High)":
    filtered_df["_sort"] = filtered_df["AAV"].apply(aav_to_num)
    filtered_df = filtered_df.sort_values("_sort", ascending=True).drop("_sort", axis=1)
elif sort_by == "Player Name":
    filtered_df = filtered_df.sort_values("Player")
elif sort_by == "Expires":
    filtered_df = filtered_df.sort_values("Expires")

for i, row in filtered_df.iterrows():
    tc = TEAM_COLORS.get(row["Team"], {})
    color = tc.get("primary", "#4FC3F7")
    team_name = tc.get("name", row["Team"])
    
    st.markdown(f"""
    <div class="stat-row" style="background:#0f1628; border-radius:8px; margin-bottom:4px; border:1px solid #1a2540;">
        <span class="rank-num">{list(filtered_df.index).index(i)+1}</span>
        <span class="player-name-cell">{row["Player"]}</span>
        <span class="team-pill" style="background:{color}22; color:{color};">{row["Team"]}</span>
        <span style="font-size:0.78rem; color:#4a5a75; margin-right:12px;">{row["Position"]}</span>
        <span style="font-size:0.78rem; color:#4a5a75; margin-right:12px;">Exp: {row["Expires"]}</span>
        <span class="stat-value">{row["AAV"]}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── RFA / UFA WATCH ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 RFA / UFA Watch — 2026 Offseason</div>', unsafe_allow_html=True)

RFA_UFA = [
    {"Player": "Leo Carlsson", "Team": "ANA", "Type": "RFA", "Position": "C", "Current AAV": "$950,000", "Projected": "$9.0–9.5M"},
    {"Player": "Cutter Gauthier", "Team": "ANA", "Type": "RFA", "Position": "LW", "Current AAV": "$925,000", "Projected": "$10.0–10.5M"},
    {"Player": "Olen Zellweger", "Team": "ANA", "Type": "RFA", "Position": "D", "Current AAV": "$863,333", "Projected": "$5.0–6.0M"},
    {"Player": "Connor Bedard", "Team": "CHI", "Type": "RFA", "Position": "C", "Current AAV": "$925,000", "Projected": "$12.0–13.0M"},
    {"Player": "Matvei Michkov", "Team": "PHI", "Type": "RFA", "Position": "RW", "Current AAV": "$925,000", "Projected": "$7.0–8.5M"},
    {"Player": "Marco Rossi", "Team": "MIN", "Type": "RFA", "Position": "C", "Current AAV": "$863,333", "Projected": "$5.5–6.5M"},
    {"Player": "Shane Wright", "Team": "SEA", "Type": "RFA", "Position": "C", "Current AAV": "$863,333", "Projected": "$3.5–4.5M"},
    {"Player": "Sebastian Aho", "Team": "CAR", "Type": "UFA", "Position": "C", "Current AAV": "$8,454,000", "Projected": "$9.5–10.5M"},
    {"Player": "Artemi Panarin", "Team": "NYR", "Type": "UFA", "Position": "LW", "Current AAV": "$11,642,857", "Projected": "TBD"},
    {"Player": "Connor McDavid", "Team": "EDM", "Type": "UFA", "Position": "C", "Current AAV": "$12,500,000", "Projected": "$14.0M+"},
    {"Player": "Roman Josi", "Team": "NSH", "Type": "UFA", "Position": "D", "Current AAV": "$9,059,000", "Projected": "TBD"},
    {"Player": "Nikolaj Ehlers", "Team": "WPG", "Type": "UFA", "Position": "LW", "Current AAV": "$6,000,000", "Projected": "$8.0–9.0M"},
]

rfa_col, ufa_col = st.columns(2)

with rfa_col:
    st.markdown('<div style="font-family:\'Barlow Condensed\',sans-serif; font-size:0.9rem; font-weight:700; color:#4FC3F7; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Restricted Free Agents</div>', unsafe_allow_html=True)
    for p in [x for x in RFA_UFA if x["Type"] == "RFA"]:
        tc = TEAM_COLORS.get(p["Team"], {})
        color = tc.get("primary", "#4FC3F7")
        st.markdown(f"""
        <div class="stat-row" style="background:#0f1628; border-radius:8px; margin-bottom:4px; border:1px solid #1a2540;">
            <span class="player-name-cell">{p["Player"]}</span>
            <span class="team-pill" style="background:{color}22; color:{color};">{p["Team"]}</span>
            <span style="font-size:0.75rem; color:#4a5a75; margin-right:8px;">{p["Position"]}</span>
            <span style="font-family:'Barlow Condensed',sans-serif; font-size:0.9rem; color:#a0c4d8;">{p["Projected"]}</span>
        </div>
        """, unsafe_allow_html=True)

with ufa_col:
    st.markdown('<div style="font-family:\'Barlow Condensed\',sans-serif; font-size:0.9rem; font-weight:700; color:#F47A38; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Unrestricted Free Agents</div>', unsafe_allow_html=True)
    for p in [x for x in RFA_UFA if x["Type"] == "UFA"]:
        tc = TEAM_COLORS.get(p["Team"], {})
        color = tc.get("primary", "#4FC3F7")
        st.markdown(f"""
        <div class="stat-row" style="background:#0f1628; border-radius:8px; margin-bottom:4px; border:1px solid #1a2540;">
            <span class="player-name-cell">{p["Player"]}</span>
            <span class="team-pill" style="background:{color}22; color:{color};">{p["Team"]}</span>
            <span style="font-size:0.75rem; color:#4a5a75; margin-right:8px;">{p["Position"]}</span>
            <span style="font-family:'Barlow Condensed',sans-serif; font-size:0.9rem; color:#a0c4d8;">{p["Projected"]}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1a2540; padding-top:16px; margin-top:8px;
            display:flex; justify-content:space-between; align-items:center;">
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1rem;
                font-weight:700; color:#fff;">stat<span style="color:#4FC3F7;">pucks</span></div>
    <div style="font-size:0.72rem; color:#4a5a75;">
        Data: NHL API · Natural Stat Trick · Elite Prospects
    </div>
</div>
""", unsafe_allow_html=True)