import streamlit as st
from app import answer_question, get_nhl_data, extract_player_name

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
    "LAK": {"primary": "#111111", "secondary": "#A2AAAD", "name": "Los Angeles Kings"},
    "MIN": {"primary": "#154734", "secondary": "#A6192E", "name": "Minnesota Wild"},
    "MTL": {"primary": "#AF1E2D", "secondary": "#192168", "name": "Montreal Canadiens"},
    "NJD": {"primary": "#CE1126", "secondary": "#003087", "name": "New Jersey Devils"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42", "name": "Nashville Predators"},
    "NYI": {"primary": "#003087", "secondary": "#FC4C02", "name": "New York Islanders"},
    "NYR": {"primary": "#0038A8", "secondary": "#CE1126", "name": "New York Rangers"},
    "OTT": {"primary": "#C8102E", "secondary": "#C69214", "name": "Ottawa Senators"},
    "PHI": {"primary": "#F74902", "secondary": "#000000", "name": "Philadelphia Flyers"},
    "PIT": {"primary": "#FCB514", "secondary": "#000000", "name": "Pittsburgh Penguins"},
    "SEA": {"primary": "#001628", "secondary": "#99D9D9", "name": "Seattle Kraken"},
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

DEFAULT_COLOR = {"primary": "#4FC3F7", "secondary": "#1a1a2e"}

GLOSSARY = {
    "GP": "Games Played",
    "TOI": "Time on Ice (minutes)",
    "G": "Goals",
    "A": "Assists",
    "PTS": "Points",
    "+/-": "Plus/Minus",
    "PIM": "Penalties in Minutes",
    "PPG": "Power Play Goals",
    "PPP": "Power Play Points",
    "SHG": "Shorthanded Goals",
    "GWG": "Game Winning Goals",
    "S%": "Shooting Percentage",
    "CF": "Corsi For (all shot attempts for)",
    "CA": "Corsi Against (all shot attempts against)",
    "CF%": "Corsi For Percentage (CF / (CF + CA))",
    "FF": "Fenwick For (unblocked shot attempts for)",
    "FA": "Fenwick Against (unblocked shot attempts against)",
    "FF%": "Fenwick For Percentage",
    "xGF": "Expected Goals For",
    "xGA": "Expected Goals Against",
    "xGF%": "Expected Goals For Percentage",
    "SCF": "Scoring Chances For",
    "SCA": "Scoring Chances Against",
    "SCF%": "Scoring Chances For Percentage",
    "HDCF": "High Danger Corsi For",
    "HDCA": "High Danger Corsi Against",
    "HDCF%": "High Danger Corsi For Percentage",
    "HDGF": "High Danger Goals For",
    "HDGA": "High Danger Goals Against",
    "iCF": "Individual Corsi For",
    "iFF": "Individual Fenwick For",
    "iSCF": "Individual Scoring Chances For",
    "ixG": "Individual Expected Goals",
    "IPP": "Individual Points Percentage",
    "OZS%": "Offensive Zone Start Percentage",
    "PDO": "PDO (on-ice SH% + on-ice SV%, luck indicator)",
    "FO%": "Faceoff Win Percentage",
    "Cap Hit": "Average Annual Value against the salary cap",
    "AAV": "Average Annual Value of contract",
    "RFA": "Restricted Free Agent",
    "UFA": "Unrestricted Free Agent",
    "ELC": "Entry Level Contract",
    "NMC": "No Movement Clause",
    "NTC": "No Trade Clause",
}

st.set_page_config(
    page_title="PuckIQ",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

.puckiq-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 20px 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 28px;
}

.puckiq-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
    margin: 0;
}

.puckiq-title span { color: #4FC3F7; }

.puckiq-subtitle {
    font-size: 0.85rem;
    color: #5a6a80;
    margin: 2px 0 0 0;
    font-weight: 400;
}

.search-section {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}

.stTextInput input {
    background: #0a0e1a !important;
    border: 1px solid #2a3a4a !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}

.stTextInput input:focus {
    border-color: #4FC3F7 !important;
    box-shadow: 0 0 0 2px rgba(79, 195, 247, 0.15) !important;
}

.stTextInput label {
    color: #8a9ab0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

.stButton button {
    background: #4FC3F7 !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 10px 32px !important;
    text-transform: uppercase !important;
}

.stButton button:hover {
    background: #81D4FA !important;
}

.team-banner {
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
}

.team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.team-badge {
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
}

.answer-box {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 28px;
}

/* Tables — scrollable horizontally but no box */
.stMarkdown table {
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 12px 0;
    min-width: 100%;
}

.stMarkdown table th {
    background: #1e2a3a;
    color: #4FC3F7;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border-bottom: 2px solid #2a3a4a;
    white-space: nowrap;
}

.stMarkdown table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1a2535;
    color: #d0d8e8;
    white-space: nowrap;
}

.stMarkdown table tr:hover td {
    background: #141c2a;
}

/* Wrap the table in a scrollable div via JS-free CSS */
.element-container:has(table) {
    overflow-x: auto;
    width: 100%;
}

.sources-bar {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}

.source-badge {
    background: #1a2535;
    border: 1px solid #2a3a4a;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #5a9fc8;
    font-weight: 500;
}

.glossary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 8px;
    padding: 16px 0;
}

.glossary-item {
    background: #0f1622;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 8px 12px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

.glossary-term {
    color: #4FC3F7;
    font-weight: 600;
    font-size: 0.82rem;
    min-width: 60px;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.5px;
}

.glossary-def {
    color: #7a8a9a;
    font-size: 0.8rem;
    line-height: 1.4;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2a3a4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="puckiq-header">
    <div>
        <h1 class="puckiq-title">Puck<span>IQ</span></h1>
        <p class="puckiq-subtitle">NHL Analytics Engine — Stats · Advanced Metrics · Career History</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Search
st.markdown('<div class="search-section">', unsafe_allow_html=True)
question = st.text_input("ASK ANYTHING ABOUT ANY NHL PLAYER", placeholder='e.g. "What is Leo Carlsson\'s xGF% and career history?"')
col1, col2 = st.columns([1, 6])
with col1:
    search_btn = st.button("SEARCH")
st.markdown('</div>', unsafe_allow_html=True)

# Sources
st.markdown("""
<div class="sources-bar">
    <span class="source-badge">📊 NHL API</span>
    <span class="source-badge">📈 Natural Stat Trick</span>
    <span class="source-badge">🏒 Elite Prospects</span>
</div>
""", unsafe_allow_html=True)

# Glossary
glossary_html = '<div class="glossary-grid">'
for term, definition in GLOSSARY.items():
    glossary_html += f'<div class="glossary-item"><span class="glossary-term">{term}</span><span class="glossary-def">{definition}</span></div>'
glossary_html += '</div>'

with st.expander("📖 Stats Glossary — click to expand"):
    st.markdown(glossary_html, unsafe_allow_html=True)

# Main logic
if search_btn and question:
    with st.spinner("Fetching data from all sources..."):
        player_name = extract_player_name(question)
        team_color = DEFAULT_COLOR
        team_display = ""
        team_abbrev = ""

        if player_name.lower() != "none":
            try:
                nhl_data = get_nhl_data(player_name)
                if nhl_data:
                    team_abbrev = nhl_data.get("currentTeamAbbrev", "")
                    team_color = TEAM_COLORS.get(team_abbrev, DEFAULT_COLOR)
                    team_display = team_color.get("name", "")
            except:
                pass

        if team_display:
            st.markdown(f"""
            <div class="team-banner" style="background: linear-gradient(135deg, {team_color['primary']}22, {team_color['primary']}11); border: 1px solid {team_color['primary']}44;">
                <div class="team-name" style="color: {team_color['primary']};">{player_name}</div>
                <span class="team-badge" style="background: {team_color['primary']}33; color: {team_color['primary']};">{team_display} · {team_abbrev}</span>
            </div>
            """, unsafe_allow_html=True)

        answer = answer_question(question)

        st.markdown(f'<div class="answer-box" style="border-top: 3px solid {team_color["primary"]};">', unsafe_allow_html=True)
        st.markdown(answer, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif search_btn and not question:
    st.warning("Please enter a question first.")