import streamlit as st
import pandas as pd
import numpy as np
from pk_engine import PKEngine

TEAM_COLORS = {
    "ANA": "#F47A38", "BOS": "#FFB81C", "BUF": "#003087", "CAR": "#CC0000",
    "CBJ": "#002654", "CGY": "#C8102E", "CHI": "#CF0A2C", "COL": "#6F263D",
    "DAL": "#006847", "DET": "#CE1126", "EDM": "#FF4C00", "FLA": "#041E42",
    "L.A": "#111111", "LAK": "#111111", "MIN": "#154734", "MTL": "#AF1E2D",
    "N.J": "#CE1126", "NJD": "#CE1126", "NSH": "#FFB81C", "N.Y": "#0038A8",
    "NYI": "#003087", "NYR": "#0038A8", "OTT": "#C8102E", "PHI": "#F74902",
    "PIT": "#FCB514", "S.J": "#006D75", "SJS": "#006D75", "SEA": "#001628",
    "STL": "#002F87", "T.B": "#002868", "TBL": "#002868", "TOR": "#003E7E",
    "UTA": "#6CACE4", "VAN": "#00205B", "VGK": "#B4975A", "WPG": "#041E42",
    "WSH": "#041E42",
}

st.set_page_config(page_title="PK Reference", page_icon="🏒", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp { background: #0a0e1a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 3rem !important; max-width: 1300px !important; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; }
.metric-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 16px; text-align: center;
}
.metric-label { font-size: 11px; color: #5a6a80; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 800; }
.ai-box {
    background: #111827; border-left: 3px solid #4FC3F7; border-radius: 8px;
    padding: 16px 20px; margin-top: 16px; font-size: 0.95rem; line-height: 1.6;
}
.stTextInput input { background: #0f1628 !important; border: 1px solid #2a3a55 !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return PKEngine()

st.title("🏒 PK Reference")
st.caption("Penalty Kill analytics powered by a proprietary scoring model — PKS (outcome) and xPKS (process)")

with st.spinner("Loading model..."):
    engine = load_engine()

st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    player_search = st.text_input("Search a player", placeholder="e.g. Leo Carlsson")
with col2:
    season_select = st.selectbox("Season", ["20252026", "20242025", "20232024", "20222023", "20212022"])

if player_search:
    score = engine.get_player_score(player_search, season_select)
    
    if 'error' in score:
        st.warning(score['error'])
        if score.get('matched_name'):
            st.write(f"Found **{score['matched_name']}**, but no PK data for that season.")
            seasons = score.get('available_seasons', [])
            if seasons:
                st.write("Available seasons:", ", ".join([f"{s[:4]}-{s[4:]}" for s in seasons]))
    else:
        if score['player'].lower() != player_search.lower():
            st.caption(f"Showing closest match for '{player_search}': **{score['player']}**")
        
        if score.get('other_matches'):
            st.caption(f"Other players found: {', '.join(score['other_matches'])}. Add jersey number to disambiguate, e.g. 'Elias Pettersson 40'")
        
        team_color = TEAM_COLORS.get(score['team'].split(',')[0].strip(), "#4FC3F7")
        
        st.markdown(f"### {score['player']} <span style='color:{team_color}; font-size:1rem;'>· {score['team']} · {score['position']}</span>", unsafe_allow_html=True)
        st.caption(f"{score['TOI']} minutes of PK time in {score['season'][:4]}-{score['season'][4:]}")
        
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">xPKS (process)</div>
                        <div class="metric-value" style="color:#4FC3F7">{score['xPKS']}</div></div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">PKS (outcome)</div>
                        <div class="metric-value" style="color:#81D4FA">{score['PKS']}</div></div>""", unsafe_allow_html=True)
        with cols[2]:
            gap_color = "#4caf50" if score['gap'] > 0 else "#e57373" if score['gap'] < 0 else "#999"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Gap (PKS-xPKS)</div>
                        <div class="metric-value" style="color:{gap_color}">{score['gap']:+.1f}</div></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">xGA / 60</div>
                        <div class="metric-value" style="font-size:1.5rem">{score['xga_per60']}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("#### Individual PK actions (per 60)")
        action_cols = st.columns(4)
        with action_cols[0]:
            st.metric("Blocks", score['pk_blocks_per60'])
        with action_cols[1]:
            st.metric("Takeaways", score['pk_takeaways_per60'])
        with action_cols[2]:
            st.metric("Giveaways", score['pk_giveaways_per60'])
        with action_cols[3]:
            st.metric("SH Goals", score['sh_goals_per60'])
        
        if st.button("Get AI analysis"):
            with st.spinner("Generating analysis..."):
                explanation = engine.explain_with_ai(score)
                st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)
else:
    st.info("Search for any NHL player above to see their PKS and xPKS scores. Try a typo or last name only — fuzzy search will find them.")