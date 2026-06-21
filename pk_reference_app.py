import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

st.set_page_config(page_title="PK Reference", page_icon="pk_logo.png", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def get_theme_css(dark):
    if dark:
        return """
        .stApp { background: #0a0e1a; color: #e2e8f0; }
        .pk-card { background: #111827; border: 1px solid #1e2a3a; }
        .pk-label { color: #5a6a80; }
        .pk-ai-box { background: #111827; border-left: 3px solid #4FC3F7; }
        .pk-subtitle { color: #8a9ab0; }
        """
    else:
        return """
        .stApp { background: #f7f9fc; color: #1a1f2e; }
        .pk-card { background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
        .pk-label { color: #6b7588; }
        .pk-ai-box { background: #ffffff; border-left: 3px solid #1a5fd6; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
        .pk-subtitle { color: #4a5568; }
        """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; }}
.stApp {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2.5rem 3rem !important; max-width: 1280px !important; }}

h1, h2, h3, h4 {{ font-family: 'Oswald', sans-serif; letter-spacing: 0.3px; }}

.pk-header {{
    display: flex; align-items: center; gap: 18px;
    padding-bottom: 20px; border-bottom: 2px solid #4FC3F7;
    margin-bottom: 28px;
}}
.pk-title {{ font-family: 'Oswald', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0; }}
.pk-subtitle {{ font-size: 0.85rem; margin: 2px 0 0 0; }}

.pk-card {{
    border-radius: 12px; padding: 18px; text-align: center;
}}
.pk-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 6px; }}
.pk-value {{ font-family: 'Oswald', sans-serif; font-size: 2.1rem; font-weight: 700; }}

.pk-ai-box {{
    border-radius: 10px; padding: 18px 22px; margin-top: 18px;
    font-size: 0.95rem; line-height: 1.65;
}}

{get_theme_css(st.session_state.dark_mode)}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return PKEngine()

header_col1, header_col2, header_col3 = st.columns([1, 5, 1])
with header_col1:
    st.image("pk_logo.png", width=90)
with header_col2:
    st.markdown("""
    <div style="padding-top: 6px;">
        <p class="pk-title">PK REFERENCE</p>
        <p class="pk-subtitle">Penalty Kill Analytics — PKS (outcome) and xPKS (process) scoring model</p>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    if st.button("🌓 Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

with st.spinner("Loading model..."):
    engine = load_engine()

st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    player_search = st.text_input("Search a player", placeholder="e.g. Leo Carlsson, or try a typo")
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

        if score.get('sample_warning'):
            st.warning(score['sample_warning'])

        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""<div class="pk-card"><div class="pk-label">xPKS (process)</div>
                        <div class="pk-value" style="color:#4FC3F7">{score['xPKS']}</div></div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="pk-card"><div class="pk-label">PKS (outcome)</div>
                        <div class="pk-value" style="color:{team_color}">{score['PKS']}</div></div>""", unsafe_allow_html=True)
        with cols[2]:
            gap_color = "#2e7d32" if score['gap'] > 0 else "#c62828" if score['gap'] < 0 else "#888"
            st.markdown(f"""<div class="pk-card"><div class="pk-label">Gap (PKS − xPKS)</div>
                        <div class="pk-value" style="color:{gap_color}">{score['gap']:+.1f}</div></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div class="pk-card"><div class="pk-label">xGA / 60</div>
                        <div class="pk-value" style="font-size:1.6rem">{score['xga_per60']}</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Score vs. League Average")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["xPKS", "PKS"],
            y=[score['xPKS'], score['PKS']],
            marker_color=["#4FC3F7", team_color],
            text=[score['xPKS'], score['PKS']],
            textposition='outside',
            width=0.45
        ))
        fig.add_hline(y=10, line_dash="dash", line_color="gray", annotation_text="League Average (10)")
        fig.update_layout(
            height=280,
            showlegend=False,
            yaxis=dict(range=[-20, 20], title="Score"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#e2e8f0" if st.session_state.dark_mode else "#1a1f2e"),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Individual PK Actions (per 60 minutes)")
        action_cols = st.columns(4)
        with action_cols[0]:
            st.metric("Blocks", score['pk_blocks_per60'])
        with action_cols[1]:
            st.metric("Takeaways", score['pk_takeaways_per60'])
        with action_cols[2]:
            st.metric("Giveaways", score['pk_giveaways_per60'])
        with action_cols[3]:
            st.metric("SH Goals", score['sh_goals_per60'])

        st.markdown("#### Performance Detail")
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown(f"""
            <div class="pk-card" style="text-align:left; padding:16px 20px;">
                <div class="pk-label" style="margin-bottom:10px;">Expected (Process)</div>
                <div style="font-size:0.95rem; line-height:1.8;">
                    Expected Goals Against / 60: <b>{score['xga_per60']}</b><br>
                    Blocks / 60: <b>{score['pk_blocks_per60']}</b><br>
                    Takeaways / 60: <b>{score['pk_takeaways_per60']}</b><br>
                    Giveaways / 60: <b>{score['pk_giveaways_per60']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with detail_cols[1]:
            st.markdown(f"""
            <div class="pk-card" style="text-align:left; padding:16px 20px;">
                <div class="pk-label" style="margin-bottom:10px;">Actual (Outcome)</div>
                <div style="font-size:0.95rem; line-height:1.8;">
                    Goals Against / 60: <b>{score['ga_per60']}</b><br>
                    PK Ice Time: <b>{score['TOI']} min</b><br>
                    Shorthanded Goals / 60: <b>{score['sh_goals_per60']}</b><br>
                    Position: <b>{score['position']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing..."):
                explanation = engine.explain_with_ai(score)
                st.markdown(f'<div class="pk-ai-box">{explanation}</div>', unsafe_allow_html=True)
else:
    st.info("Search for any NHL player above. Typos and last-name-only searches work — fuzzy matching will find them.")