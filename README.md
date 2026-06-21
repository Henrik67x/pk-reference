# PK Reference

A proprietary penalty kill analytics engine for the NHL. PK Reference scores every NHL skater's penalty kill performance using two custom metrics, **PKS** and **xPKS**, built on a machine learning model trained on five seasons of validated data.

Built by Henrik, a high school freshman, as an independent research project in hockey analytics.

## What it does

PK Reference answers a question existing public tools don't: *how good is this specific player at killing penalties, independent of his teammates, his goalie, and his deployment?*

Search any NHL player and get:

- **PKS (Penalty Kill Score)** — an outcome-based score reflecting what actually happened on the ice (goals against) while adjusting for context
- **xPKS (Expected Penalty Kill Score)** — a process-based score reflecting shot quality suppressed and individual actions taken, independent of whether the goalie made the save
- **The gap between them** — when PKS is well below xPKS, the player's goaltending/teammates let him down. When PKS is well above xPKS, he was bailed out.

Both scores are scaled so that **10 is league average**, **18–19 is an elite season**, and **negative scores indicate genuinely poor performance** — not just below average.

## How the model works

PK Reference uses a two-stage residual model, not a simple weighted average:

**Stage 1 — Context model (Ridge regression).** Learns what expected goals against (xGA/60) an *average* player would allow given deployment factors outside their control: defensive zone start %, ice time per game, position, goalie save percentage behind them, team defensive structure, and puck luck (PDO).

**Stage 2 — Individual contribution model (XGBoost).** Takes the residual left over after removing context, and explains it using only what the player directly controls: blocks, takeaways, giveaways, hits, shorthanded goals, faceoff win % (for centers), and a composite individual defensive score.

The final score combines both stages and is normalized against the league distribution.

## Data sources

All data is pulled live and validated against [Natural Stat Trick](https://www.naturalstattrick.com), isolated specifically to 4-on-5 penalty kill situations across the 2021-22 through 2025-26 seasons. Player biographical data (age, jersey numbers) is pulled from the official NHL API.

An earlier version of this project attempted to extract individual PK stats directly from raw NHL play-by-play data. That approach produced inconsistent results when cross-checked against NST's validated numbers (see `pk_pbp_collector.py` for the deprecated approach) — a good reminder that "more data" isn't always better than "trusted data." The final model uses NST exclusively for individual and on-ice statistics. I kept it here to show how I worked through this project

## Validation

The scoring model was checked against known PK units (Carolina's penalty kill, consistently one of the league's best, scores its core players — Slavin, Jarvis, Aho, Martinook — near the top of the league) and against personal observation of the 2025-26 Anaheim Ducks.

A separate prediction model was built and tested across two independent held-out seasons (2024-25 and 2025-26): given only information known *before* a season started (age, prior-season performance, ice time trends, deployment), the model predicts next-season individual PK performance **19–21% more accurately than the naive baseline** of assuming a player simply repeats their prior season — a result that held consistently across both test seasons.

The model's largest individual prediction misses cluster on the 2025-26 Columbus Blue Jackets, whose penalty kill collapsed mid-season. Indeed, Columbus made a coaching change (Dean Evason → Rick Bowness) partway through the year and dealt with inconsistent goaltending — context the model has no way to see in advance from individual stats alone. This is treated as a finding, not a flaw: PK performance is meaningfully driven by system and coaching factors that pure player-history models can't capture.

## Features

- Natural language player search with typo tolerance (fuzzy matching, like how Jacob Slavin is actually spelled Jaccob Slavin)
- Disambiguation by jersey number for players sharing a name (e.g., the two Elias Petterssons)
- Season-by-season scoring (2021-22 through 2025-26)
- AI-generated natural language analysis of any player's score (via Gemini)
- Built on a dark, professional UI (using AI)

## Tech stack

- **Python** — pandas, numpy, scikit-learn, XGBoost
- **Streamlit** — app interface
- **Google Gemini API** — natural language explanations
- **Natural Stat Trick** — validated penalty kill data source
- **NHL API** — player biographical and roster data

## Project status

This is an active, ongoing project. Planned next steps include extending the model to goaltenders (a PK-specific goals-saved-above-expected metric), building out a power play equivalent, and deploying publicly with a permanent domain.

## Why this exists

Most public hockey analytics tools (CapFriendly, PuckPedia, Natural Stat Trick itself) were built by passionate individuals who taught themselves the data and the code, not by large organizations. This project follows that same path — built from scratch, sourced from public data, validated against known outcomes, and published openly so the methodology can be checked, criticized, and improved.