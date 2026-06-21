import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from google import genai
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

CONTEXT_FEATURES = [
    'dz_start_pct', 'toi_pg', 'is_defense', 'is_center',
    'sv_pct', 'ca_per60', 'pdo',
]

INDIVIDUAL_FEATURES = [
    'pk_blocks_per60', 'block_rate', 'pk_takeaways_per60',
    'pk_giveaways_per60', 'pk_ta_ga_ratio', 'pk_hits_per60',
    'pk_penalties_taken_per60', 'sh_goals_per60', 'pk_fo_pct',
    'ihdcf_per60', 'individual_pk_score',
]

PREDICT_FEATURES = [
    'age', 'prior_xga_per60', 'prior_TOI', 'toi_trend',
    'xga_trend', 'team_changed', 'dz_start_pct',
]

TARGET_xPKS = 'xga_per60'
TARGET_PKS = 'ga_per60'


class PKEngine:
    def __init__(self, data_path="pk_final_dataset.csv", age_path="player_birthdates.csv",
                 jersey_path="player_jersey_numbers.csv"):
        print("Initializing PK Engine...")
        self.df = pd.read_csv(data_path, dtype={'season': str})
        self.ages = pd.read_csv(age_path)

        try:
            self.jerseys = pd.read_csv(jersey_path)
        except FileNotFoundError:
            self.jerseys = pd.DataFrame(columns=['player', 'jersey_number'])

        self._prepare_data()
        self._train_scoring_models()
        self._train_prediction_model()
        print("PK Engine ready.")

    def _prepare_data(self):
        df = self.df
        df = df.replace([np.inf, -np.inf], np.nan)

        if df['sv_pct'].median() < 1:
            df['sv_pct'] = df['sv_pct'] * 100

        for col in CONTEXT_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

        for col in INDIVIDUAL_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

        for col in [TARGET_xPKS, TARGET_PKS]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        xga_99 = df[TARGET_xPKS].quantile(0.99)
        ga_99 = df[TARGET_PKS].quantile(0.99)
        df = df[df[TARGET_xPKS] <= xga_99].copy()
        df = df[df[TARGET_PKS] <= ga_99].copy()
        df = df.dropna(subset=[TARGET_xPKS, TARGET_PKS])

        df = df.merge(self.ages, on='player', how='left')
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')

        df = df.merge(self.jerseys, on='player', how='left')

        def season_start(season):
            return pd.Timestamp(year=int(season[:4]), month=10, day=1)

        df['season_start'] = df['season'].apply(season_start)
        df['age'] = (df['season_start'] - df['birth_date']).dt.days / 365.25

        df = df.sort_values(['player', 'season'])
        df['prior_xga_per60'] = df.groupby('player')[TARGET_xPKS].shift(1)
        df['prior_TOI'] = df.groupby('player')['TOI'].shift(1)
        df['prior2_xga_per60'] = df.groupby('player')[TARGET_xPKS].shift(2)
        df['toi_trend'] = df['TOI'] - df['prior_TOI']
        df['xga_trend'] = df['prior_xga_per60'] - df['prior2_xga_per60']
        df['prior_team'] = df.groupby('player')['team_clean'].shift(1)
        df['team_changed'] = (df['team_clean'] != df['prior_team']).astype(int)

        self.df = df

    def _train_scoring_models(self):
        df = self.df
        train = df[df['season'] != '20242025'].copy()

        self.context_model_xpks = Ridge(alpha=1.0)
        self.context_model_pks = Ridge(alpha=1.0)

        X_train_ctx = train[CONTEXT_FEATURES].values
        self.context_model_xpks.fit(X_train_ctx, train[TARGET_xPKS])
        self.context_model_pks.fit(X_train_ctx, train[TARGET_PKS])

        train['expected_xga'] = self.context_model_xpks.predict(X_train_ctx)
        train['expected_ga'] = self.context_model_pks.predict(X_train_ctx)
        train['residual_xga'] = train[TARGET_xPKS] - train['expected_xga']
        train['residual_ga'] = train[TARGET_PKS] - train['expected_ga']

        def make_xgb():
            return XGBRegressor(
                n_estimators=400, max_depth=3, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=8,
                gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, verbosity=0
            )

        self.ind_model_xpks = make_xgb()
        self.ind_model_pks = make_xgb()

        X_train_ind = train[INDIVIDUAL_FEATURES].values
        self.ind_model_xpks.fit(X_train_ind, train['residual_xga'])
        self.ind_model_pks.fit(X_train_ind, train['residual_ga'])

        train['ind_contribution_xga_self'] = self.ind_model_xpks.predict(X_train_ind)
        train['ind_contribution_ga_self'] = self.ind_model_pks.predict(X_train_ind)
        train['raw_xPKS'] = -(train['expected_xga'] + train['ind_contribution_xga_self'])
        train['raw_PKS'] = -(train['expected_ga'] + train['ind_contribution_ga_self'])

        self._reference_xpks = train['raw_xPKS'].values
        self._reference_pks = train['raw_PKS'].values

    def _train_prediction_model(self):
        df = self.df
        train_set = df[df['season'].isin(['20222023', '20232024', '20242025'])].copy()
        train_set = train_set.dropna(subset=PREDICT_FEATURES + ['xga_per60'])

        for col in PREDICT_FEATURES:
            train_set[col] = pd.to_numeric(train_set[col], errors='coerce')

        self._predict_medians = train_set[PREDICT_FEATURES].median()
        train_set[PREDICT_FEATURES] = train_set[PREDICT_FEATURES].fillna(self._predict_medians)

        self.pred_model = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, verbosity=0
        )
        self.pred_model.fit(train_set[PREDICT_FEATURES], train_set['xga_per60'])

    def _normalize_score(self, raw_value, reference_scores, avg=10, elite=18.5, max_val=20):
        median = np.median(reference_scores)
        p95 = np.percentile(reference_scores, 95)
        p5 = np.percentile(reference_scores, 5)

        if raw_value >= median:
            normalized = (raw_value - median) / (p95 - median)
            score = avg + normalized * (elite - avg)
        else:
            normalized = (raw_value - median) / (median - p5)
            score = avg + normalized * avg

        return round(float(np.clip(score, -20, max_val)), 1)

    def find_player_matches(self, search_query):
        """Handles case-insensitivity, typos, and jersey number disambiguation"""
        search_query = search_query.strip()
        all_players = sorted(self.df['player'].unique())

        # Check for jersey number disambiguation (e.g. "Elias Pettersson 40")
        parts = search_query.rsplit(' ', 1)
        jersey_number = None
        name_query = search_query

        if len(parts) == 2 and parts[1].isdigit():
            name_query = parts[0]
            jersey_number = int(parts[1])

        # Exact case-insensitive match
        exact_matches = [p for p in all_players if p.lower() == name_query.lower()]

        # If exact match(es) found and jersey number given, filter by number
        if exact_matches and jersey_number is not None:
            filtered = []
            for p in exact_matches:
                num_row = self.jerseys[self.jerseys['player'] == p]
                if len(num_row) > 0 and num_row.iloc[0]['jersey_number'] == jersey_number:
                    filtered.append(p)
            if filtered:
                return filtered

        if exact_matches:
            return exact_matches

        # Fuzzy match for typos (lowercase comparison for better matching)
        lower_map = {p.lower(): p for p in all_players}
        close_lower = get_close_matches(name_query.lower(), list(lower_map.keys()), n=5, cutoff=0.7)
        close = [lower_map[c] for c in close_lower]

        if close and jersey_number is not None:
            filtered = []
            for p in close:
                num_row = self.jerseys[self.jerseys['player'] == p]
                if len(num_row) > 0 and num_row.iloc[0]['jersey_number'] == jersey_number:
                    filtered.append(p)
            if filtered:
                return filtered

        return close

    def get_player_score(self, player_name, season='20252026', fallback_to_recent=False):
        matches = self.find_player_matches(player_name)

        if not matches:
            return {'error': f"No player found matching '{player_name}'", 'suggestions': []}

        actual_name = matches[0]

        row = self.df[(self.df['player'] == actual_name) & (self.df['season'] == season)]

        used_season = season
        if len(row) == 0 and fallback_to_recent:
            player_rows = self.df[self.df['player'] == actual_name].sort_values('season', ascending=False)
            if len(player_rows) > 0:
                row = player_rows.iloc[[0]]
                used_season = row.iloc[0]['season']
            else:
                return {'error': f"No PK data found for {actual_name} in any season (under TOI threshold)"}
        elif len(row) == 0:
            available_seasons = self.df[self.df['player'] == actual_name]['season'].unique().tolist()
            return {
                'error': f"No PK data for {actual_name} in {season[:4]}-{season[4:]}.",
                'available_seasons': sorted(available_seasons, reverse=True),
                'matched_name': actual_name
            }

        row = row.iloc[0]

        ctx_x = self.context_model_xpks.predict([row[CONTEXT_FEATURES].values])[0]
        ctx_g = self.context_model_pks.predict([row[CONTEXT_FEATURES].values])[0]
        ind_x = self.ind_model_xpks.predict([row[INDIVIDUAL_FEATURES].values])[0]
        ind_g = self.ind_model_pks.predict([row[INDIVIDUAL_FEATURES].values])[0]

        raw_xpks = -(ctx_x + ind_x)
        raw_pks = -(ctx_g + ind_g)

        xPKS = self._normalize_score(raw_xpks, self._reference_xpks)
        PKS = self._normalize_score(raw_pks, self._reference_pks)

        return {
            'player': row['player'],
            'team': row['team_clean'],
            'season': used_season,
            'season_is_fallback': used_season != season,
            'TOI': round(row['TOI'], 1),
            'position': row['position'],
            'xPKS': xPKS,
            'PKS': PKS,
            'gap': round(PKS - xPKS, 1),
            'xga_per60': round(row['xga_per60'], 2),
            'ga_per60': round(row['ga_per60'], 2),
            'pk_blocks_per60': round(row['pk_blocks_per60'], 2),
            'pk_takeaways_per60': round(row['pk_takeaways_per60'], 2),
            'pk_giveaways_per60': round(row['pk_giveaways_per60'], 2),
            'sh_goals_per60': round(row['sh_goals_per60'], 2),
            'other_matches': matches[1:] if len(matches) > 1 else []
        }

    def predict_next_season(self, player_name, current_season='20252026'):
        matches = self.find_player_matches(player_name)
        if not matches:
            return None
        actual_name = matches[0]

        row = self.df[(self.df['player'] == actual_name) & (self.df['season'] == current_season)]
        if len(row) == 0:
            return None
        row = row.iloc[0]

        features = {
            'age': row['age'] + 1,
            'prior_xga_per60': row['xga_per60'],
            'prior_TOI': row['TOI'],
            'toi_trend': row['toi_trend'] if pd.notna(row['toi_trend']) else 0,
            'xga_trend': row['xga_trend'] if pd.notna(row['xga_trend']) else 0,
            'team_changed': 0,
            'dz_start_pct': row['dz_start_pct'],
        }

        X = pd.DataFrame([features])[PREDICT_FEATURES]
        X = X.fillna(self._predict_medians)

        predicted_xga = self.pred_model.predict(X)[0]

        return {
            'player': row['player'],
            'predicted_next_season_xga_per60': round(predicted_xga, 2),
            'current_xga_per60': round(row['xga_per60'], 2),
            'trend': 'improving' if predicted_xga < row['xga_per60'] else 'declining',
        }

    def explain_with_ai(self, player_data):
        prompt = f"""You are a professional NHL penalty kill analyst. Explain this player's PK performance data in 3-4 concise, professional sentences. Do not state obvious context as insight. Be specific with numbers.

Player data:
{player_data}

PKS = outcome-based score (what actually happened), scale -20 to 20, 10 is average, 18-19 is elite.
xPKS = process-based score (expected based on individual actions and shot quality suppression), same scale.
If PKS > xPKS, the player was likely bailed out by goaltending/teammates. If PKS < xPKS, they were let down despite good process.

Write the analysis now."""

        response = client.models.generate_content(
            model="models/gemini-3-flash-preview",
            contents=prompt
        )
        return response.text


if __name__ == "__main__":
    engine = PKEngine()

    print("\n--- Testing typo: 'Jacob Slavin' (should match Jaccob Slavin) ---")
    print(engine.find_player_matches("Jacob Slavin"))

    print("\n--- Testing exact: 'Leo Carlsson' ---")
    score = engine.get_player_score("Leo Carlsson", "20252026")
    print(score)