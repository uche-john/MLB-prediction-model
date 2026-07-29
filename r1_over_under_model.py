# %% [markdown]
# # 1st Inning Over/Under 0.5 — classification model
#
# Binary target, so we use a classifier and evaluate with metrics that actually
# matter for a betting line: log loss and Brier score (calibration), plus AUC
# (ranking ability) — not raw accuracy, which can look fine while being useless
# if the base rate is already close to 50/50.

# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
from mlb_r1_helpers import get_schedule, add_features
from first_inning_pitcher_features import add_r1_pitcher_features

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load real historical data (run build_r1_dataset.py first)

# %%
DATA_PATH = "historical_r1_games_raw.csv"

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run build_r1_dataset.py first — it pulls "
        "real games and computes home_sp_fip, away_sp_fip, "
        "home_team_wrc_plus, away_team_wrc_plus, park_factor_runs, and the "
        "pitcher-specific rolling 1st-inning stats."
    )

features = [
    "home_sp_fip", "away_sp_fip",
    "home_sp_r1_runs_avg", "away_sp_r1_runs_avg",
    "home_sp_r1_run_rate", "away_sp_r1_run_rate",
    "home_team_wrc_plus", "away_team_wrc_plus", "park_factor_runs",
]

# Sort chronologically — required for time-based (walk-forward) validation.
# Shuffled CV lets the model "see the future" relative to some test rows,
# which is exactly the kind of leakage that makes backtests look better than
# a real forward-looking bet would perform.
data = data.sort_values("date").reset_index(drop=True)

X = data[features]
y = data["over_0_5"]

print(f"{len(data)} games ({data['date'].min()} to {data['date'].max()}). "
      f"Base rate (Over 0.5): {y.mean():.1%}")

# %% [markdown]
# ## 2. Walk-forward (time-based) baseline vs model
#
# TimeSeriesSplit trains on an earlier chunk of games and tests on the chunk
# right after it, then rolls forward — never testing on games that happened
# before the training data. This is the honest way to validate a sports
# model: shuffled CV can leak information across time and make a model look
# better than it would actually perform betting forward, game by game.

# %%
tscv = TimeSeriesSplit(n_splits=5)

baseline_prob = np.full_like(y, y.mean(), dtype=float)
print(f"Baseline log loss:  {log_loss(y, baseline_prob):.4f}")
print(f"Baseline Brier:      {brier_score_loss(y, baseline_prob):.4f}")

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)

cv_logloss = -cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
cv_auc = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
print(f"Walk-forward CV log loss: {cv_logloss.mean():.4f} +/- {cv_logloss.std():.4f}")
print(f"Walk-forward CV AUC:       {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
print("AUC around 0.5 = no better than a coin flip. Compare CV log loss to the baseline above.")
print("Fold-by-fold (later folds = trained on more history):")
for i, (ll, auc) in enumerate(zip(cv_logloss, cv_auc)):
    print(f"  fold {i + 1}: log loss {ll:.4f}, AUC {auc:.4f}")

# %% [markdown]
# ## 3. Fit final model on all-but-the-last chronological chunk + check calibration
#
# The held-out set is now the LAST games by date, not a random 15% — so this
# mirrors how the model would actually be used: trained on the past, tested
# on games it hasn't seen yet in time.

# %%
val_size = max(int(len(data) * 0.15), 50)
X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
print(f"Train: {len(X_train)} games ({data['date'].iloc[0]} to {data['date'].iloc[-val_size - 1]})")
print(f"Val:   {len(X_val)} games ({data['date'].iloc[-val_size]} to {data['date'].iloc[-1]})")

final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
    eval_metric="logloss",
)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs = final_model.predict_proba(X_val)[:, 1]
print(f"Held-out log loss: {log_loss(y_val, val_probs):.4f}")
print(f"Held-out Brier:    {brier_score_loss(y_val, val_probs):.4f}")
print(f"Held-out AUC:      {roc_auc_score(y_val, val_probs):.4f}")

prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10)
print("\nCalibration (predicted vs actual, by bin):")
for pt, pp in zip(prob_true, prob_pred):
    print(f"  predicted ~{pp:.2f} -> actual {pt:.2f}")

importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# %% [markdown]
# ## 4. Today's and tomorrow's real games, with real live stats
#
# Pulls today/tomorrow's schedule + probable pitchers from MLB Stats API, then
# computes the same rolling FIP / offense-proxy / park-factor / pitcher R1
# features used in training — no more NaN placeholders.
#
# The pitcher-specific R1 stats need history_source explicitly, since
# upcoming_games has no r1_home_runs/r1_away_runs yet (those games haven't
# been played) — we pass in the training data (which still has those
# columns) so it has something to build the rolling stat from.

# %%
TODAY = date.today()
TOMORROW = TODAY + timedelta(days=1)

upcoming_games = get_schedule(TODAY.isoformat(), TOMORROW.isoformat(), finished_only=False)

if upcoming_games.empty:
    print("No scheduled games found for today/tomorrow (check date or API status).")
else:
    upcoming_games = add_features(upcoming_games)
    upcoming_games = add_r1_pitcher_features(upcoming_games, history_source=data)

    missing_pitcher = upcoming_games["home_sp_id"].isna() | upcoming_games["away_sp_id"].isna()
    if missing_pitcher.any():
        print(f"Note: {missing_pitcher.sum()} game(s) missing a probable pitcher "
              "(not yet announced) — their FIP will be NaN until MLB posts it.")

    # %%
    if upcoming_games[features].isna().any().any():
        print("Some rows still have missing stats (usually early-season pitchers "
              "with no trailing starts yet, or probable pitcher not yet announced):")
        print(upcoming_games.loc[upcoming_games[features].isna().any(axis=1),
                                  ["date", "home", "away"] + features])

    ready = upcoming_games.dropna(subset=features)
    if ready.empty:
        print("No games have complete stats yet — try again closer to game time.")
    else:
        probs = final_model.predict_proba(ready[features])[:, 1]
        ready = ready.copy()
        ready["prob_over_0_5"] = probs
        print(ready[["date", "home", "away", "prob_over_0_5"]].sort_values(
            "prob_over_0_5", ascending=False
        ).to_string(index=False))