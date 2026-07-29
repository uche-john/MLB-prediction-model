# %% [markdown]
# # Full-game run total — regression model
#
# Predicts total combined (home+away) runs across all 9 innings. No new
# data needed — this is derived from inning_1_runs..inning_9_runs, already
# in historical_highest_inning_raw.csv from build_highest_inning_dataset.py.
#
# This is a REGRESSION problem (predicting a number), not classification —
# unlike the inning-timing questions, "how many total runs" aggregates
# across all 9 innings' worth of scoring opportunities, so it's the same
# shape of question as the R1 over/under model (which showed a real, if
# small, edge) but with much more signal to work with (a full game instead
# of one inning).
#
# CAVEAT: historical_highest_inning_raw.csv only contains games that did
# NOT go to extra innings (excluded per an earlier requirement). So this
# model predicts "total runs, given the game ends in 9," not overall total
# runs including extra-inning games. Worth knowing if you compare this
# against a real sportsbook total line, which doesn't have that carve-out.

# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlb_r1_helpers import get_schedule, add_features

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load data and derive the target (no new API calls needed)

# %%
DATA_PATH = "historical_highest_inning_raw.csv"

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run build_highest_inning_dataset.py first "
        "(this script reuses that same dataset, just with a different target)."
    )

data = data.sort_values("date").reset_index(drop=True)

inning_cols = [f"inning_{i}_runs" for i in range(1, 10)]
data["total_runs"] = data[inning_cols].sum(axis=1)

features = ["home_sp_fip", "away_sp_fip", "home_team_wrc_plus", "away_team_wrc_plus", "park_factor_runs"]

X = data[features]
y = data["total_runs"]

print(f"{len(data)} nine-inning games ({data['date'].min()} to {data['date'].max()})")
print(f"Total runs: mean {y.mean():.2f}, median {y.median():.1f}, "
      f"std {y.std():.2f}, min {y.min()}, max {y.max()}")

# %% [markdown]
# ## 2. Baseline: always predict the training-set mean
#
# The regression equivalent of "always guess the base rate" — if the model
# can't beat this, it isn't adding anything over just knowing the league
# average.

# %%
baseline_pred = np.full_like(y, y.mean(), dtype=float)
print(f"Baseline MAE:  {mean_absolute_error(y, baseline_pred):.3f} runs")
print(f"Baseline RMSE: {np.sqrt(mean_squared_error(y, baseline_pred)):.3f} runs")

# %% [markdown]
# ## 3. Walk-forward (time-based) CV

# %%
tscv = TimeSeriesSplit(n_splits=5)

cv_mae, cv_rmse, cv_r2 = [], [], []
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    fold_model = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE,
    )
    fold_model.fit(X_tr, y_tr)
    preds = fold_model.predict(X_te)

    cv_mae.append(mean_absolute_error(y_te, preds))
    cv_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
    cv_r2.append(r2_score(y_te, preds))

cv_mae, cv_rmse, cv_r2 = np.array(cv_mae), np.array(cv_rmse), np.array(cv_r2)
print(f"Walk-forward CV MAE:  {cv_mae.mean():.3f} +/- {cv_mae.std():.3f} runs "
      f"(baseline: {mean_absolute_error(y, baseline_pred):.3f})")
print(f"Walk-forward CV RMSE: {cv_rmse.mean():.3f} +/- {cv_rmse.std():.3f} runs "
      f"(baseline: {np.sqrt(mean_squared_error(y, baseline_pred)):.3f})")
print(f"Walk-forward CV R^2:  {cv_r2.mean():.4f} +/- {cv_r2.std():.4f} "
      "(0 = no better than always guessing the mean, negative = worse)")
print("Fold-by-fold:")
for i, (mae, rmse, r2) in enumerate(zip(cv_mae, cv_rmse, cv_r2)):
    print(f"  fold {i + 1}: MAE {mae:.3f}, RMSE {rmse:.3f}, R^2 {r2:.4f}")

# %% [markdown]
# ## 4. Fit final model on all-but-the-last chronological chunk

# %%
val_size = max(int(len(data) * 0.15), 50)
X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
print(f"Train: {len(X_train)} games ({data['date'].iloc[0]} to {data['date'].iloc[-val_size - 1]})")
print(f"Val:   {len(X_val)} games ({data['date'].iloc[-val_size]} to {data['date'].iloc[-1]})")

final_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
    eval_metric="rmse",
)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_preds = final_model.predict(X_val)
print(f"Held-out MAE:  {mean_absolute_error(y_val, val_preds):.3f} runs")
print(f"Held-out RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.3f} runs")
print(f"Held-out R^2:  {r2_score(y_val, val_preds):.4f}")

importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# %% [markdown]
# ## 5. Calibration check: predicted vs actual, by predicted-total bucket
#
# Groups held-out predictions into buckets (e.g. "predicted ~7 runs") and
# compares to the ACTUAL average runs in that bucket. A well-calibrated
# regressor should have these roughly match — if predicted-8 games actually
# average 6 runs, the model's numbers aren't trustworthy even if MAE looks
# okay on average.

# %%
val_df = pd.DataFrame({"predicted": val_preds, "actual": y_val.values})
val_df["predicted_bucket"] = val_df["predicted"].round().astype(int)
calib = val_df.groupby("predicted_bucket").agg(
    n_games=("actual", "size"),
    avg_predicted=("predicted", "mean"),
    avg_actual=("actual", "mean"),
).sort_index()
print(calib)

# %% [markdown]
# ## 6. Today's and tomorrow's real games

# %%
TODAY = date.today()
TOMORROW = TODAY + timedelta(days=1)

upcoming_games = get_schedule(TODAY.isoformat(), TOMORROW.isoformat(), finished_only=False)

if upcoming_games.empty:
    print("No scheduled games found for today/tomorrow (check date or API status).")
else:
    upcoming_games = add_features(upcoming_games)

    missing_pitcher = upcoming_games["home_sp_id"].isna() | upcoming_games["away_sp_id"].isna()
    if missing_pitcher.any():
        print(f"Note: {missing_pitcher.sum()} game(s) missing a probable pitcher "
              "(not yet announced) — their FIP will be NaN until MLB posts it.")

    ready = upcoming_games.dropna(subset=features)
    if ready.empty:
        print("No games have complete stats yet — try again closer to game time.")
    else:
        preds = final_model.predict(ready[features])
        ready = ready.copy()
        ready["predicted_total_runs"] = preds
        print(ready[["date", "home", "away", "predicted_total_runs"]]
              .sort_values("predicted_total_runs", ascending=False).to_string(index=False))
