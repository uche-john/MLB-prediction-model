# %% [markdown]
# # Highest-scoring-inning model — multiclass classification
#
# 10 classes: innings 1-9, plus "tie" (2+ innings tied for the game's max
# combined runs). Extra-inning/shortened games are already excluded from the
# dataset. Same walk-forward validation discipline as the R1 model: log loss
# is the primary metric (calibration-aware), accuracy is compared against
# the "always guess the single most common class" baseline rather than
# treated as meaningful on its own — with 10 classes, raw accuracy is easy
# to misread.

# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from mlb_r1_helpers import get_schedule, add_features

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load real historical data (run build_highest_inning_dataset.py first)

# %%
DATA_PATH = "historical_highest_inning_raw.csv"

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run build_highest_inning_dataset.py first."
    )

features = ["home_sp_fip", "away_sp_fip", "home_team_wrc_plus", "away_team_wrc_plus", "park_factor_runs"]

data = data.sort_values("date").reset_index(drop=True)

le = LabelEncoder()
data["highest_inning"] = data["highest_inning"].astype(str)  # so "tie" and "1".."9" are consistent strings
y = le.fit_transform(data["highest_inning"])
X = data[features]

print(f"{len(data)} nine-inning games ({data['date'].min()} to {data['date'].max()})")
print("Class distribution:")
print(data["highest_inning"].value_counts().sort_index())
print(f"Classes (encoded order): {list(le.classes_)}")

# %% [markdown]
# ## 2. Baselines
#
# Two baselines worth knowing:
#   - "always predict the most common class" -> the accuracy floor
#   - "predict the empirical class distribution for every game" -> the log
#     loss floor (a model that's just as calibrated as the raw base rates,
#     with no per-game information at all)

# %%
most_common_class = pd.Series(y).mode()[0]
baseline_acc = (y == most_common_class).mean()
print(f"Baseline accuracy (always guess '{le.classes_[most_common_class]}'): {baseline_acc:.1%}")

class_probs = pd.Series(y).value_counts(normalize=True).sort_index()
baseline_probs = np.tile(class_probs.reindex(range(len(le.classes_)), fill_value=1e-6).values, (len(y), 1))
baseline_probs = baseline_probs / baseline_probs.sum(axis=1, keepdims=True)
print(f"Baseline log loss (empirical class distribution): {log_loss(y, baseline_probs, labels=range(len(le.classes_))):.4f}")

# %% [markdown]
# ## 3. Walk-forward (time-based) CV

# %%
tscv = TimeSeriesSplit(n_splits=5)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=len(le.classes_),
)

cv_logloss = -cross_val_score(model, X, y, cv=tscv, scoring="neg_log_loss")
cv_acc = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
print(f"Walk-forward CV log loss: {cv_logloss.mean():.4f} +/- {cv_logloss.std():.4f}")
print(f"Walk-forward CV accuracy: {cv_acc.mean():.1%} +/- {cv_acc.std():.1%} "
      f"(baseline: {baseline_acc:.1%})")
print("Fold-by-fold:")
for i, (ll, acc) in enumerate(zip(cv_logloss, cv_acc)):
    print(f"  fold {i + 1}: log loss {ll:.4f}, accuracy {acc:.1%}")

# %% [markdown]
# ## 4. Fit final model on all-but-the-last chronological chunk

# %%
val_size = max(int(len(data) * 0.15), 50)
X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]
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
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=len(le.classes_),
)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs = final_model.predict_proba(X_val)
val_preds = final_model.predict(X_val)
print(f"Held-out log loss: {log_loss(y_val, val_probs, labels=range(len(le.classes_))):.4f}")
print(f"Held-out accuracy: {accuracy_score(y_val, val_preds):.1%} (baseline: {baseline_acc:.1%})")

print("\nConfusion matrix (rows=actual, cols=predicted), class order:", list(le.classes_))
print(confusion_matrix(y_val, val_preds, labels=range(len(le.classes_))))

importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# %% [markdown]
# ## 5. Today's and tomorrow's real games

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
        probs = final_model.predict_proba(ready[features])
        pred_idx = probs.argmax(axis=1)
        ready = ready.copy()
        ready["predicted_highest_inning"] = le.inverse_transform(pred_idx)
        ready["predicted_prob"] = probs.max(axis=1)
        print(ready[["date", "home", "away", "predicted_highest_inning", "predicted_prob"]]
              .sort_values("predicted_prob", ascending=False).to_string(index=False))
