# %% [markdown]
# # First-run-inning model — multiclass classification
#
# 9 classes: which inning (1-9) has the game's first combined run. No new
# data needed — this is derived from inning_1_runs..inning_9_runs, already
# in historical_highest_inning_raw.csv from build_highest_inning_dataset.py.
#
# This is structurally closer to the R1 over/under question (which showed a
# small real edge) than "highest scoring inning" (which showed none) — a
# team/pitcher matchup's quality should directly affect how EARLY a run is
# likely, even if it says little about which single inning ends up with the
# most runs. Worth testing rather than assuming, though.

# %%
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from mlb_inning_helpers import compute_first_run_inning

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
data["first_run_inning"] = compute_first_run_inning(data)

no_run_rows = (data["first_run_inning"] == "no_run").sum()
if no_run_rows:
    print(f"Warning: {no_run_rows} games had 0 runs across all 9 innings "
          "(unexpected — should have gone to extras and been excluded already). "
          "Dropping them.")
    data = data[data["first_run_inning"] != "no_run"].reset_index(drop=True)

features = ["home_sp_fip", "away_sp_fip", "home_team_wrc_plus", "away_team_wrc_plus", "park_factor_runs"]

le = LabelEncoder()
y = le.fit_transform(data["first_run_inning"])
X = data[features]

print(f"{len(data)} nine-inning games ({data['date'].min()} to {data['date'].max()})")
print("Class distribution:")
print(data["first_run_inning"].value_counts().sort_index())
print(f"Classes (encoded order): {list(le.classes_)}")

# %% [markdown]
# ## 2. Baselines

# %%
most_common_class = pd.Series(y).mode()[0]
baseline_acc = (y == most_common_class).mean()
print(f"Baseline accuracy (always guess inning '{le.classes_[most_common_class]}'): {baseline_acc:.1%}")

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

# NOTE: cross_val_score's built-in "neg_log_loss" scorer infers which
# classes are present from each fold's y_true, rather than being told there
# are 9 possible classes total. Since some classes here are rare (e.g.
# inning 9 has only a handful of games total), an early fold's test set can
# end up with zero examples of a class — the scorer then errors because the
# model's predict_proba has 9 columns but the fold "only knows about" 8
# classes. Looping manually and passing labels=range(len(le.classes_))
# explicitly to log_loss avoids that.
all_labels = list(range(len(le.classes_)))
cv_logloss, cv_acc = [], []
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    fold_model = XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        objective="multi:softprob", num_class=len(le.classes_),
    )
    fold_model.fit(X_tr, y_tr)
    probs = fold_model.predict_proba(X_te)
    preds = fold_model.predict(X_te)

    cv_logloss.append(log_loss(y_te, probs, labels=all_labels))
    cv_acc.append(accuracy_score(y_te, preds))

cv_logloss = np.array(cv_logloss)
cv_acc = np.array(cv_acc)
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
# ## Notes
#
# - Unlike "highest scoring inning," there's no "tie" class here — the first
#   run happens in exactly one inning, always (since scoreless-through-9
#   games are already excluded from this dataset).
# - If the confusion matrix again shows the model collapsing to always
#   predicting the single most common class (almost certainly inning 1,
#   since that mirrors the ~49% over_0_5 base rate from the R1 model), that
#   means the same conclusion applies here: pre-game stats alone don't
#   distinguish WHEN the first run comes beyond what's already captured by
#   whether it happens in the 1st inning specifically.
# - This script doesn't include a live "predict today's games" section like
#   the other two — add one the same way (get_schedule + add_features on
#   upcoming_games, then final_model.predict_proba) once you've checked
#   whether the walk-forward numbers actually justify it.
