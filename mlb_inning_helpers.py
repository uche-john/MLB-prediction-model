"""
Helpers for the "highest-scoring inning" multiclass model.

Reuses schedule pulling, rolling FIP, rolling team-offense proxy, and park
factor from mlb_r1_helpers.py (those are generic enough to work here
unchanged) and adds:
  - full 9-inning linescore fetch (not just inning 1, like get_r1_runs)
  - actual starter IDs (from boxscore — more reliable than probable pitcher)
  - highest_inning target computation: combined (home+away) runs per
    inning, tie-aware

Games that went to extra innings or were shortened (rain, etc.) are
EXCLUDED per your requirements — only games with exactly 9 innings on the
linescore are used.
"""

import pandas as pd
import numpy as np
from mlb_r1_helpers import SESSION, REQUEST_TIMEOUT, get_schedule, add_features  # noqa: F401 (re-exported for convenience)


def _get_starting_pitcher_id(team_box: dict):
    pitchers = team_box.get("pitchers", [])
    return pitchers[0] if pitchers else None


def get_game_innings(game_pk: int) -> dict | None:
    """
    Fetches the full linescore + starters for one game.

    Returns None if the linescore itself can't be fetched at all.
    Returns {"game_pk": ..., "excluded": True, "n_innings": N} if the game
    isn't exactly 9 innings (extras or a shortened/rain-called game) — the
    build script filters these out before merging features.
    Otherwise returns a full row with per-inning runs and the target.
    """
    try:
        line_resp = SESSION.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore",
            timeout=REQUEST_TIMEOUT,
        )
        line_resp.raise_for_status()
        line_data = line_resp.json()
    except Exception as e:
        print(f"  skipping game {game_pk} (linescore fetch failed: {e})")
        return None

    innings = line_data.get("innings", [])
    if len(innings) != 9:
        return {"game_pk": game_pk, "excluded": True, "n_innings": len(innings)}

    combined = []
    for inn in innings:
        h = inn.get("home", {}).get("runs", 0) or 0
        a = inn.get("away", {}).get("runs", 0) or 0
        combined.append(h + a)

    max_runs = max(combined)
    top = [i + 1 for i, r in enumerate(combined) if r == max_runs]
    highest_inning = "tie" if len(top) > 1 else str(top[0])

    home_sp_id = away_sp_id = None
    try:
        box_resp = SESSION.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
            timeout=REQUEST_TIMEOUT,
        )
        box_resp.raise_for_status()
        box = box_resp.json().get("teams", {})
        home_sp_id = _get_starting_pitcher_id(box.get("home", {}))
        away_sp_id = _get_starting_pitcher_id(box.get("away", {}))
    except Exception as e:
        print(f"  game {game_pk}: boxscore fetch failed ({e}), starters unknown")

    row = {
        "game_pk": game_pk,
        "excluded": False,
        "n_innings": 9,
        "max_runs": max_runs,
        "highest_inning": highest_inning,
        "home_sp_id_actual": home_sp_id,
        "away_sp_id_actual": away_sp_id,
    }
    for i, r in enumerate(combined, start=1):
        row[f"inning_{i}_runs"] = r
    return row


def compute_first_run_inning(df: pd.DataFrame) -> pd.Series:
    """
    Given a dataframe with inning_1_runs..inning_9_runs (as saved in
    historical_highest_inning_raw.csv), returns the inning number (1-9) of
    the first combined (home+away) run in each game, as a string column
    (consistent with how highest_inning is stored as strings).

    Every row should have a valid answer here, since games with 0 total
    runs through 9 innings go to extras and are already excluded from this
    dataset (build_highest_inning_dataset.py only keeps exactly-9-inning
    games) — so a scoreless 9 innings can't appear in this data.
    """
    inning_cols = [f"inning_{i}_runs" for i in range(1, 10)]
    missing = [c for c in inning_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} — make sure you're passing "
            "historical_highest_inning_raw.csv (or the pre-save games "
            "dataframe from build_highest_inning_dataset.py), which has "
            "inning_1_runs..inning_9_runs."
        )

    scored = (df[inning_cols] > 0).values  # bool array, shape (n_games, 9)
    first_idx = scored.argmax(axis=1)  # index of first True; 0 if no True at all
    has_any_run = scored.any(axis=1)

    result = pd.Series(first_idx + 1, index=df.index).astype(str)
    result[~has_any_run] = "no_run"  # defensive guard; shouldn't occur in practice
    return result
