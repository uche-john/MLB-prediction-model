# %% [markdown]
# # First-inning-specific pitcher features (v2 — self-referential, no new API calls)
#
# Your merged games dataframe (right after the schedule+cache merge in
# build_r1_dataset.py, step 6) already has everything needed for this:
# home_sp_id, away_sp_id, r1_home_runs, r1_away_runs — for every start any
# pitcher made. A pitcher's OWN past rows in that same dataset are literally
# his 1st-inning history. So instead of hitting the live API again per
# pitcher (which would also require guessing at gameLog's schema for
# game_pk/home-away, which get_pitcher_game_log() doesn't currently expose),
# we build the rolling stat straight from data you already have on disk.
#
# Who allowed the runs, mechanically:
#   - the HOME starter pitches the TOP of the 1st -> he "allowed" r1_away_runs
#   - the AWAY starter pitches the BOTTOM of the 1st -> he "allowed" r1_home_runs

import pandas as pd
import numpy as np


def build_pitcher_start_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    history: a dataframe with date, home_sp_id, away_sp_id, r1_home_runs,
             r1_away_runs — i.e. games right after the step-6 merge in
             build_r1_dataset.py (before those columns get dropped/renamed),
             OR historical_r1_games_raw.csv loaded back in (it still has
             these columns — step 8 only drops ROWS missing FIP/wRC+, it
             doesn't drop the r1_* columns).

    Returns a long "one row per start" table: pitcher_id, date, r1_runs_allowed.
    """
    required = {"date", "home_sp_id", "away_sp_id", "r1_home_runs", "r1_away_runs"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"history dataframe is missing columns {missing} — make sure you're "
            "passing the merged dataset (post step-6 merge) or "
            "historical_r1_games_raw.csv, not a dataframe that's already had "
            "these columns dropped."
        )

    home_starts = history[["date", "home_sp_id", "r1_away_runs"]].rename(
        columns={"home_sp_id": "pitcher_id", "r1_away_runs": "r1_runs_allowed"}
    )
    away_starts = history[["date", "away_sp_id", "r1_home_runs"]].rename(
        columns={"away_sp_id": "pitcher_id", "r1_home_runs": "r1_runs_allowed"}
    )
    starts = pd.concat([home_starts, away_starts], ignore_index=True)
    starts = starts.dropna(subset=["pitcher_id", "r1_runs_allowed"])
    starts["pitcher_id"] = starts["pitcher_id"].astype(int)
    starts["date"] = pd.to_datetime(starts["date"])
    starts = starts.sort_values(["pitcher_id", "date"]).reset_index(drop=True)
    return starts


def rolling_r1_pitcher_stats(starts: pd.DataFrame, pitcher_id: int, as_of_date,
                              n_recent: int = 8):
    """
    Leak-safe: only uses starts strictly before as_of_date.
    Returns (r1_runs_avg, r1_run_rate, n_starts_used).
    """
    as_of = pd.to_datetime(as_of_date)
    prior = starts[(starts["pitcher_id"] == pitcher_id) & (starts["date"] < as_of)]
    prior = prior.tail(n_recent)

    if prior.empty:
        return np.nan, np.nan, 0

    runs = prior["r1_runs_allowed"]
    return float(runs.mean()), float((runs > 0).mean()), len(prior)


def add_r1_pitcher_features(games: pd.DataFrame, history_source: pd.DataFrame = None,
                             n_recent: int = 8) -> pd.DataFrame:
    """
    games: the dataframe you want to ADD features to. Needs date, home_sp_id,
           away_sp_id.

    history_source: the dataframe used to BUILD the pitcher-start lookup table
           (needs date, home_sp_id, away_sp_id, r1_home_runs, r1_away_runs).

           - When featurizing your TRAINING set: leave this as None. `games`
             itself has r1_home_runs/r1_away_runs at this point (call this
             right after add_features(), still inside build_r1_dataset.py,
             before any columns get dropped) so it can be its own history.

           - When featurizing UPCOMING games for live prediction: you MUST
             pass history_source explicitly, e.g.
             history_source=pd.read_csv("historical_r1_games_raw.csv")
             because upcoming_games has no r1_home_runs/r1_away_runs yet
             (those games haven't been played).

    Adds: home_sp_r1_runs_avg, home_sp_r1_run_rate, home_sp_r1_starts_n,
          away_sp_r1_runs_avg, away_sp_r1_run_rate, away_sp_r1_starts_n
    """
    hist = history_source if history_source is not None else games
    starts = build_pitcher_start_history(hist)

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])

    for side in ["home", "away"]:
        avgs, rates, ns = [], [], []
        for row in games.itertuples():
            pid = getattr(row, f"{side}_sp_id")
            if pd.isna(pid):
                avgs.append(np.nan)
                rates.append(np.nan)
                ns.append(0)
                continue
            avg, rate, n = rolling_r1_pitcher_stats(starts, int(pid), row.date, n_recent)
            avgs.append(avg)
            rates.append(rate)
            ns.append(n)

        games[f"{side}_sp_r1_runs_avg"] = avgs
        games[f"{side}_sp_r1_run_rate"] = rates
        games[f"{side}_sp_r1_starts_n"] = ns

    return games
