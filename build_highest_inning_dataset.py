# %% [markdown]
# # Build a real historical "highest-scoring inning" dataset
#
# Same pipeline shape as build_r1_dataset.py: pull schedules, fetch results
# per game (cached, resumable), merge, add rolling FIP / offense-proxy /
# park-factor features, save. The difference is get_game_innings() pulls the
# FULL 9-inning linescore instead of just inning 1, and the target is which
# inning (combined home+away runs) had the most runs — "tie" if 2+ innings
# are tied for the max. Games that went to extras or were shortened are
# excluded (only exactly-9-inning games are kept).
#
# Run this locally (needs real internet access to statsapi.mlb.com).
# pip install pandas numpy requests

# %%
import os
import pandas as pd
from mlb_r1_helpers import get_schedule, add_features
from mlb_inning_helpers import get_game_innings

CACHE_PATH = "highest_inning_games_cache.csv"

# %% [markdown]
# ## 1. Season windows to pull (same as the R1 pipeline)

# %%
SEASON_WINDOWS = [
    ("2023-04-01", "2023-10-01"),
    ("2024-04-01", "2024-10-01"),
    ("2025-04-01", "2025-10-01"),
    ("2026-04-01", "2026-07-24"),  # current season, up through last completed slate
]
FULL_SEASON_WINDOWS = {"2026-04-01"}
TARGET_SAMPLE_SIZE = 3000
RANDOM_STATE = 42

# %% [markdown]
# ## 2. Pull schedules

# %%
schedule_frames = []
for start, end in SEASON_WINDOWS:
    s = get_schedule(start, end, finished_only=True)
    print(f"{start} to {end}: {len(s)} completed games")
    s["_window_start"] = start
    schedule_frames.append(s)

schedule = pd.concat(schedule_frames, ignore_index=True)
print(f"Total completed games across all windows (pre-subsample): {len(schedule)}")

# %% [markdown]
# ## 3. Stratified subsample by month (same logic as the R1 build script)

# %%
full_mask = schedule["_window_start"].isin(FULL_SEASON_WINDOWS)
full_part = schedule[full_mask].copy()
sample_part = schedule[~full_mask].copy()

if len(sample_part) > 0:
    sample_part["month"] = pd.to_datetime(sample_part["date"]).dt.to_period("M")
    frac = min(1.0, TARGET_SAMPLE_SIZE / len(sample_part))
    sample_part = (
        sample_part.groupby("month", group_keys=False)
        .sample(frac=frac, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    sample_part = sample_part.drop(columns="month", errors="ignore")

schedule = pd.concat([full_part, sample_part], ignore_index=True).drop(columns="_window_start")
print(f"Schedule after subsampling: {len(schedule)} games "
      f"({len(full_part)} kept in full, {len(sample_part)} sampled)")

before_dedup_sched = len(schedule)
schedule = schedule.drop_duplicates(subset="game_pk").reset_index(drop=True)
print(f"Deduplicated schedule: {before_dedup_sched} -> {len(schedule)} rows "
      f"({before_dedup_sched - len(schedule)} duplicates removed)")

# %% [markdown]
# ## 4. Fetch full linescore + starters per game, resuming from cache

# %%
if os.path.exists(CACHE_PATH):
    cache_df = pd.read_csv(CACHE_PATH)
    done_pks = set(cache_df["game_pk"])
    print(f"Resuming: {len(done_pks)} games already cached from a previous run")
else:
    cache_df = pd.DataFrame()
    done_pks = set()

to_fetch = [gpk for gpk in schedule["game_pk"] if gpk not in done_pks]
print(f"{len(to_fetch)} games left to fetch")

buffer = []
for i, gpk in enumerate(to_fetch):
    result = get_game_innings(int(gpk))
    if result:
        buffer.append(result)

    if (i + 1) % 50 == 0 or (i + 1) == len(to_fetch):
        if buffer:
            chunk = pd.DataFrame(buffer)
            write_header = not os.path.exists(CACHE_PATH)
            chunk.to_csv(CACHE_PATH, mode="a", header=write_header, index=False)
            buffer = []
        print(f"{i + 1}/{len(to_fetch)} fetched and flushed to {CACHE_PATH}")

innings_df = pd.read_csv(CACHE_PATH)
print(f"Cache now holds {len(innings_df)} games total (before dedup)")

before_dedup_cache = len(innings_df)
innings_df = innings_df.drop_duplicates(subset="game_pk", keep="last").reset_index(drop=True)
print(f"Deduplicated cache: {before_dedup_cache} -> {len(innings_df)} rows "
      f"({before_dedup_cache - len(innings_df)} duplicate game_pk rows removed)")
innings_df.to_csv(CACHE_PATH, index=False)

# %% [markdown]
# ## 5. Drop excluded games (extras / shortened) and merge with schedule

# %%
before_excl = len(innings_df)
innings_df = innings_df[innings_df["excluded"] == False].copy()  # noqa: E712
print(f"Dropped {before_excl - len(innings_df)} games that went to extras or "
      f"were shortened (not exactly 9 innings) ({before_excl} -> {len(innings_df)})")

games = schedule.merge(innings_df, on="game_pk", how="inner", suffixes=("_sched", "_line"))

if "date_sched" in games.columns and "date_line" in games.columns:
    games["date"] = games["date_sched"].fillna(games["date_line"])
    games = games.drop(columns=["date_sched", "date_line"])
elif "date_sched" in games.columns:
    games = games.rename(columns={"date_sched": "date"})

games["home_sp_id"] = games["home_sp_id_actual"].fillna(games["home_sp_id"])
games["away_sp_id"] = games["away_sp_id_actual"].fillna(games["away_sp_id"])
games = games.drop(columns=["home_sp_id_actual", "away_sp_id_actual"])

assert len(games) <= min(len(schedule), len(innings_df)), (
    f"Merged row count ({len(games)}) exceeds both input frames — duplicate "
    "game_pk values are still present in one of them."
)

print(f"Merged dataset: {len(games)} nine-inning games.")
print("Highest-inning class distribution:")
print(games["highest_inning"].value_counts().sort_index())

# %% [markdown]
# ## 6. Add rolling FIP / team-offense proxy / park factor (reused as-is)

# %%
games = add_features(games)

print(f"home_sp_fip missing: {games['home_sp_fip'].isna().mean():.1%}")
print(f"away_sp_fip missing: {games['away_sp_fip'].isna().mean():.1%}")
print(f"home_team_wrc_plus missing: {games['home_team_wrc_plus'].isna().mean():.1%}")
print(f"away_team_wrc_plus missing: {games['away_team_wrc_plus'].isna().mean():.1%}")

# %% [markdown]
# ## 7. Drop rows missing required features and save

# %%
required = ["home_sp_fip", "away_sp_fip", "home_team_wrc_plus", "away_team_wrc_plus", "park_factor_runs"]
before = len(games)
games = games.dropna(subset=required)
print(f"Dropped {before - len(games)} rows with missing rolling stats "
      f"({before} -> {len(games)})")

games = games.sort_values("date")
games.to_csv("historical_highest_inning_raw.csv", index=False)
print(f"Saved {len(games)} games to historical_highest_inning_raw.csv")

# %% [markdown]
# ## Notes / known limitations
#
# - Same underlying FIP/offense-proxy/park-factor caveats as the R1 pipeline
#   (see build_r1_dataset.py's notes) — these are pre-game stats, they don't
#   carry any info about WHEN in the game runs will cluster (e.g. bullpen
#   quality, 3rd-time-through-the-order effects). This is likely to be a
#   hard target for the same reason R1 over/under was: worth sanity-checking
#   with a simple baseline (predict the single most common inning) before
#   reading too much into any model's apparent accuracy.
# - "tie" is its own class (2+ innings tied for the game's max combined runs)
#   rather than being broken arbitrarily — this will likely be one of the
#   MORE common classes in low-scoring games (e.g. two innings each with the
#   game's only run), so don't be surprised if it's not a rare class.
# - Extra-inning and shortened games are excluded entirely, per your setup.
