# %% [markdown]

# # Build a real historical R1 (1st inning) dataset — multi-season (subsampled)
#
# Run this locally (needs real internet access to statsapi.mlb.com).
# Produces historical_r1_games_raw.csv with real games across multiple
# regular seasons, real rolling (pre-game, non-leaky) pitcher FIP, a rolling
# team-offense proxy, park factor, real pitcher-specific rolling 1st-inning
# stats, and the real over_0_5 target from actual 1st-inning linescores.
#
# This version stratified-subsamples the schedule BEFORE the expensive
# per-game fetch loop, so you get broad season/month coverage without
# pulling every single game. The current season is kept in full since
# it's the most representative data for a live model.
#
# It caches progress to r1_games_cache.csv as it goes — if it crashes or
# you Ctrl+C it, just rerun the script; already-fetched games are skipped
# automatically. Because the cache is appended to across resumed runs, it
# can accumulate duplicate game_pk rows over time — this version dedupes
# both the schedule and the cache before merging to guard against that.
#
# pip install pandas numpy requests

# %%
import os
import pandas as pd
from mlb_r1_helpers import get_schedule, get_r1_runs, add_features
from first_inning_pitcher_features import add_r1_pitcher_features

CACHE_PATH = "r1_games_cache.csv"

# %% [markdown]
# ## 1. Season windows to pull
#
# Regular-season months only (skips winter, when there's nothing to fetch
# anyway). Add/remove seasons here — more seasons = more training data, but
# also a longer run.

# %%
SEASON_WINDOWS = [
    ("2023-04-01", "2023-10-01"),
    ("2024-04-01", "2024-10-01"),
    ("2025-04-01", "2025-10-01"),
    ("2026-04-01", "2026-07-24"),  # current season, up through last completed slate
]

# Season windows that should always be kept IN FULL (not subsampled).
# The current season is your freshest, most representative data —
# don't thin it out.
FULL_SEASON_WINDOWS = {"2026-04-01"}

# %% [markdown]
# ## 2. Sample size controls
#
# TARGET_SAMPLE_SIZE caps how many games get pulled from the older
# (non-full) season windows, stratified by month so coverage stays even
# across the season rather than clumping in April/May.

# %%
TARGET_SAMPLE_SIZE = 3000
RANDOM_STATE = 42

# %% [markdown]
# ## 3. Pull schedules for all season windows (cheap — one call per window)

# %%
schedule_frames = []
for start, end in SEASON_WINDOWS:
    s = get_schedule(start, end, finished_only=True)
    print(f"{start} to {end}: {len(s)} completed games")
    s["_window_start"] = start  # tag so we know which window each row came from
    schedule_frames.append(s)

schedule = pd.concat(schedule_frames, ignore_index=True)
print(f"Total completed games across all windows (pre-subsample): {len(schedule)}")

# %% [markdown]
# ## 4. Stratified subsample by month, before the expensive fetch loop
#
# Full-season windows (e.g. current season) are kept entirely. Everything
# else is proportionally sampled per month so no month gets skewed out of
# the training data.

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

# %% [markdown]
# ## 4b. Dedupe schedule by game_pk (safety check)
#
# SEASON_WINDOWS shouldn't overlap, but this is a cheap guard against any
# accidental duplicate game_pk rows before the expensive fetch loop.

# %%
before_dedup_sched = len(schedule)
schedule = schedule.drop_duplicates(subset="game_pk").reset_index(drop=True)
print(f"Deduplicated schedule: {before_dedup_sched} -> {len(schedule)} rows "
      f"({before_dedup_sched - len(schedule)} duplicates removed)")

# %% [markdown]
# ## 5. Fetch R1 results per game, resuming from cache if it exists
#
# This is the slow, API-heavy part. Progress is flushed to disk every 50
# games so a crash partway through doesn't cost you a full re-pull.

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
    result = get_r1_runs(int(gpk))
    if result:
        buffer.append(result)

    if (i + 1) % 50 == 0 or (i + 1) == len(to_fetch):
        if buffer:
            chunk = pd.DataFrame(buffer)
            write_header = not os.path.exists(CACHE_PATH)
            chunk.to_csv(CACHE_PATH, mode="a", header=write_header, index=False)
            buffer = []
        print(f"{i + 1}/{len(to_fetch)} fetched and flushed to {CACHE_PATH}")

r1_df = pd.read_csv(CACHE_PATH)
print(f"Cache now holds {len(r1_df)} games total (before dedup)")

# %% [markdown]
# ## 5b. Dedupe the cache by game_pk
#
# The cache is appended to (mode="a") every time this script is run or
# resumed, which can accumulate duplicate game_pk rows across multiple
# sessions. An inner merge with duplicates on either side will silently
# multiply rows — keep="last" keeps the most recently fetched copy of
# each game.

# %%
before_dedup_cache = len(r1_df)
r1_df = r1_df.drop_duplicates(subset="game_pk", keep="last").reset_index(drop=True)
print(f"Deduplicated cache: {before_dedup_cache} -> {len(r1_df)} rows "
      f"({before_dedup_cache - len(r1_df)} duplicate game_pk rows removed)")

# Optional: rewrite the deduped cache back to disk so it stays clean on
# future resumed runs instead of re-accumulating duplicates.
r1_df.to_csv(CACHE_PATH, index=False)

# %% [markdown]
# ## 6. Merge schedule + cached R1 results
#
# Explicit suffixes on any overlapping non-key columns (e.g. "date" showing
# up in both frames) prevent pandas from silently renaming to date_x/date_y
# and dropping the plain "date" column the rest of the script relies on.

# %%
games = schedule.merge(r1_df, on="game_pk", how="inner", suffixes=("_sched", "_r1"))

# Reconcile date column if it existed in both frames
if "date_sched" in games.columns and "date_r1" in games.columns:
    games["date"] = games["date_sched"].fillna(games["date_r1"])
    games = games.drop(columns=["date_sched", "date_r1"])
elif "date_sched" in games.columns:
    games = games.rename(columns={"date_sched": "date"})
elif "date_r1" in games.columns:
    games = games.rename(columns={"date_r1": "date"})

games["home_sp_id"] = games["home_sp_id_actual"].fillna(games["home_sp_id"])
games["away_sp_id"] = games["away_sp_id_actual"].fillna(games["away_sp_id"])
games = games.drop(columns=["home_sp_id_actual", "away_sp_id_actual"])

# Safety check: merged result should never exceed the smaller input frame
# on an inner join over a unique key. If it does, duplicates slipped through.
assert len(games) <= min(len(schedule), len(r1_df)), (
    f"Merged row count ({len(games)}) exceeds both input frames "
    f"(schedule={len(schedule)}, r1_df={len(r1_df)}) — duplicate game_pk "
    "values are still present in one of them."
)

print(f"Merged dataset: {len(games)} games with R1 totals. "
      f"Base rate over_0_5: {games['over_0_5'].mean():.1%}")

# %% [markdown]
# ## 7. Add rolling FIP, rolling team-offense proxy, and park factor
#
# All computed strictly from data BEFORE each game's date, so nothing leaks.
# This part re-fetches pitcher/team game logs, which are cached in-memory
# per (id, season) within this run.

# %%
games = add_features(games)

print(f"home_sp_fip missing: {games['home_sp_fip'].isna().mean():.1%}")
print(f"away_sp_fip missing: {games['away_sp_fip'].isna().mean():.1%}")
print(f"home_team_wrc_plus missing: {games['home_team_wrc_plus'].isna().mean():.1%}")
print(f"away_team_wrc_plus missing: {games['away_team_wrc_plus'].isna().mean():.1%}")

# %% [markdown]
# ## 7b. Add pitcher-specific rolling 1st-inning stats
#
# This is self-referential: games (at this point) still has r1_home_runs /
# r1_away_runs from the step-6 merge, so a pitcher's own past rows in this
# same dataframe ARE his 1st-inning history — no extra API calls needed.
# Must run BEFORE step 8, which only drops rows (not the r1_* columns), so
# this still works even after the dropna below.
#
# Mechanically: the HOME starter pitches the top of the 1st (allows
# r1_away_runs), the AWAY starter pitches the bottom of the 1st (allows
# r1_home_runs). See first_inning_pitcher_features.py for details.

# %%
games = add_r1_pitcher_features(games)  # history_source=None -> uses games itself

print(f"home_sp_r1_runs_avg missing: {games['home_sp_r1_runs_avg'].isna().mean():.1%}")
print(f"away_sp_r1_runs_avg missing: {games['away_sp_r1_runs_avg'].isna().mean():.1%}")

# %% [markdown]
# ## 8. Drop rows missing required features and save

# %%
required = [
    "home_sp_fip", "away_sp_fip",
    "home_team_wrc_plus", "away_team_wrc_plus",
    "park_factor_runs",
    "home_sp_r1_runs_avg", "away_sp_r1_runs_avg",
    "home_sp_r1_run_rate", "away_sp_r1_run_rate",
]
before = len(games)
games = games.dropna(subset=required)
print(f"Dropped {before - len(games)} rows with missing rolling stats "
      f"({before} -> {len(games)})")

games = games.sort_values("date")
games.to_csv("historical_r1_games_raw.csv", index=False)
print(f"Saved {len(games)} games to historical_r1_games_raw.csv")

# %% [markdown]
# ## Notes / known limitations
#
# - home_team_wrc_plus / away_team_wrc_plus are actually a rolling
#   runs-per-game proxy, NOT true park/league-adjusted wRC+.
# - park_factor_runs is a small hardcoded approximate table, not pulled live.
# - home_sp_r1_runs_avg / away_sp_r1_runs_avg / *_run_rate are a pitcher's
#   OWN rolling 1st-inning performance (last N starts, leak-safe, built
#   from r1_home_runs/r1_away_runs already in this dataset — see
#   first_inning_pitcher_features.py). *_starts_n tells you how many prior
#   starts each average is based on — worth checking whether low-n rows
#   (pitchers early in their career/season) hurt calibration.
# - Delete r1_games_cache.csv if you ever want to force a full clean re-pull
#   instead of resuming.
# - TARGET_SAMPLE_SIZE controls the subsample of OLDER seasons only; the
#   current season (2026) is always kept in full. Raise TARGET_SAMPLE_SIZE
#   if cross-validation results suggest you need more historical data.
# - Steps 4b and 5b dedupe schedule/cache by game_pk before merging, and
#   step 6 asserts the merged row count can't exceed either input — this
#   guards against the cache silently accumulating duplicate rows across
#   multiple resumed runs.
