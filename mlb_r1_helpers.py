"""
Shared helpers for the 1st-inning (R1) Over/Under pipeline.
Used by both build_r1_dataset.py and r1_over_under_model.py so the training
data and the live prediction data are built the exact same way.

Data source: MLB Stats API only (statsapi.mlb.com) — free, no key, no
pybaseball dependency. Run locally; this needs real internet access.
"""

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import numpy as np

# Shared session with retries — a handful of these calls will always time out
# or hiccup over hundreds of requests, so retry instead of crashing the run.
SESSION = requests.Session()
_retries = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=_retries))
REQUEST_TIMEOUT = 15

FIP_CONSTANT = 3.10          # approximate league constant across recent seasons
TRAILING_STARTS = 5          # prior starts used for rolling pitcher FIP
TRAILING_GAMES = 10          # prior games used for rolling team offense proxy

# Stable MLB team IDs -> abbreviation. The schedule endpoint doesn't reliably
# return an "abbreviation" field (it only appears when hydrated), which was
# silently breaking the park_factor_runs join (falling back to full team
# names, which never match PARK_FACTORS_RUNS keys). Team IDs are stable, so
# this mapping is the reliable source of truth instead.
TEAM_ID_TO_ABBREV = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SD", 136: "SEA",
    137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Approximate single-value park run factors (100 = neutral). These are
# illustrative/rounded, not pulled live — replace with a sourced table
# (e.g. FanGraphs' published Guts park factors) if you want more precision.
PARK_FACTORS_RUNS = {
    "COL": 112, "CIN": 105, "BOS": 104, "TEX": 103, "PHI": 102,
    "BAL": 101, "TOR": 101, "CWS": 101, "MIL": 100, "ATL": 100,
    "ARI": 100, "AZ": 100, "HOU": 99, "STL": 99, "WSH": 99, "MIN": 99,
    "CHC": 98, "LAA": 98, "KC": 98, "TB": 97, "NYY": 97, "CLE": 97,
    "DET": 96, "SD": 96, "SF": 95, "PIT": 95, "SEA": 94, "NYM": 94,
    "LAD": 94, "MIA": 93, "ATH": 97, "OAK": 97,
}


# ---------------------------------------------------------------------------
# Schedule + real R1 (1st inning only) results
# ---------------------------------------------------------------------------

def get_schedule(start_date: str, end_date: str, finished_only: bool = True) -> pd.DataFrame:
    """start_date/end_date as 'YYYY-MM-DD'. Set finished_only=False to include
    scheduled/upcoming games (needed for predicting today/tomorrow)."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
        "hydrate": "probablePitcher",
    }
    resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("detailedState")
            if finished_only and state != "Final":
                continue
            if not finished_only and state not in ("Scheduled", "Pre-Game", "Warmup"):
                continue

            home_team = g["teams"]["home"]["team"]
            away_team = g["teams"]["away"]["team"]
            home_pp = g["teams"]["home"].get("probablePitcher", {})
            away_pp = g["teams"]["away"].get("probablePitcher", {})

            rows.append({
                "game_pk": g["gamePk"],
                "date": day["date"],
                "home": TEAM_ID_TO_ABBREV.get(home_team["id"], home_team["name"]),
                "away": TEAM_ID_TO_ABBREV.get(away_team["id"], away_team["name"]),
                "home_team_id": home_team["id"],
                "away_team_id": away_team["id"],
                "home_sp_id": home_pp.get("id"),
                "away_sp_id": away_pp.get("id"),
            })
    return pd.DataFrame(rows)


def _get_starting_pitcher_id(team_box: dict):
    pitchers = team_box.get("pitchers", [])
    return pitchers[0] if pitchers else None


def get_r1_runs(game_pk: int) -> dict | None:
    """Fetch ONLY inning-1 runs + starters, via the lightweight linescore and
    boxscore endpoints instead of the much larger full play-by-play feed."""
    try:
        line_resp = SESSION.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore",
            timeout=REQUEST_TIMEOUT,
        )
        line_resp.raise_for_status()
        line_data = line_resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  skipping game {game_pk} (linescore fetch failed: {e})")
        return None

    innings = line_data.get("innings", [])
    inning_1 = next((i for i in innings if i.get("num") == 1), None)
    if inning_1 is None:
        return None
    home_runs = inning_1.get("home", {}).get("runs", 0) or 0
    away_runs = inning_1.get("away", {}).get("runs", 0) or 0

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
    except requests.exceptions.RequestException as e:
        print(f"  game {game_pk}: boxscore fetch failed ({e}), starters unknown")

    return {
        "game_pk": game_pk,
        "r1_home_runs": home_runs,
        "r1_away_runs": away_runs,
        "r1_total_runs": home_runs + away_runs,
        "over_0_5": int((home_runs + away_runs) > 0),
        # overwrite probable-pitcher IDs with actual starters, if available
        "home_sp_id_actual": home_sp_id,
        "away_sp_id_actual": away_sp_id,
    }


# ---------------------------------------------------------------------------
# Rolling (pre-game, non-leaky) starting pitcher FIP
# ---------------------------------------------------------------------------

_pitcher_log_cache: dict[tuple[int, int], pd.DataFrame] = {}


def get_pitcher_game_log(pitcher_id: int, season: int) -> pd.DataFrame:
    key = (pitcher_id, season)
    if key in _pitcher_log_cache:
        return _pitcher_log_cache[key]
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    try:
        resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        _pitcher_log_cache[key] = pd.DataFrame()
        return _pitcher_log_cache[key]
    payload = resp.json()
    stats_blocks = payload.get("stats", [])
    splits = stats_blocks[0].get("splits", []) if stats_blocks else []
    rows = []
    for s in splits:
        stat = s.get("stat", {})
        if not stat.get("gamesStarted"):
            continue  # relief outings excluded — we want starts only
        rows.append({
            "date": s.get("date"),
            "ip": float(stat.get("inningsPitched", 0) or 0),
            "hr": stat.get("homeRuns", 0) or 0,
            "bb": stat.get("baseOnBalls", 0) or 0,
            "hbp": stat.get("hitByPitch", 0) or 0,
            "k": stat.get("strikeOuts", 0) or 0,
        })
    df = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    _pitcher_log_cache[key] = df
    return df


def trailing_fip(pitcher_id, game_date: str, season: int) -> float:
    if pd.isna(pitcher_id):
        return np.nan
    log = get_pitcher_game_log(int(pitcher_id), season)
    if log.empty:
        return np.nan
    prior = log[log["date"] < game_date].tail(TRAILING_STARTS)
    ip = prior["ip"].sum()
    if ip == 0:
        return np.nan
    hr, bb, hbp, k = prior["hr"].sum(), prior["bb"].sum(), prior["hbp"].sum(), prior["k"].sum()
    return ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + FIP_CONSTANT


# ---------------------------------------------------------------------------
# Rolling team offense proxy (stand-in for wRC+ — see module docstring)
# ---------------------------------------------------------------------------

_team_log_cache: dict[tuple[int, int], pd.DataFrame] = {}


def get_team_game_log(team_id: int, season: int) -> pd.DataFrame:
    key = (team_id, season)
    if key in _team_log_cache:
        return _team_log_cache[key]
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {"stats": "gameLog", "group": "hitting", "season": season}
    try:
        resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        _team_log_cache[key] = pd.DataFrame()
        return _team_log_cache[key]
    payload = resp.json()
    stats_blocks = payload.get("stats", [])
    splits = stats_blocks[0].get("splits", []) if stats_blocks else []
    rows = [{"date": s.get("date"), "runs": s.get("stat", {}).get("runs", 0) or 0} for s in splits]
    df = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    _team_log_cache[key] = df
    return df


def trailing_runs_per_game(team_id, game_date: str, season: int) -> float:
    if pd.isna(team_id):
        return np.nan
    log = get_team_game_log(int(team_id), season)
    if log.empty:
        return np.nan
    prior = log[log["date"] < game_date].tail(TRAILING_GAMES)
    if prior.empty:
        return np.nan
    return prior["runs"].mean()


def park_factor_for(home_abbrev: str) -> float:
    return PARK_FACTORS_RUNS.get(home_abbrev, 100)


def add_features(games: pd.DataFrame, sp_id_col_home="home_sp_id", sp_id_col_away="away_sp_id") -> pd.DataFrame:
    """Adds home_sp_fip, away_sp_fip, home_team_wrc_plus, away_team_wrc_plus,
    park_factor_runs to a games dataframe that has: date, home, away,
    home_team_id, away_team_id, and the two starting-pitcher id columns."""
    games = games.copy()
    games["season"] = pd.to_datetime(games["date"]).dt.year

    games["home_sp_fip"] = games.apply(
        lambda r: trailing_fip(r[sp_id_col_home], r["date"], r["season"]), axis=1
    )
    games["away_sp_fip"] = games.apply(
        lambda r: trailing_fip(r[sp_id_col_away], r["date"], r["season"]), axis=1
    )
    games["home_team_wrc_plus"] = games.apply(
        lambda r: trailing_runs_per_game(r["home_team_id"], r["date"], r["season"]), axis=1
    )
    games["away_team_wrc_plus"] = games.apply(
        lambda r: trailing_runs_per_game(r["away_team_id"], r["date"], r["season"]), axis=1
    )
    games["park_factor_runs"] = games["home"].apply(park_factor_for)
    return games

