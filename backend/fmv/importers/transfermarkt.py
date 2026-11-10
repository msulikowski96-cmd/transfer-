"""Importer for the open ``transfermarkt-datasets`` collection (github.com/dcaribou/transfermarkt-datasets).

What the source gives us and how it maps onto the schema
-------------------------------------------------------
* players, clubs, competitions        -> players, clubs, leagues
* appearances (+ games)               -> player_statistics: minutes, goals, assists,
                                         appearances, plus team goals conceded and
                                         clean sheets derived from the match results
* transfers                           -> transfers (type inferred, see ``classify_transfers``)
* player_valuations                   -> market_valuations (reference only, never a label)

What it does NOT have, and how that is handled
----------------------------------------------
* No event data (xG, xA, shots, tackles, ...): those columns stay NULL and the
  league-adjusted performance index is built from what exists.
* No contract history: ``players.contract_expiration_date`` is today's contract
  and may be a renewal signed years after a given transfer. Using it for past
  deals would leak the future, so the contracts table is left empty.
* No injury history: the injuries table is left empty.
* No transfer type: loans and loan returns are inferred from the player coming
  back to the same club within 400 days (this also removes paid loan fees from
  the labels). Fee 0 -> free transfer, fee missing -> undisclosed (not a label).

Licence note: the collection is published as CC0, but its content originates
from Transfermarkt, whose terms restrict reuse. Treat it as research data, not as
a licensed source for a commercial product.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
FILES = ["competitions", "clubs", "players", "games", "appearances", "transfers", "player_valuations"]
SOURCE = "transfermarkt-datasets"

# Transfermarkt competition code -> (our league id, name, country)
LEAGUES = {
    "GB1": (1, "Premier League", "England"),
    "ES1": (2, "La Liga", "Spain"),
    "L1": (3, "Bundesliga", "Germany"),
    "IT1": (4, "Serie A", "Italy"),
    "FR1": (5, "Ligue 1", "France"),
    "PO1": (6, "Liga Portugal", "Portugal"),
    "NL1": (7, "Eredivisie", "Netherlands"),
    "PL1": (8, "Ekstraklasa", "Poland"),
}
OTHER_LEAGUE = (9, "Inne ligi", "—")  # clubs of other competitions in the source: buyers and sellers only

POSITIONS = {
    "Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "DM", "Central Midfield": "CM", "Attacking Midfield": "AM",
    "Left Midfield": "W", "Right Midfield": "W", "Left Winger": "W", "Right Winger": "W",
    "Centre-Forward": "ST", "Second Striker": "ST",
}
POSITION_FALLBACK = {"Goalkeeper": "GK", "Defender": "CB", "Midfield": "CM", "Attack": "ST"}
GROUP = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "W": "FWD", "ST": "FWD"}

LOAN_DAYS = 400


def download(dest: Path, base_url: str = BASE_URL, force: bool = False) -> Path:
    """Fetch the gzipped CSV files into ``dest`` (skips files that are already there)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / f"{name}.csv.gz"
        if target.exists() and not force:
            continue
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(f"{base_url}/{name}.csv.gz", timeout=300) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(target)
    return dest


def _read(src: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(src / f"{name}.csv.gz", **kw)


def classify_transfers(t: pd.DataFrame) -> pd.DataFrame:
    """Adds ``type``: permanent | loan | loan_return | free.

    A move A -> B followed by the same player going B -> A within LOAN_DAYS is a
    loan and its return, whatever fee is attached to the first leg.
    """
    t = t.sort_values(["player_id", "date"]).reset_index(drop=True)
    t["type"] = np.where(t["fee_eur"] == 0, "free", "permanent")
    nxt = t.groupby("player_id").shift(-1)
    back = ((nxt["from_club_tm"] == t["to_club_tm"]) & (nxt["to_club_tm"] == t["from_club_tm"])
            & ((nxt["date"] - t["date"]).dt.days <= LOAN_DAYS) & ~(nxt["fee_eur"] > 0))
    t.loc[back, "type"] = "loan"
    ret = back.shift(1, fill_value=False) & (t["player_id"] == t["player_id"].shift(1))
    t.loc[ret, "type"] = "loan_return"
    return t


def load(src: Path, live_cache: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build the source tables of our schema from the downloaded files.

    With ``live_cache`` (responses fetched by ``tm_live.fetch``) the dump is enriched with
    point-in-time contracts, injury history and Ekstraklasa league statistics.
    """
    comps = _read(src, "competitions")
    clubs_raw = _read(src, "clubs")
    games = _read(src, "games", parse_dates=["date"],
                  usecols=["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
                           "home_club_goals", "away_club_goals"])
    apps = _read(src, "appearances", parse_dates=["date"],
                 usecols=["game_id", "player_id", "player_club_id", "competition_id", "date", "goals", "assists", "minutes_played"])
    data_end = games["date"].max()

    # ---- leagues and clubs ----------------------------------------------------------
    leagues = pd.DataFrame([{"id": i, "name": n, "country": c} for i, n, c in [*LEAGUES.values(), OTHER_LEAGUE]])
    league_of = clubs_raw["domestic_competition_id"].map(lambda c: LEAGUES.get(c, OTHER_LEAGUE)[0])
    clubs = pd.DataFrame({"id": clubs_raw["club_id"].astype(int), "name": clubs_raw["name"], "league_id": league_of})
    club_ids = set(clubs["id"])

    # ---- seasons: start year, Aug 1 - May 31 (the convention used everywhere else) ----
    league_games = games[games["competition_id"].isin(LEAGUES)].copy()
    years = sorted(y for y in league_games["season"].unique() if pd.Timestamp(int(y) + 1, 5, 31) < data_end)
    seasons = pd.DataFrame({"id": years, "name": [f"{y}/{str(y + 1)[2:]}" for y in years],
                            "start_date": [pd.Timestamp(y, 8, 1) for y in years],
                            "end_date": [pd.Timestamp(y + 1, 5, 31) for y in years]})
    league_games = league_games[league_games["season"].isin(years)]

    # ---- team context per club and season --------------------------------------------
    home = league_games.rename(columns={"home_club_id": "club_id", "home_club_goals": "gf", "away_club_goals": "ga"})
    away = league_games.rename(columns={"away_club_id": "club_id", "away_club_goals": "gf", "home_club_goals": "ga"})
    side = pd.concat([home, away])[["game_id", "season", "club_id", "gf", "ga"]]
    side["pts"] = np.select([side["gf"] > side["ga"], side["gf"] == side["ga"]], [3, 1], 0)
    team = side.groupby(["club_id", "season"]).agg(games=("game_id", "size"), ppg=("pts", "mean")).reset_index()

    # ---- player statistics (domestic league appearances only) ----------------------
    a = apps[apps["game_id"].isin(league_games["game_id"])].merge(
        side[["game_id", "club_id", "ga", "season"]], left_on=["game_id", "player_club_id"], right_on=["game_id", "club_id"])
    a = a[a["club_id"].isin(club_ids)]
    a["conceded"] = a["ga"] * a["minutes_played"] / 90.0
    a["clean"] = ((a["ga"] == 0) & (a["minutes_played"] >= 60)).astype(int)
    a["league_id"] = a["competition_id"].map(lambda c: LEAGUES[c][0])
    stats = a.groupby(["player_id", "season", "club_id", "league_id"]).agg(
        first=("date", "min"), appearances=("game_id", "size"), minutes=("minutes_played", "sum"),
        goals=("goals", "sum"), assists=("assists", "sum"), goals_conceded=("conceded", "sum"),
        clean_sheets=("clean", "sum")).reset_index()
    stats["goals_conceded"] = stats["goals_conceded"].round().astype(int)
    stats["stint"] = stats.groupby(["player_id", "season"])["first"].rank(method="first").astype(int)
    stats = stats.merge(team.rename(columns={"season": "season"}), on=["club_id", "season"], how="left")
    stats["team_minutes"] = (stats["games"].fillna(0) * 90).astype(int)
    stats["team_points_per_game"] = stats["ppg"].round(3)
    stats = stats.rename(columns={"season": "season_id"})
    if live_cache is not None:
        stats = pd.concat([stats, _ekstraklasa_stats(live_cache, team, years, club_ids, stats)], ignore_index=True)
    for col in ["xg", "xa", "shots", "key_passes", "progressive_passes", "tackles", "interceptions", "aerials_won", "saves"]:
        stats[col] = np.nan  # not published by this source
    stats = stats.drop(columns=["first", "games", "ppg"])

    # ---- players --------------------------------------------------------------------
    pl = _read(src, "players", parse_dates=["date_of_birth"])
    pl = pl[pl["player_id"].isin(stats["player_id"]) & pl["date_of_birth"].notna()]
    pos = pl["sub_position"].map(POSITIONS).fillna(pl["position"].map(POSITION_FALLBACK))
    pl = pl.assign(pos=pos)[pos.notna()]
    players = pd.DataFrame({
        "id": pl["player_id"].astype(int), "name": pl["name"], "birth_date": pl["date_of_birth"].dt.normalize(),
        "nationality": pl["country_of_citizenship"].fillna("unknown"), "position": pl["pos"],
        "position_group": pl["pos"].map(GROUP), "foot": pl["foot"].where(pl["foot"].notna(), None),
        "height_cm": pl["height_in_cm"].where(pl["height_in_cm"] > 0).astype("Int64"),
    })
    keep = set(players["id"])
    stats = stats[stats["player_id"].isin(keep)].reset_index(drop=True)
    stats.insert(0, "id", np.arange(1, len(stats) + 1))

    # ---- transfers -------------------------------------------------------------------
    tr = _read(src, "transfers", parse_dates=["transfer_date"])
    tr = tr[tr["player_id"].isin(keep) & (tr["transfer_date"] <= data_end)]
    tr = pd.DataFrame({"player_id": tr["player_id"].astype(int), "date": tr["transfer_date"],
                       "from_club_tm": tr["from_club_id"], "to_club_tm": tr["to_club_id"],
                       "fee_eur": pd.to_numeric(tr["transfer_fee"], errors="coerce")})
    tr = tr.drop_duplicates(["player_id", "date", "from_club_tm", "to_club_tm"])
    tr = classify_transfers(tr)
    transfers = pd.DataFrame({
        "id": np.arange(1, len(tr) + 1), "player_id": tr["player_id"], "date": tr["date"],
        "from_club_id": tr["from_club_tm"].where(tr["from_club_tm"].isin(club_ids)).astype("Int64"),
        "to_club_id": tr["to_club_tm"].where(tr["to_club_tm"].isin(club_ids)).astype("Int64"),
        "type": tr["type"],
        "fee_eur": np.where(tr["type"].isin(["permanent", "free"]), tr["fee_eur"], np.nan),
        "addons_eur": np.nan,
        "fee_confirmed": (tr["type"] == "free") | ((tr["type"] == "permanent") & (tr["fee_eur"] > 0)),
        "source": SOURCE,
    })

    # ---- reference valuations (benchmark only) --------------------------------------------
    val = _read(src, "player_valuations", parse_dates=["date"], usecols=["player_id", "date", "market_value_in_eur"])
    val = val[val["player_id"].isin(keep) & (val["market_value_in_eur"] > 0) & (val["date"] <= data_end)]
    valuations = pd.DataFrame({"id": np.arange(1, len(val) + 1), "player_id": val["player_id"].astype(int),
                               "date": val["date"].dt.normalize(), "value_eur": val["market_value_in_eur"].astype(float),
                               "source": "transfermarkt"})

    contracts = pd.DataFrame(columns=["player_id", "club_id", "start_date", "end_date", "source"])
    injuries = pd.DataFrame(columns=["player_id", "start_date", "end_date", "days_missed", "type"])
    if live_cache is not None:
        from . import tm_live

        # today's contract is visible from the last yearly snapshot on (used for current valuations only)
        visible = pd.Timestamp(data_end.year, 6, 14) if data_end.month >= 7 else pd.Timestamp(data_end.year - 1, 6, 14)
        contracts = pd.concat([tm_live.contracts(live_cache), tm_live.current_contracts(live_cache, players, visible)])
        contracts = contracts[contracts["player_id"].isin(keep) & contracts["club_id"].isin(club_ids)]
        injuries = tm_live.injuries(live_cache)
        injuries = injuries[injuries["player_id"].isin(keep) & (injuries["start_date"] <= data_end)]
    contracts = contracts.reset_index(drop=True)
    contracts.insert(0, "id", np.arange(1, len(contracts) + 1))
    injuries = injuries.reset_index(drop=True)
    injuries.insert(0, "id", np.arange(1, len(injuries) + 1))
    return {"leagues": leagues, "clubs": clubs, "seasons": seasons, "players": players, "player_statistics": stats,
            "transfers": transfers, "contracts": contracts, "injuries": injuries, "market_valuations": valuations}


def _ekstraklasa_stats(live_cache: Path, team: pd.DataFrame, years: list[int], club_ids: set,
                       existing: pd.DataFrame) -> pd.DataFrame:
    """Ekstraklasa league statistics from the live API (the dump has no Ekstraklasa appearances).

    Team context (games played, points per game) comes from the dump's match results where they
    exist (recent seasons only); otherwise team minutes are unknown (0) and points per game missing.
    """
    from . import tm_live

    s = tm_live.league_stats(live_cache, "PL1")
    s = s[s["season"].isin(years) & s["club_id"].isin(club_ids) & (s["appearances"] > 0)]
    s = s.groupby(["player_id", "season", "club_id"], as_index=False).sum()
    s["league_id"] = LEAGUES["PL1"][0]
    taken = existing.groupby(["player_id", "season_id"]).size().rename("n").reset_index()
    s = s.merge(taken.rename(columns={"season_id": "season"}), on=["player_id", "season"], how="left")
    s["stint"] = s.groupby(["player_id", "season"]).cumcount() + 1 + s["n"].fillna(0).astype(int)
    s = s.merge(team, on=["club_id", "season"], how="left")
    s["team_minutes"] = (s["games"].fillna(0) * 90).astype(int)
    s["team_points_per_game"] = s["ppg"].round(3)
    return s.drop(columns=["n", "games", "ppg"]).rename(columns={"season": "season_id"})
