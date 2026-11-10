"""Enrichment from Transfermarkt's JSON API via the ``transfermarkt-api`` SDK (github.com/felipeall/transfermarkt-api).

Fills the gaps of the static ``transfermarkt-datasets`` dump:

* contract with the selling club at the moment of every transfer (``remainingContractDays``),
* injury history with dates,
* league statistics for Ekstraklasa players (absent from the dump).

Responses are cached as JSON files (one per player and endpoint), so a fetch can be
interrupted and resumed, and imports never hit the network.

Data note: the SDK reads Transfermarkt's internal, undocumented API. Its MIT licence covers
the code, not the data; Transfermarkt's terms restrict automated collection. Research use only.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pandas as pd

ENDPOINTS = ("transfers", "injuries", "stats")
LEAGUE_CODES = ["GB1", "ES1", "L1", "IT1", "FR1", "PO1", "NL1", "PL1"]


def select_players(src: Path) -> dict[str, list[int]]:
    """Players worth enriching, from the downloaded dump files.

    labels:     sold for a disclosed fee by a club of the covered leagues since 2012
    current:    played in a covered league in the last season of the dump
    ekstraklasa: currently in Ekstraklasa, or moved to/from an Ekstraklasa club since 2014
    """
    clubs = pd.read_csv(src / "clubs.csv.gz")
    ours = set(clubs.loc[clubs["domestic_competition_id"].isin(LEAGUE_CODES), "club_id"])
    pl1 = set(clubs.loc[clubs["domestic_competition_id"] == "PL1", "club_id"])
    tr = pd.read_csv(src / "transfers.csv.gz", parse_dates=["transfer_date"])
    games = pd.read_csv(src / "games.csv.gz", usecols=["date"], parse_dates=["date"])
    end = games["date"].max()
    tr = tr[(tr["transfer_date"] >= "2012-01-01") & (tr["transfer_date"] <= end)]
    labels = set(tr.loc[(tr["transfer_fee"] > 0) & tr["from_club_id"].isin(ours), "player_id"])
    apps = pd.read_csv(src / "appearances.csv.gz", usecols=["player_id", "competition_id", "date"], parse_dates=["date"])
    season_start = pd.Timestamp(end.year - (1 if end.month < 8 else 0), 8, 1)
    current = set(apps.loc[apps["competition_id"].isin(LEAGUE_CODES) & (apps["date"] >= season_start - pd.DateOffset(years=1)), "player_id"])
    players = pd.read_csv(src / "players.csv.gz", usecols=["player_id", "current_club_domestic_competition_id", "last_season"])
    ekstra = set(players.loc[(players["current_club_domestic_competition_id"] == "PL1") & (players["last_season"] >= end.year - 2), "player_id"])
    ekstra |= set(tr.loc[(tr["from_club_id"].isin(pl1) | tr["to_club_id"].isin(pl1)) & (tr["transfer_date"] >= "2014-01-01"), "player_id"])
    return {"labels": sorted(int(i) for i in labels), "current": sorted(int(i) for i in current),
            "ekstraklasa": sorted(int(i) for i in ekstra)}


def _path(cache: Path, endpoint: str, player_id: int) -> Path:
    return cache / endpoint / f"{player_id}.json"


async def _fetch_one(tm, endpoint: str, player_id: int) -> dict:
    if endpoint == "transfers":
        return (await tm.player_transfers(str(player_id))).model_dump(mode="json", by_alias=True)
    if endpoint == "stats":
        return (await tm.player_stats(str(player_id))).model_dump(mode="json", by_alias=True)
    if endpoint == "injuries":  # paginated, 15 per page
        first = (await tm.player_injuries(str(player_id))).model_dump(mode="json", by_alias=True)
        injuries = list(first.get("injuries") or [])
        for page in range(2, int(first.get("lastPageNumber") or 1) + 1):
            nxt = (await tm.player_injuries(str(player_id), page_number=page)).model_dump(mode="json", by_alias=True)
            injuries += nxt.get("injuries") or []
        return {"id": str(player_id), "updatedAt": first.get("updatedAt"), "injuries": injuries}
    raise ValueError(endpoint)


async def _fetch(jobs: list[tuple[str, int]], cache: Path, concurrency: int, delay: float) -> dict:
    from transfermarkt_api import Transfermarkt

    sem = asyncio.Semaphore(concurrency)
    done = {"ok": 0, "missing": 0, "error": 0}
    started = time.time()

    async with Transfermarkt() as tm:
        async def run(endpoint: str, player_id: int) -> None:
            async with sem:
                target = _path(cache, endpoint, player_id)
                for attempt in range(4):
                    try:
                        data = await _fetch_one(tm, endpoint, player_id)
                        done["ok"] += 1
                        break
                    except Exception as exc:  # noqa: BLE001 - the SDK raises its own hierarchy
                        name = type(exc).__name__.lower()
                        if "notfound" in name or "404" in str(exc):
                            data = {"id": str(player_id), "missing": True}
                            done["missing"] += 1
                            break
                        if attempt == 3:
                            done["error"] += 1
                            return
                        await asyncio.sleep(5 * 2 ** attempt)  # blocked or throttled: back off
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(data, ensure_ascii=False))
                await asyncio.sleep(delay)
                n = sum(done.values())
                if n % 500 == 0:
                    print(f"  {n}/{len(jobs)} in {time.time() - started:.0f}s {done}", flush=True)

        await asyncio.gather(*(run(e, p) for e, p in jobs))
    return done


def fetch(src: Path, cache: Path, concurrency: int = 3, delay: float = 0.2) -> dict:
    """Fetch everything that is not cached yet. Safe to re-run."""
    groups = select_players(src)
    everyone = sorted(set(groups["labels"]) | set(groups["current"]) | set(groups["ekstraklasa"]))
    jobs = [(e, p) for p in everyone for e in ("transfers", "injuries")]
    jobs += [("stats", p) for p in groups["ekstraklasa"]]
    todo = [(e, p) for e, p in jobs if not _path(cache, e, p).exists()]
    print(f"players={len(everyone)} jobs={len(jobs)} to fetch={len(todo)}", flush=True)
    result = asyncio.run(_fetch(todo, cache, concurrency, delay)) if todo else {}
    return {"players": len(everyone), "fetched": result}


def _load(cache: Path, endpoint: str) -> dict[int, dict]:
    out = {}
    for f in (cache / endpoint).glob("*.json") if (cache / endpoint).exists() else []:
        data = json.loads(f.read_text())
        if not data.get("missing"):
            out[int(f.stem)] = data
    return out


def contracts(cache: Path) -> pd.DataFrame:
    """Contract with the selling club as it stood on each transfer date.

    ``remainingContractDays`` is recorded at the transfer, so the end date is point-in-time.
    The row becomes visible two days before the deal: features of the deal (as of the day
    before) see it, earlier snapshots do not, because nothing says when that contract was signed.
    ``contractUntil`` is ignored: it is today's contract with the buying club (renewals included).
    """
    rows = []
    for pid, data in _load(cache, "transfers").items():
        for t in data.get("transfers") or []:
            days, club = t.get("remainingContractDays"), (t.get("clubFrom") or {}).get("id")
            if days is None or not t.get("date") or t.get("upcoming") or not club:
                continue
            date = pd.Timestamp(t["date"])
            rows.append({"player_id": pid, "club_id": int(club), "start_date": date - pd.Timedelta(days=2),
                         "end_date": date + pd.Timedelta(days=int(days)), "source": "tm-api:transfer"})
    return pd.DataFrame(rows, columns=["player_id", "club_id", "start_date", "end_date", "source"])


def current_contracts(cache: Path, players: pd.DataFrame, visible_from: pd.Timestamp) -> pd.DataFrame:
    """Today's contract with the current club (``contractUntil`` of the latest transfer), for the latest valuation only."""
    rows = []
    for pid, data in _load(cache, "transfers").items():
        past = [t for t in data.get("transfers") or [] if not t.get("upcoming") and t.get("date")]
        if not past:
            continue
        last = max(past, key=lambda t: t["date"])
        if last.get("contractUntil") and (last.get("clubTo") or {}).get("id"):
            rows.append({"player_id": pid, "club_id": int(last["clubTo"]["id"]), "start_date": visible_from,
                         "end_date": pd.Timestamp(last["contractUntil"]), "source": "tm-api:current"})
    out = pd.DataFrame(rows, columns=["player_id", "club_id", "start_date", "end_date", "source"])
    return out[out["player_id"].isin(set(players["id"]))]


def injuries(cache: Path) -> pd.DataFrame:
    rows = []
    for pid, data in _load(cache, "injuries").items():
        for i in data.get("injuries") or []:
            if not i.get("fromDate"):
                continue
            start = pd.Timestamp(i["fromDate"])
            end = pd.Timestamp(i["untilDate"]) if i.get("untilDate") else start + pd.Timedelta(days=int(i.get("days") or 0))
            rows.append({"player_id": pid, "start_date": start, "end_date": end,
                         "days_missed": int(i.get("days") or (end - start).days), "type": str(i.get("injury") or "")[:40]})
    return pd.DataFrame(rows, columns=["player_id", "start_date", "end_date", "days_missed", "type"])


def league_stats(cache: Path, competition: str = "PL1") -> pd.DataFrame:
    """Per player, season and club league totals for one competition (used for Ekstraklasa)."""
    rows = []
    for pid, data in _load(cache, "stats").items():
        for s in data.get("stats") or []:
            if s.get("competitionId") != competition or not s.get("clubId") or not s.get("seasonId"):
                continue
            rows.append({"player_id": pid, "season": int(s["seasonId"]), "club_id": int(s["clubId"]),
                         "appearances": int(s.get("appearances") or 0), "minutes": int(s.get("minutesPlayed") or 0),
                         "goals": int(s.get("goals") or 0), "assists": int(s.get("assists") or 0),
                         "goals_conceded": int(s.get("goalsConceded") or 0), "clean_sheets": int(s.get("cleanSheets") or 0)})
    return pd.DataFrame(rows, columns=["player_id", "season", "club_id", "appearances", "minutes", "goals", "assists",
                                       "goals_conceded", "clean_sheets"])
