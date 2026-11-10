"""Synthetic football world used until licensed data sources are connected.

IMPORTANT: every player, club, fee and statistic produced here is fictional.
The generator exists so that the whole pipeline (schema -> features -> models ->
evaluation -> API) can be developed and tested end to end. Metrics measured on
this data say that the *method* recovers a known signal; they say nothing about
accuracy on the real transfer market.

The world has a hidden "ability" per player and a hidden latent value that
drives fees. Neither is written to the database: models only see what a real
data provider would publish (statistics, contracts, injuries, transfers).
The hidden truth is returned separately for diagnostics in tests.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Hidden league strength (1.0 = strongest). The League Adjustment Engine has to
# rediscover the ordering from players moving between leagues.
_STRENGTH = [1.00, 0.92, 0.90, 0.89, 0.82, 0.68, 0.66, 0.45]
_N_CLUBS = [20, 20, 18, 20, 18, 18, 18, 18]

_POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
_POS_P = [0.09, 0.17, 0.14, 0.10, 0.14, 0.09, 0.13, 0.14]
_GROUP = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "W": "FWD", "ST": "FWD"}
_POS_VALUE = {"GK": 0.55, "DEF": 0.85, "MID": 1.0, "FWD": 1.18}
# per-90 base rates for a league-average player: xg, xa, shots, key passes,
# progressive passes, tackles, interceptions, aerials won
_BASE = {
    "GK": (0.00, 0.005, 0.0, 0.05, 1.0, 0.05, 0.1, 0.3),
    "CB": (0.04, 0.02, 0.5, 0.3, 3.5, 1.6, 1.7, 3.0),
    "FB": (0.03, 0.09, 0.5, 1.0, 4.0, 2.2, 1.3, 1.2),
    "DM": (0.05, 0.06, 0.8, 0.9, 5.5, 2.8, 1.8, 1.5),
    "CM": (0.09, 0.11, 1.2, 1.4, 5.5, 2.0, 1.1, 1.0),
    "AM": (0.20, 0.20, 2.2, 2.2, 4.5, 1.2, 0.6, 0.6),
    "W": (0.24, 0.18, 2.4, 1.8, 3.2, 1.3, 0.6, 0.6),
    "ST": (0.42, 0.10, 3.0, 1.0, 1.5, 0.6, 0.3, 2.4),
}
_NATIONS = ["England", "Spain", "Germany", "Italy", "France", "Portugal", "Netherlands", "Poland",
            "Brazil", "Argentina", "Belgium", "Croatia", "Denmark", "Senegal", "Nigeria", "Serbia",
            "Sweden", "Austria", "Ukraine", "Japan"]
_SYL_A = ["Ar", "Bel", "Cor", "Dan", "El", "Fab", "Gor", "Hal", "Iv", "Jar", "Kal", "Lor", "Mar", "Nor",
          "Or", "Pal", "Quin", "Ros", "Sal", "Tor", "Ul", "Val", "Wen", "Xav", "Yor", "Zel"]
_SYL_B = ["an", "ek", "io", "os", "ik", "en", "ar", "us", "el", "on", "as", "im"]
_SYL_C = ["ski", "ez", "sen", "ov", "ini", "aro", "mann", "eira", "ic", "son", "etti", "aux", "berg", "ak"]
_CLUB_SUFFIX = ["FC", "United", "City", "Athletic", "Sporting", "Rovers", "Union", "Racing"]


@dataclass
class SyntheticWorld:
    tables: dict[str, pd.DataFrame]
    truth: pd.DataFrame  # player_id, season_id, ability, latent_value_eur (never stored in the DB)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed: int = 7, n_players: int = 6000, first_season: int = 2012, last_season: int = 2024) -> SyntheticWorld:
    rng = np.random.default_rng(seed)
    years = list(range(first_season, last_season + 1))

    leagues = pd.DataFrame({"id": range(1, 9), "name": [n for n, _ in config.LEAGUES],
                            "country": [c for _, c in config.LEAGUES]})
    league_q = {i + 1: 0.44 + 0.36 * s for i, s in enumerate(_STRENGTH)}  # average ability level
    seasons = pd.DataFrame({
        "id": years, "name": [f"{y}/{str(y + 1)[2:]}" for y in years],
        "start_date": [dt.date(y, 8, 1) for y in years], "end_date": [dt.date(y + 1, 5, 31) for y in years],
    })

    clubs, used = [], set()
    for lid, n in zip(leagues["id"], _N_CLUBS):
        for _ in range(n):
            while True:
                name = f"{rng.choice(_SYL_A)}{rng.choice(_SYL_B)}{rng.choice(['', 'ton', 'burg', 'ia', 'ova'])} {rng.choice(_CLUB_SUFFIX)}"
                if name not in used:
                    used.add(name)
                    break
            dq = rng.normal(0, 0.045)
            clubs.append({"id": len(clubs) + 1, "name": name, "league_id": lid, "q": league_q[lid] + dq,
                          "logw": 2.6 * _STRENGTH[lid - 1] + 5.0 * dq + rng.normal(0, 0.25)})
    clubs = pd.DataFrame(clubs)
    club_q = clubs["q"].to_numpy()
    club_league = clubs["league_id"].to_numpy()
    wealth_z = (clubs["logw"] - clubs["logw"].mean()) / clubs["logw"].std()
    wealth_z = wealth_z.to_numpy()
    club_ids = clubs["id"].to_numpy()

    def pick_club(target_q: float, exclude: int | None = None, wealth_pull: float = 0.3) -> int:
        w = np.exp(-(((club_q - target_q) / 0.04) ** 2) + wealth_pull * wealth_z)
        if exclude is not None:
            w[exclude - 1] = 0.0
        return int(rng.choice(club_ids, p=w / w.sum()))

    players: list[dict] = []
    state: dict[int, dict] = {}
    contracts: list[dict] = []
    transfers: list[dict] = []
    stats: list[dict] = []
    injuries: list[dict] = []
    valuations: list[dict] = []
    truth: list[dict] = []

    def ability(p: dict, age: float) -> float:
        if age < p["peak"]:
            a = p["pot"] - 0.018 * p["dev"] * (p["peak"] - age) ** 1.3
        else:
            a = p["pot"] - 0.006 * (age - p["peak"]) ** 2
        return float(a + p["form"])

    def latent_value(p: dict, a: float, age: float, year: int) -> float:
        eff = a + (0.45 * max(0.0, p["pot_seen"] - a) if age < p["peak"] else 0.0)
        age_f = 1.0 if age <= 27 else np.exp(-0.16 * (age - 27) - (0.12 * (age - 31) if age > 31 else 0.0))
        return float(18e6 * np.exp(7.5 * (eff - 0.85)) * age_f * _POS_VALUE[p["group"]] * 1.06 ** (year - first_season))

    def sign(pid: int, club: int, start: dt.date, years_len: int) -> None:
        end = dt.date(start.year + years_len + (1 if start.month >= 7 else 0), 6, 30)
        contracts.append({"id": len(contracts) + 1, "player_id": pid, "club_id": club,
                          "start_date": start, "end_date": end, "source": "synthetic"})
        state[pid]["contract_end"] = end

    def new_player(season: int, age: float) -> None:
        pid = len(players) + 1
        pos = str(rng.choice(_POSITIONS, p=_POS_P))
        birth = dt.date(season, 8, 1) - dt.timedelta(days=int(age * 365.25))
        p = {"id": pid, "name": f"{rng.choice(_SYL_A)}{rng.choice(_SYL_B)} {rng.choice(_SYL_A)}{rng.choice(_SYL_C)}",
             "birth_date": birth, "nationality": str(rng.choice(_NATIONS)), "position": pos, "position_group": _GROUP[pos],
             "foot": str(rng.choice(["right", "left", "both"], p=[0.7, 0.24, 0.06])),
             "height_cm": int(rng.normal(190 if pos == "GK" else 186 if pos in ("CB", "ST") else 179, 5))}
        players.append(p)
        pot = float(np.clip(rng.normal(0.75, 0.10), 0.45, 1.0))
        s = {"group": _GROUP[pos], "pos": pos, "birth": birth, "pot": pot, "pot_seen": pot + rng.normal(0, 0.03),
             "peak": rng.normal(29 if pos == "GK" else 27, 1.2), "dev": rng.uniform(0.6, 1.4), "form": 0.0,
             "finishing": rng.normal(1.0, 0.08), "fragile": rng.gamma(2.0, 0.5), "loan_parent": None, "active": True}
        state[pid] = s
        a = ability(s, age)
        s["club"] = pick_club(a + rng.normal(0, 0.04))
        sign(pid, s["club"], dt.date(season, 7, 1) - dt.timedelta(days=int(rng.integers(0, 900))), int(rng.integers(2, 5)))
        # contracts signed in the past must still be running
        if s["contract_end"] <= dt.date(season, 8, 1):
            contracts[-1]["end_date"] = s["contract_end"] = dt.date(season + int(rng.integers(1, 4)), 6, 30)

    def record_transfer(pid: int, date: dt.date, to_club: int, kind: str, value: float = 0.0) -> None:
        s = state[pid]
        row = {"id": len(transfers) + 1, "player_id": pid, "date": date, "from_club_id": s["club"], "to_club_id": to_club,
               "type": kind, "fee_eur": None, "addons_eur": None, "fee_confirmed": False, "source": "synthetic"}
        if kind == "free":
            row.update(fee_eur=0.0, fee_confirmed=True)
        elif kind == "permanent":
            months = max(0.0, (s["contract_end"] - date).days / 30.44)
            contract_f = min(1.0, 0.35 + 0.65 * min(months, 36) / 36)
            window_end = dt.date(date.year, 9, 1) if date.month >= 6 else dt.date(date.year, 2, 1)
            deadline_f = 1.10 if (window_end - date).days <= 3 else 1.0
            fee = value * contract_f * np.exp(0.18 * wealth_z[to_club - 1]) * deadline_f * rng.lognormal(0, 0.22)
            fee = float(np.round(fee, -4 if fee < 2e6 else -5))
            if rng.random() < 0.87 and fee > 0:  # the rest stay "undisclosed"
                row.update(fee_eur=fee, fee_confirmed=True)
                if rng.random() < 0.3:
                    row["addons_eur"] = float(np.round(fee * rng.uniform(0.05, 0.25), -4))
        transfers.append(row)
        s["club"] = to_club
        if kind in ("permanent", "free"):
            sign(pid, to_club, date, int(rng.integers(3, 6)))

    def season_stint(pid: int, year: int, a: float, frac: float, start_frac: float, stint: int, ppg: dict) -> None:
        s = state[pid]
        club = s["club"]
        lid = int(club_league[club - 1])
        lq, cq = league_q[lid], club_q[club - 1]
        team_minutes = int(round(34 * 90 * frac))
        s_start = dt.date(year, 8, 1) + dt.timedelta(days=int(300 * start_frac))
        missed = 0
        for _ in range(rng.poisson(0.55 * s["fragile"] * frac)):
            days = int(min(240, rng.gamma(1.3, 28) + 4))
            st = s_start + dt.timedelta(days=int(rng.integers(0, max(2, int(300 * frac)))))
            injuries.append({"id": len(injuries) + 1, "player_id": pid, "start_date": st,
                             "end_date": st + dt.timedelta(days=days), "days_missed": days,
                             "type": str(rng.choice(["muscle", "knee", "ankle", "other"]))})
            missed += days
        avail = max(0.0, 1 - missed / (300 * frac))
        share = _sigmoid((a - cq) / 0.05 + rng.normal(0, 0.7)) * avail
        minutes = int(share * team_minutes * 0.97)
        m90 = minutes / 90
        att = np.exp(2.2 * (a - lq) + 1.0 * (cq - lq))
        dfn = np.exp(0.8 * (a - lq))
        b = _BASE[s["pos"]]
        xg = rng.gamma(max(b[0] * att * m90, 1e-6) * 4, 0.25)
        xa = rng.gamma(max(b[1] * att * m90, 1e-6) * 4, 0.25)
        row = {"id": len(stats) + 1, "player_id": pid, "season_id": year, "club_id": club, "league_id": lid, "stint": stint,
               "appearances": int(min(round(34 * frac), np.ceil(m90 * 1.25))), "minutes": minutes,
               "goals": int(rng.poisson(xg * s["finishing"])), "assists": int(rng.poisson(xa)),
               "xg": round(float(xg), 2), "xa": round(float(xa), 2),
               "shots": int(rng.poisson(b[2] * att ** 0.6 * m90)), "key_passes": int(rng.poisson(b[3] * att ** 0.8 * m90)),
               "progressive_passes": int(rng.poisson(b[4] * dfn * m90)), "tackles": int(rng.poisson(b[5] * dfn * m90)),
               "interceptions": int(rng.poisson(b[6] * dfn * m90)), "aerials_won": int(rng.poisson(b[7] * dfn * m90)),
               "saves": 0, "goals_conceded": 0, "clean_sheets": 0,
               "team_minutes": team_minutes, "team_points_per_game": ppg[club]}
        if s["pos"] == "GK":
            faced = rng.poisson(4.3 * np.exp(-1.5 * (cq - lq)) * m90)
            save_p = float(np.clip(0.70 + 0.5 * (a - lq), 0.5, 0.9))
            row["saves"] = int(rng.binomial(faced, save_p))
            row["goals_conceded"] = int(faced - row["saves"])
            row["clean_sheets"] = int(rng.binomial(int(m90), float(np.clip(np.exp(-(1 - save_p) * faced / max(m90, 1)), 0, 1))))
        stats.append(row)
        s["last_share"] = share

    # ---- initial population -------------------------------------------------
    for _ in range(int(n_players * 0.52)):
        new_player(first_season, float(np.clip(rng.triangular(17, 23, 35), 17, 35)))

    for year in years:
        ppg = {int(c): float(np.clip(1.38 + 6.0 * (club_q[c - 1] - league_q[int(club_league[c - 1])]) + rng.normal(0, 0.15), 0.5, 2.6))
               for c in club_ids}
        active = [pid for pid, s in state.items() if s["active"]]
        for pid in active:
            s = state[pid]
            age = (dt.date(year, 12, 31) - s["birth"]).days / 365.25
            s["form"] = 0.6 * s["form"] + rng.normal(0, 0.02)
            a = ability(s, age)
            val = latent_value(s, a, age, year)
            truth.append({"player_id": pid, "season_id": year, "ability": a, "latent_value_eur": val})
            lid = int(club_league[s["club"] - 1])
            valuations.append({"id": len(valuations) + 1, "player_id": pid, "date": dt.date(year + 1, 1, 1),
                               "value_eur": float(np.round(val * np.exp(0.3 * (_STRENGTH[lid - 1] - 0.8)) * rng.lognormal(0, 0.28), -4)),
                               "source": "synthetic-portal"})
            # winter move
            if s["loan_parent"] is None and rng.random() < 0.022 and s["contract_end"] > dt.date(year + 1, 6, 30):
                season_stint(pid, year, a, 0.5, 0.0, 1, ppg)
                date = dt.date(year + 1, 1, int(rng.integers(2, 32)))
                record_transfer(pid, date, pick_club(a + rng.normal(0.01, 0.03), s["club"]), "permanent", val)
                season_stint(pid, year, a, 0.5, 0.5, 2, ppg)
            else:
                season_stint(pid, year, a, 1.0, 0.0, 1, ppg)
            s["a"], s["val"], s["age"] = a, val, age
            valuations.append({"id": len(valuations) + 1, "player_id": pid, "date": dt.date(year + 1, 6, 1),
                               "value_eur": float(np.round(val * np.exp(0.3 * (_STRENGTH[lid - 1] - 0.8)) * rng.lognormal(0, 0.28), -4)),
                               "source": "synthetic-portal"})

        if year == last_season:
            break
        # ---- summer window of year+1 ----------------------------------------
        summer = dt.date(year + 1, 7, 1)
        for pid in active:
            s = state[pid]
            a, age = s["a"], s["age"]
            date = summer + dt.timedelta(days=int(min(62, rng.exponential(30)) if rng.random() < 0.85 else rng.integers(59, 63)))
            if s["loan_parent"] is not None:
                record_transfer(pid, summer, s["loan_parent"], "loan_return")
                s["loan_parent"] = None
                continue
            if age > 33 and rng.random() < 0.35 + 0.2 * (age - 33) or a < 0.42 and age > 24:
                s["active"] = False
                continue
            months = (s["contract_end"] - summer).days / 30.44
            if months <= 13 and rng.random() < (0.5 if months > 1 else 0.45):
                sign(pid, s["club"], summer, int(rng.integers(2, 5)))
                continue
            if 13 < months <= 25 and rng.random() < 0.3:
                sign(pid, s["club"], summer, int(rng.integers(3, 6)))
                continue
            if months <= 1:  # contract ran out
                if age > 31 and rng.random() < 0.4:
                    s["active"] = False
                else:
                    record_transfer(pid, date, pick_club(a + rng.normal(-0.01, 0.03), s["club"], 0.0), "free")
                continue
            cq = club_q[s["club"] - 1]
            if age <= 22 and s.get("last_share", 1) < 0.25 and rng.random() < 0.3:
                parent = s["club"]
                record_transfer(pid, date, pick_club(a + rng.normal(0, 0.02), parent, 0.0), "loan")
                s["loan_parent"] = parent
                continue
            p_move = 0.07 + 0.25 * _sigmoid((a - cq - 0.03) / 0.03) + 0.15 * _sigmoid((cq - a - 0.08) / 0.03) + (0.08 if months <= 13 else 0)
            if rng.random() < p_move:
                record_transfer(pid, date, pick_club(a + rng.normal(0.01, 0.03), s["club"]), "permanent", latent_value(s, a, age + 0.5, year + 1))
        # ---- rookies ----------------------------------------------------------
        for _ in range(int(n_players * 0.48 / (len(years) - 1))):
            new_player(year + 1, float(rng.uniform(17, 21)))

    tables = {
        "leagues": leagues, "clubs": clubs[["id", "name", "league_id"]], "seasons": seasons,
        "players": pd.DataFrame(players), "player_statistics": pd.DataFrame(stats),
        "transfers": pd.DataFrame(transfers), "contracts": pd.DataFrame(contracts),
        "injuries": pd.DataFrame(injuries), "market_valuations": pd.DataFrame(valuations),
    }
    return SyntheticWorld(tables=tables, truth=pd.DataFrame(truth))
