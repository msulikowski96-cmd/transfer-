"""Point-in-time feature engineering.

Every feature of a (player, as_of) snapshot is computed only from records that
were knowable strictly before ``as_of``:

* season statistics only from seasons that had *ended* before ``as_of``;
* the contract that was running then, with its end date as originally signed;
* injuries that had started before ``as_of`` (ongoing ones truncated at ``as_of``);
* transfers of the previous five years only; the market index from fees paid in the previous 24 months.

``tests/test_features.py`` checks this by deleting everything after a date and
asserting that the features do not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from . import db
from .league_adjustment import COUNT_COLS, LeagueAdjuster, add_rates

RATE_COLS = ["goals90", "assists90", "xg90", "xa90", "shots90", "key_passes90", "progressive_passes90",
             "def_actions90", "aerials_won90", "save_pct", "clean_sheet_rate"]

PLAYER_NUM = (
    ["age", "height_cm", "l1_minutes", "l1_minutes_share", "l1_appearances"]
    + [f"l1_{c}" for c in RATE_COLS]
    + ["l1_idx_raw", "l1_idx_adj", "l2_minutes", "l2_idx_adj", "idx_trend",
       "career_minutes", "career_seasons", "career_idx_adj", "best_idx_adj",
       "league_difficulty", "club_ppg_last", "contract_months",
       "inj_days_365", "inj_days_730", "inj_count_730",
       "n_prev_transfers", "n_prev_loans", "last_fee_rel", "max_fee_rel"]
)
PLAYER_CAT = ["position", "position_group", "foot", "nationality", "league"]

DEAL_NUM = ["buyer_league_difficulty", "league_step", "cross_league", "buyer_ppg_last",
            "buyer_spend_3y_rel", "buyer_max_fee_3y_rel", "seller_income_3y_rel", "days_to_window_close"]
DEAL_CAT = ["buyer_league", "window"]

# transfer history looks back this far (again a fixed window, see "career" features)
HISTORY_DAYS = 5 * 365

# human readable factor groups used by the explanation engine
FACTOR_GROUPS = {
    "Wiek i potencjał": ["age", "idx_trend"],
    "Statystyki ofensywne": ["l1_goals90", "l1_assists90", "l1_xg90", "l1_xa90", "l1_shots90", "l1_key_passes90"],
    "Gra defensywna i rozegranie": ["l1_progressive_passes90", "l1_def_actions90", "l1_aerials_won90",
                                    "l1_save_pct", "l1_clean_sheet_rate"],
    "Ogólny poziom gry (skorygowany o ligę)": ["l1_idx_raw", "l1_idx_adj", "l2_idx_adj", "career_idx_adj", "best_idx_adj"],
    "Minuty i doświadczenie": ["l1_minutes", "l1_minutes_share", "l1_appearances", "l2_minutes",
                               "career_minutes", "career_seasons"],
    "Poziom ligi i klubu": ["league", "league_difficulty", "club_ppg_last"],
    "Długość kontraktu": ["contract_months"],
    "Kontuzje i dostępność": ["inj_days_365", "inj_days_730", "inj_count_730"],
    "Historia transferowa": ["n_prev_transfers", "n_prev_loans", "last_fee_rel", "max_fee_rel"],
    "Pozycja i profil": ["position", "position_group", "foot", "nationality", "height_cm"],
}


@dataclass
class Data:
    leagues: pd.DataFrame
    clubs: pd.DataFrame
    seasons: pd.DataFrame
    players: pd.DataFrame
    stats: pd.DataFrame
    transfers: pd.DataFrame
    contracts: pd.DataFrame
    injuries: pd.DataFrame
    valuations: pd.DataFrame
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_tables(cls, t: dict[str, pd.DataFrame]) -> "Data":
        t = {k: v.copy() for k, v in t.items()}
        for name, cols in db.DATE_COLUMNS.items():
            for c in cols:
                t[name][c] = pd.to_datetime(t[name][c]).astype("datetime64[ns]")
        tr = t["transfers"]
        tr["fee_eur"] = pd.to_numeric(tr["fee_eur"], errors="coerce")
        tr["fee_confirmed"] = tr["fee_confirmed"].astype(bool)
        return cls(leagues=t["leagues"], clubs=t["clubs"], seasons=t["seasons"], players=t["players"],
                   stats=t["player_statistics"], transfers=tr, contracts=t["contracts"],
                   injuries=t["injuries"], valuations=t["market_valuations"])

    @classmethod
    def from_db(cls, engine: Engine) -> "Data":
        return cls.from_tables(db.read_tables(engine))

    def fee_transfers(self) -> pd.DataFrame:
        """Permanent transfers with a confirmed, positive fixed fee - the only training labels.

        Loans, loan returns, free transfers and undisclosed fees are excluded;
        conditional add-ons are not part of the label.
        """
        t = self.transfers
        return t[(t["type"] == "permanent") & t["fee_confirmed"] & (t["fee_eur"] > 0)]


def _asof(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, by_left: str, by_right: str) -> pd.DataFrame:
    """Latest ``right`` row strictly before ``left_on`` for every left row (order preserved)."""
    left = left.reset_index(drop=True)
    left["_order"] = np.arange(len(left))
    left[left_on] = left[left_on].astype("datetime64[ns]")
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")})
    merged = pd.merge_asof(
        left.sort_values(left_on), right.sort_values(right_on),
        left_on=left_on, right_on=right_on, left_by=by_left, right_by=by_right, allow_exact_matches=False,
    )
    drop = [c for c in ("_order", by_right) if c in merged.columns and c != by_left]
    return merged.sort_values("_order").drop(columns=drop).reset_index(drop=True)


def market_index(data: Data, as_of: pd.Series, days: int = 730) -> np.ndarray:
    """Mean log fee of disclosed permanent transfers in the ``days`` before each date.

    Used to express values relative to the fee level of the time, so that tree
    models do not have to extrapolate transfer-fee inflation.
    """
    fees = data.fee_transfers().sort_values("date")
    dates = fees["date"].to_numpy(dtype="datetime64[ns]")
    cum = np.concatenate([[0.0], np.cumsum(np.log(fees["fee_eur"].to_numpy(dtype=float)))])
    a = as_of.to_numpy(dtype="datetime64[ns]")
    hi = np.searchsorted(dates, a, side="left")
    lo = np.searchsorted(dates, a - np.timedelta64(days, "D"), side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = (cum[hi] - cum[lo]) / n
    fallback = cum[-1] / max(len(fees), 1) if len(fees) else 0.0
    # too few deals in the window (very start of the history): use everything before the date
    few = n < 30
    idx[few] = np.where(hi[few] > 0, cum[hi[few]] / np.maximum(hi[few], 1), fallback)
    return idx


def _season_table(data: Data, adj: LeagueAdjuster) -> pd.DataFrame:
    """Player-season aggregates (stints summed) with rates and league-adjusted index."""
    key = ("season_table", id(adj))
    if key in data._cache:
        return data._cache[key]
    st = data.stats.merge(data.players[["id", "position_group"]], left_on="player_id", right_on="id", suffixes=("", "_p"))
    st = adj.transform(st)
    has = st["idx_raw"].notna()
    st["_m"] = st["minutes"].where(has, 0)
    st["_w_raw"] = (st["idx_raw"] * st["minutes"]).where(has, 0.0)
    st["_w_dif"] = (adj.difficulty(st["league_id"]) * st["minutes"]).where(has, 0.0)
    st["_w_ppg"] = st["team_points_per_game"] * st["team_minutes"]
    agg = st.groupby(["player_id", "season_id"], as_index=False)[
        ["minutes", "team_minutes", "appearances", "_m", "_w_raw", "_w_dif", "_w_ppg"] + COUNT_COLS].sum()
    agg = add_rates(agg)
    m = agg["_m"].where(agg["_m"] > 0)
    shrink = agg["_m"] / (agg["_m"] + 600.0)  # few minutes -> pull towards the position average
    agg["idx_raw"] = agg["_w_raw"] / m * shrink
    agg["idx_adj"] = agg["idx_raw"] + agg["_w_dif"] / m
    agg["minutes_share"] = agg["minutes"] / agg["team_minutes"].where(agg["team_minutes"] > 0)
    agg["team_ppg"] = agg["_w_ppg"] / agg["team_minutes"].where(agg["team_minutes"] > 0)
    agg = agg.merge(data.seasons[["id", "end_date"]].rename(columns={"id": "season_id", "end_date": "season_end"}), on="season_id")
    data._cache[key] = agg
    return agg


def club_context(data: Data, pairs: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time club facts for (club_id, as_of) pairs.

    ppg_last  - points per game in the club's last completed season
    spend_3y / max_fee_3y / income_3y - log1p of confirmed fees paid / received in
    the previous three years (a proxy for purchasing power until club financial
    statements are connected).
    """
    pairs = pairs[["club_id", "as_of"]].drop_duplicates().reset_index(drop=True)
    cs = data.stats.groupby(["club_id", "season_id"], as_index=False)["team_points_per_game"].mean()
    cs = cs.merge(data.seasons[["id", "end_date"]], left_on="season_id", right_on="id")
    out = _asof(pairs, cs[["club_id", "end_date", "team_points_per_game"]].rename(columns={"club_id": "_c"}),
                "as_of", "end_date", "club_id", "_c")
    out = out.rename(columns={"team_points_per_game": "ppg_last"})[["club_id", "as_of", "ppg_last"]]
    fees = data.fee_transfers()
    for side, name in (("to_club_id", "spend"), ("from_club_id", "income")):
        m = pairs.merge(fees[[side, "date", "fee_eur"]], left_on="club_id", right_on=side)
        m = m[(m["date"] < m["as_of"]) & (m["date"] >= m["as_of"] - pd.Timedelta(days=1095))]
        g = m.groupby(["club_id", "as_of"])["fee_eur"].agg(["sum", "max"]).reset_index()
        g = g.rename(columns={"sum": f"{name}_3y", "max": f"{name}_max_3y"})
        out = out.merge(g, on=["club_id", "as_of"], how="left")
    return out.fillna({"spend_3y": 0.0, "spend_max_3y": 0.0, "income_3y": 0.0, "income_max_3y": 0.0})


def build_player_features(data: Data, snaps: pd.DataFrame, adj: LeagueAdjuster) -> pd.DataFrame:
    """Features for (player_id, as_of) snapshots.

    If ``snaps`` has a ``club_id`` column it is used as the player's club (for
    transfers: the selling club); otherwise the club is derived from the
    player's first registered club and the transfers before ``as_of``.
    """
    s = snaps.reset_index(drop=True).copy()
    s["as_of"] = pd.to_datetime(s["as_of"]).astype("datetime64[ns]")
    s["sid"] = np.arange(len(s))
    pl = data.players.rename(columns={"id": "player_id"})
    s = s.merge(pl[["player_id", "birth_date", "position", "position_group", "foot", "nationality", "height_cm"]],
                on="player_id", how="left")
    s["age"] = (s["as_of"] - s["birth_date"]).dt.days / 365.25
    s["market_index"] = market_index(data, s["as_of"])

    # ---- season statistics ---------------------------------------------------
    st = _season_table(data, adj)
    m = s[["sid", "player_id", "as_of"]].merge(st, on="player_id")
    m = m[m["season_end"] < m["as_of"]]
    m["ago"] = (m["as_of"] - m["season_end"]).dt.days // 365
    l1 = m[m["ago"] == 0].set_index("sid")
    l2 = m[m["ago"] == 1].set_index("sid")
    f = pd.DataFrame(index=s["sid"])
    f["l1_minutes"] = l1["minutes"]
    f["l1_minutes_share"] = l1["minutes_share"]
    f["l1_appearances"] = l1["appearances"]
    for c in RATE_COLS:
        f[f"l1_{c}"] = l1[c]
    f["l1_idx_raw"] = l1["idx_raw"]
    f["l1_idx_adj"] = l1["idx_adj"]
    f["l2_minutes"] = l2["minutes"]
    f["l2_idx_adj"] = l2["idx_adj"]
    f["idx_trend"] = f["l1_idx_adj"] - f["l2_idx_adj"]
    # "career" = the last three completed seasons: a fixed window keeps the features
    # comparable between the start of the data history and today
    r = m[m["ago"] <= 2].copy()
    r["_w"] = (r["idx_adj"] * r["_m"]).fillna(0.0)
    car = r.groupby("sid").agg(career_minutes=("minutes", "sum"), career_seasons=("season_id", "nunique"),
                               _w=("_w", "sum"), _m=("_m", "sum"), best_idx_adj=("idx_adj", "max"))
    f["career_minutes"] = car["career_minutes"]
    f["career_seasons"] = car["career_seasons"]
    f["career_idx_adj"] = car["_w"] / car["_m"].where(car["_m"] > 0)
    f["best_idx_adj"] = car["best_idx_adj"]
    f[["l1_minutes", "l2_minutes", "career_minutes", "career_seasons"]] = \
        f[["l1_minutes", "l2_minutes", "career_minutes", "career_seasons"]].fillna(0)
    s = s.join(f, on="sid")

    # ---- club and league at as_of ----------------------------------------------
    if "club_id" not in s.columns:
        first = data.stats.merge(data.seasons[["id", "start_date"]], left_on="season_id", right_on="id")
        first = first.sort_values(["start_date", "stint"]).drop_duplicates("player_id")
        events = pd.concat([
            first[["player_id", "start_date", "club_id"]].rename(columns={"start_date": "date"}),
            data.transfers[["player_id", "date", "to_club_id"]].rename(columns={"to_club_id": "club_id"}),
        ]).dropna(subset=["club_id"]).rename(columns={"player_id": "_p"})
        events["club_id"] = events["club_id"].astype("int64")
        s = _asof(s, events, "as_of", "date", "player_id", "_p").drop(columns=["date"])
    s = s.merge(data.clubs.rename(columns={"id": "club_id", "name": "club"})[["club_id", "club", "league_id"]],
                on="club_id", how="left")
    s = s.merge(data.leagues.rename(columns={"id": "league_id", "name": "league"})[["league_id", "league"]],
                on="league_id", how="left")
    s["league_difficulty"] = adj.difficulty(s["league_id"])
    ctx = club_context(data, s.dropna(subset=["club_id"]))
    s = s.merge(ctx.rename(columns={"ppg_last": "club_ppg_last", "income_3y": "seller_income_3y"})[
        ["club_id", "as_of", "club_ppg_last", "seller_income_3y"]], on=["club_id", "as_of"], how="left")

    # ---- contract ----------------------------------------------------------------
    con = data.contracts[["player_id", "start_date", "end_date"]].rename(columns={"player_id": "_p"})
    s = _asof(s, con, "as_of", "start_date", "player_id", "_p")
    s["contract_months"] = ((s["end_date"] - s["as_of"]).dt.days / 30.44).clip(lower=0)
    s = s.drop(columns=["start_date", "end_date"])

    # ---- injuries ----------------------------------------------------------------
    inj = s[["sid", "player_id", "as_of"]].merge(data.injuries[["player_id", "start_date", "end_date"]], on="player_id")
    inj = inj[(inj["start_date"] < inj["as_of"]) & (inj["start_date"] >= inj["as_of"] - pd.Timedelta(days=730))]
    inj["days"] = (inj[["end_date", "as_of"]].min(axis=1) - inj["start_date"]).dt.days
    recent = inj["start_date"] >= inj["as_of"] - pd.Timedelta(days=365)
    g = pd.DataFrame({
        "inj_days_365": inj[recent].groupby("sid")["days"].sum(),
        "inj_days_730": inj.groupby("sid")["days"].sum(),
        "inj_count_730": inj.groupby("sid")["days"].size(),
    })
    s = s.join(g, on="sid")
    s[["inj_days_365", "inj_days_730", "inj_count_730"]] = s[["inj_days_365", "inj_days_730", "inj_count_730"]].fillna(0)

    # ---- transfer history ----------------------------------------------------------
    tr = s[["sid", "player_id", "as_of", "market_index"]].merge(
        data.transfers[["player_id", "date", "type", "fee_eur", "fee_confirmed"]], on="player_id")
    tr = tr[(tr["date"] < tr["as_of"]) & (tr["date"] >= tr["as_of"] - pd.Timedelta(days=HISTORY_DAYS))]
    paid = tr[(tr["type"] == "permanent") & tr["fee_confirmed"] & (tr["fee_eur"] > 0)].copy()
    # earlier fees relative to today's fee level (log scale)
    paid["rel"] = np.log(paid["fee_eur"]) - paid["market_index"]
    g = pd.DataFrame({
        "n_prev_transfers": tr[tr["type"].isin(["permanent", "free"])].groupby("sid").size(),
        "n_prev_loans": tr[tr["type"] == "loan"].groupby("sid").size(),
        "last_fee_rel": paid.sort_values("date").groupby("sid")["rel"].last(),
        "max_fee_rel": paid.groupby("sid")["rel"].max(),
    })
    s = s.join(g, on="sid")
    s[["n_prev_transfers", "n_prev_loans"]] = s[["n_prev_transfers", "n_prev_loans"]].fillna(0)

    # ---- external reference valuation (benchmark only, never a model input) ---------
    val = data.valuations[["player_id", "date", "value_eur"]].rename(
        columns={"player_id": "_p", "value_eur": "ref_value_eur", "date": "ref_date"})
    s = _asof(s, val, "as_of", "ref_date", "player_id", "_p")
    s.loc[(s["as_of"] - s["ref_date"]).dt.days > 400, "ref_value_eur"] = np.nan

    for c in PLAYER_CAT:
        s[c] = s[c].fillna("unknown").astype(str)
    return s.drop(columns=["birth_date", "ref_date"]).sort_values("sid").reset_index(drop=True)


def add_deal_features(data: Data, feats: pd.DataFrame, adj: LeagueAdjuster) -> pd.DataFrame:
    """Transaction context for the Transfer Fee Engine. Needs ``buyer_club_id``."""
    s = feats.reset_index(drop=True).copy()
    buyer = data.clubs.rename(columns={"id": "buyer_club_id", "league_id": "buyer_league_id"})[["buyer_club_id", "buyer_league_id"]]
    s = s.merge(buyer, on="buyer_club_id", how="left")
    s = s.merge(data.leagues.rename(columns={"id": "buyer_league_id", "name": "buyer_league"})[["buyer_league_id", "buyer_league"]],
                on="buyer_league_id", how="left")
    s["buyer_league"] = s["buyer_league"].fillna("unknown").astype(str)
    s["buyer_league_difficulty"] = adj.difficulty(s["buyer_league_id"])
    s["league_step"] = s["buyer_league_difficulty"] - s["league_difficulty"]
    s["cross_league"] = (s["buyer_league_id"] != s["league_id"]).astype(float)
    ctx = club_context(data, s[["buyer_club_id", "as_of"]].rename(columns={"buyer_club_id": "club_id"}))
    ctx = ctx.rename(columns={"club_id": "buyer_club_id", "ppg_last": "buyer_ppg_last"})
    s = s.merge(ctx[["buyer_club_id", "as_of", "buyer_ppg_last", "spend_3y", "spend_max_3y"]], on=["buyer_club_id", "as_of"], how="left")
    # money relative to the fee level of the time
    s["buyer_spend_3y_rel"] = np.log1p(s["spend_3y"].fillna(0)) - s["market_index"]
    s["buyer_max_fee_3y_rel"] = np.log1p(s["spend_max_3y"].fillna(0)) - s["market_index"]
    s["seller_income_3y_rel"] = np.log1p(s["seller_income_3y"].fillna(0)) - s["market_index"]
    deal = s["deal_date"] if "deal_date" in s.columns else s["as_of"] + pd.Timedelta(days=1)
    deal = pd.to_datetime(deal)
    summer = deal.dt.month.between(6, 9)
    s["window"] = np.where(summer, "summer", "winter")
    close = pd.to_datetime(np.where(
        summer, deal.dt.year.astype(str) + "-09-01",
        np.where(deal.dt.month >= 10, (deal.dt.year + 1).astype(str), deal.dt.year.astype(str)) + "-02-01"))
    s["days_to_window_close"] = (close - deal).dt.days.clip(lower=0)
    return s.drop(columns=["spend_3y", "spend_max_3y"])


def data_quality(feats: pd.DataFrame) -> pd.Series:
    """0..1 score: how much of the evidence the valuation relies on is actually there."""
    parts = [
        (feats["l1_minutes"] >= 900) * 0.35 + ((feats["l1_minutes"] >= 300) & (feats["l1_minutes"] < 900)) * 0.15,
        (feats["l2_minutes"] >= 900) * 0.20,
        feats["contract_months"].notna() * 0.20,
        (feats["career_minutes"] >= 3000) * 0.15,
        feats["club_ppg_last"].notna() * 0.10,
    ]
    return sum(parts).astype(float).round(2)
