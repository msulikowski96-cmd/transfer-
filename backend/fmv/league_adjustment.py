"""Model D - League Adjustment Engine.

Makes performances comparable across leagues. 20 goals in a weaker league are
not the same evidence of quality as 20 goals in a stronger one.

Method
------
1. Every player-season stint gets a raw performance index: a position-group
   specific weighted sum of per-90 statistics, standardised inside the position
   group (z-score).
2. League difficulty is estimated from *the same players in different leagues*.
   For consecutive seasons of one player:

       z[t+1] - z[t] = D[league_t] - D[league_t+1] + ageing(age) + noise

   Players who stay give the ageing curve, players who move identify the
   difference D between two leagues. Solved with ridge regression.
3. Adjusted index = raw index + D[league].

The engine is fitted only on seasons that ended before the training cut-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MINUTES = 900

# weights of per-90 statistics inside the raw index
_WEIGHTS = {
    "FWD": {"xg90": 0.6, "goals90": 0.4, "xa90": 0.5, "assists90": 0.2, "shots90": 0.03, "key_passes90": 0.05},
    "MID": {"xg90": 0.4, "goals90": 0.3, "xa90": 0.5, "assists90": 0.2, "key_passes90": 0.06,
            "progressive_passes90": 0.02, "def_actions90": 0.03},
    "DEF": {"def_actions90": 0.05, "aerials_won90": 0.04, "progressive_passes90": 0.015, "xg90": 0.3, "xa90": 0.4},
    "GK": {"save_pct": 3.0, "clean_sheet_rate": 0.3},
}
COUNT_COLS = ["goals", "assists", "xg", "xa", "shots", "key_passes", "progressive_passes",
              "tackles", "interceptions", "aerials_won", "saves", "goals_conceded", "clean_sheets"]


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates from counting statistics (expects ``minutes`` and COUNT_COLS)."""
    out = df.copy()
    m90 = (out["minutes"] / 90).where(out["minutes"] > 0)
    for c in ["goals", "assists", "xg", "xa", "shots", "key_passes", "progressive_passes", "aerials_won"]:
        out[f"{c}90"] = out[c] / m90
    out["def_actions90"] = (out["tackles"] + out["interceptions"]) / m90
    faced = out["saves"] + out["goals_conceded"]
    out["save_pct"] = (out["saves"] / faced.where(faced > 0)).astype(float)
    out["clean_sheet_rate"] = out["clean_sheets"] / m90
    return out


def raw_index(df: pd.DataFrame) -> pd.Series:
    """Un-standardised performance index; needs rate columns and ``position_group``."""
    idx = pd.Series(np.nan, index=df.index, dtype=float)
    for group, weights in _WEIGHTS.items():
        mask = df["position_group"] == group
        if mask.any():
            idx[mask] = sum(w * df.loc[mask, col].astype(float) for col, w in weights.items())
    return idx


class LeagueAdjuster:
    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge
        self.norm_: dict[str, tuple[float, float]] = {}
        self.difficulty_: dict[int, float] = {}
        self.n_movers_: int = 0

    # -- fitting ---------------------------------------------------------------
    def fit(self, stats: pd.DataFrame, players: pd.DataFrame, seasons: pd.DataFrame) -> "LeagueAdjuster":
        """``stats`` must already be restricted to seasons known at training time."""
        df = stats.merge(players[["id", "position_group", "birth_date"]], left_on="player_id", right_on="id",
                         suffixes=("", "_p"))
        df = add_rates(df)
        df["raw"] = raw_index(df)
        solid = df[(df["minutes"] >= MIN_MINUTES) & df["raw"].notna()]
        for group, g in solid.groupby("position_group"):
            self.norm_[group] = (float(g["raw"].mean()), float(g["raw"].std() or 1.0))
        solid = solid.assign(z=self._standardise(solid))
        # one league per player-season: the stint with most minutes
        solid = solid.sort_values("minutes").drop_duplicates(["player_id", "season_id"], keep="last")
        nxt = solid[["player_id", "season_id", "league_id", "z"]].copy()
        nxt["season_id"] -= 1
        pairs = solid.merge(nxt, on=["player_id", "season_id"], suffixes=("", "_next"))
        start = pairs["season_id"].map(seasons.set_index("id")["start_date"])
        age = (pd.to_datetime(start) - pd.to_datetime(pairs["birth_date"])).dt.days / 365.25

        league_ids = sorted(stats["league_id"].unique())
        col = {lid: i for i, lid in enumerate(league_ids)}
        X = np.zeros((len(pairs), len(league_ids) + 3))
        rows = np.arange(len(pairs))
        moved = (pairs["league_id"] != pairs["league_id_next"]).to_numpy()
        np.add.at(X, (rows, pairs["league_id"].map(col).to_numpy()), 1.0)
        np.add.at(X, (rows, pairs["league_id_next"].map(col).to_numpy()), -1.0)
        a = (age.to_numpy() - 26) / 5
        X[:, -3], X[:, -2], X[:, -1] = 1.0, a, a ** 2
        y = (pairs["z_next"] - pairs["z"]).to_numpy()
        penalty = np.eye(X.shape[1]) * self.ridge
        penalty[-3:, -3:] = 0.0  # do not shrink the ageing curve
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ y) if len(pairs) else np.zeros(X.shape[1])
        d = beta[: len(league_ids)]
        d = d - d.mean()
        self.difficulty_ = {int(lid): float(d[i]) for lid, i in col.items()}
        self.n_movers_ = int(moved.sum())
        return self

    def _standardise(self, df: pd.DataFrame) -> pd.Series:
        mean = df["position_group"].map(lambda g: self.norm_.get(g, (0.0, 1.0))[0])
        std = df["position_group"].map(lambda g: self.norm_.get(g, (0.0, 1.0))[1])
        return (df["raw"] - mean) / std

    # -- applying --------------------------------------------------------------
    def difficulty(self, league_id: pd.Series) -> pd.Series:
        return league_id.map(self.difficulty_).astype(float).fillna(0.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds rate columns, ``idx_raw`` and ``idx_adj``. Needs ``position_group`` and ``league_id``."""
        out = add_rates(df)
        out["raw"] = raw_index(out)
        out["idx_raw"] = self._standardise(out)
        out["idx_adj"] = out["idx_raw"] + self.difficulty(out["league_id"])
        return out.drop(columns="raw")
