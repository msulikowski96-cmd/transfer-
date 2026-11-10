"""Valuation engines.

Model A  MarketValueEngine   CatBoost (+ LightGBM comparison model) on player-only
                             features, blended with weights fitted on validation data.
Model B  TransferFeeEngine   XGBoost multi-quantile regression on player + deal features.
Model C  PotentialEngine     LightGBM quantile models for the 12 and 24 month change
                             of the market value.

All models work on ``log(fee) - market_index`` (the value relative to the fee
level of the time). Intervals are conformal: Model A uses a locally adaptive
split-conformal interval around the ensemble, Models B and C use conformalised
quantile regression (raw q10/q90 shifted by a constant fitted on held-out data),
so that the nominal coverage holds on unseen data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import nnls
from xgboost import XGBRegressor

from . import config
from .features import DEAL_CAT, DEAL_NUM, PLAYER_CAT, PLAYER_NUM

ALPHA = (1 - config.INTERVAL_COVERAGE) / 2  # 0.10
QUANTILES = [ALPHA, 0.5, 1 - ALPHA]


class _Encoder:
    """Stable pandas categoricals so LightGBM / XGBoost see the same codes at predict time."""

    def __init__(self, num: list[str], cat: list[str]):
        self.num, self.cat = list(num), list(cat)
        self.categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "_Encoder":
        self.categories = {c: sorted(df[c].astype(str).unique()) for c in self.cat}
        return self

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.num].astype(float).copy()
        for c in self.cat:
            out[c] = pd.Categorical(df[c].astype(str), categories=self.categories[c])
        return out

    def catboost(self, df: pd.DataFrame, y=None) -> Pool:
        x = df[self.num + self.cat].copy()
        x[self.num] = x[self.num].astype(float)
        x[self.cat] = x[self.cat].astype(str)
        return Pool(x, y, cat_features=self.cat)


def _conformal_offset(y, low, high, alpha: float = 2 * ALPHA) -> float:
    """CQR correction: the (1-alpha) empirical quantile of the conformity scores."""
    score = np.maximum(low - y, y - high)
    n = len(score)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(score, level))


def target(df: pd.DataFrame) -> np.ndarray:
    return (np.log(df["fee_eur"].astype(float)) - df["market_index"]).to_numpy()


class MarketValueEngine:
    """Value under standard market conditions: no information about the buyer or the deal."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.enc = _Encoder(PLAYER_NUM, PLAYER_CAT)
        self.weights = {"catboost": 0.5, "lightgbm": 0.5}
        self.offset = 0.0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "MarketValueEngine":
        self.enc.fit(pd.concat([train, val]))
        y_tr, y_va = target(train), target(val)
        self.cat = CatBoostRegressor(iterations=3000, learning_rate=0.04, depth=6, loss_function="RMSE",
                                     random_seed=self.seed, verbose=False, od_type="Iter", od_wait=150,
                                     allow_writing_files=False, thread_count=4)
        self.cat.fit(self.enc.catboost(train, y_tr), eval_set=self.enc.catboost(val, y_va))
        x_tr, x_va = self.enc.frame(train), self.enc.frame(val)
        lgb_params = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, cat_smooth=20, random_state=self.seed, verbose=-1, n_jobs=4)
        callbacks = [early_stopping(150, verbose=False), log_evaluation(0)]
        self.lgb = LGBMRegressor(**lgb_params).fit(x_tr, y_tr, eval_X=x_va, eval_y=y_va, callbacks=callbacks)
        # ensemble weights: non-negative least squares on validation predictions
        P = np.column_stack([self.cat.predict(self.enc.catboost(val)), self.lgb.predict(x_va)])
        w, _ = nnls(P, y_va)
        w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
        self.weights = {"catboost": float(w[0]), "lightgbm": float(w[1])}
        # interval: locally adaptive split-conformal. A scale model learns how large the
        # error typically is for a given kind of player (from out-of-fold residuals on the
        # training set); validation data then fixes the multiplier that gives the coverage.
        fixed = {**lgb_params, "n_estimators": max(50, int(self.lgb.best_iteration_ or 300))}
        oof = np.zeros(len(train))
        fold = np.random.default_rng(self.seed).integers(0, 4, len(train))
        for k in range(4):
            m = fold == k
            oof[m] = LGBMRegressor(**fixed).fit(x_tr[~m], y_tr[~m]).predict(x_tr[m])
        self.scale = LGBMRegressor(**{**lgb_params, "n_estimators": 200, "min_child_samples": 50, "num_leaves": 15}
                                   ).fit(x_tr, np.abs(y_tr - oof))
        self.scale_floor = float(np.median(np.abs(y_tr - oof))) * 0.5
        self.oof_ = oof  # honest (out-of-fold) valuations of the training deals, used by Model B
        self.sigma_cuts = [float(c) for c in np.quantile(self._sigma(x_tr), [1 / 3, 2 / 3])]
        resid = np.abs(y_va - P @ w) / self._sigma(x_va)
        n = len(resid)
        self.offset = float(np.quantile(resid, min(1.0, np.ceil((n + 1) * config.INTERVAL_COVERAGE) / n)))
        return self

    def _sigma(self, x: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.scale.predict(x), self.scale_floor)

    def uncertainty_level(self, df: pd.DataFrame) -> np.ndarray:
        """low / medium / high: the expected error for this kind of player, in thirds of the training deals."""
        sigma = self._sigma(self.enc.frame(df))
        return np.where(sigma <= self.sigma_cuts[0], "low", np.where(sigma <= self.sigma_cuts[1], "medium", "high"))

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"catboost": self.cat.predict(self.enc.catboost(df)),
                             "lightgbm": self.lgb.predict(self.enc.frame(df))}, index=df.index)

    def predict_rel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log value relative to the market index: columns rel, rel_low, rel_high."""
        comp = self.components(df)
        rel = comp["catboost"] * self.weights["catboost"] + comp["lightgbm"] * self.weights["lightgbm"]
        half = self.offset * self._sigma(self.enc.frame(df))
        return pd.DataFrame({"rel": rel, "rel_low": rel - half, "rel_high": rel + half}, index=df.index)

    def shap(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values of the CatBoost component (log scale); last column is the base value."""
        values = self.cat.get_feature_importance(self.enc.catboost(df), type="ShapValues")
        return pd.DataFrame(values, columns=self.enc.num + self.enc.cat + ["_base"], index=df.index)


class TransferFeeEngine:
    """Fee expected in one specific transaction (given buyer, seller, window, contract).

    Modelled as a premium or discount to the market value: the target is
    ``log(fee) - log(market value)`` and the quantiles of that premium are learned
    from the deal context. Input frames need a ``mv_rel`` column (Model A output;
    out-of-fold for the training rows so the premium is not learned on over-fitted values).
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.enc = _Encoder(PLAYER_NUM + DEAL_NUM + ["mv_rel"], PLAYER_CAT + DEAL_CAT)
        self.offset = 0.0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "TransferFeeEngine":
        self.enc.fit(pd.concat([train, val]))
        y_tr, y_va = target(train) - train["mv_rel"].to_numpy(), target(val) - val["mv_rel"].to_numpy()
        self.xgb = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES), tree_method="hist",
                                enable_categorical=True, max_cat_to_onehot=8, n_estimators=2000, learning_rate=0.03, max_depth=4,
                                min_child_weight=10, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=100,
                                random_state=self.seed, n_jobs=4)
        self.xgb.fit(self.enc.frame(train), y_tr, eval_set=[(self.enc.frame(val), y_va)], verbose=False)
        q = self._raw(val)
        self.offset = _conformal_offset(y_va, q[:, 0], q[:, 2])
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        return np.sort(self.xgb.predict(self.enc.frame(df)).reshape(len(df), 3), axis=1)

    def predict_rel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log fee relative to the market index: columns rel, rel_low, rel_high."""
        q = self._raw(df) + df["mv_rel"].to_numpy()[:, None]
        return pd.DataFrame({"rel": q[:, 1], "rel_low": np.minimum(q[:, 0] - self.offset, q[:, 1]),
                             "rel_high": np.maximum(q[:, 2] + self.offset, q[:, 1])}, index=df.index)


class DriftCorrector:
    """Online recalibration against market drift.

    Keeps the log residuals of every labelled deal. The correction for a
    valuation date is the (shrunk) mean residual of the deals closed in the
    ``days`` before that date - information that was public at the time. It
    absorbs fee inflation and shifts in the mix of deals that the trailing
    market index does not capture.
    """

    def __init__(self, days: int = 365, prior: float = 30.0):
        self.days, self.prior = days, prior
        self.dates = np.array([], dtype="datetime64[ns]")
        self.cum = np.array([0.0])

    def fit(self, deal_dates: pd.Series, residuals: np.ndarray) -> "DriftCorrector":
        order = np.argsort(deal_dates.to_numpy(dtype="datetime64[ns]"))
        self.dates = deal_dates.to_numpy(dtype="datetime64[ns]")[order]
        self.cum = np.concatenate([[0.0], np.cumsum(np.asarray(residuals, dtype=float)[order])])
        return self

    def at(self, as_of: pd.Series) -> np.ndarray:
        a = as_of.to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(self.dates, a, side="left")
        lo = np.searchsorted(self.dates, a - np.timedelta64(self.days, "D"), side="left")
        return (self.cum[hi] - self.cum[lo]) / np.maximum(hi - lo + self.prior, 1e-9)


class PotentialEngine:
    """Change of the (relative, log) market value over 12 and 24 months.

    Trained on yearly snapshots of the same player: features at t, target is the
    Market Value Engine's valuation at t+h minus its valuation at t. This learns
    development and ageing curves from how players' evidence actually evolved.
    """

    HORIZONS = (12, 24)
    FEATURES = PLAYER_NUM + ["mv_rel"]

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.enc = _Encoder(self.FEATURES, PLAYER_CAT)
        self.models: dict[tuple[int, float], LGBMRegressor] = {}
        self.offsets = {h: 0.0 for h in self.HORIZONS}

    def fit(self, panel: pd.DataFrame) -> "PotentialEngine":
        """``panel`` has the features plus ``delta_12`` / ``delta_24`` (NaN when unknown)."""
        self.enc.fit(panel)
        for h in self.HORIZONS:
            d = panel[panel[f"delta_{h}"].notna()]
            x, y = self.enc.frame(d), d[f"delta_{h}"].to_numpy()
            for q in QUANTILES:
                self.models[(h, q)] = LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, learning_rate=0.04,
                                                    num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, random_state=self.seed, verbose=-1, n_jobs=4).fit(x, y)
        return self

    def calibrate(self, panel: pd.DataFrame) -> "PotentialEngine":
        """Conformal correction of the bands on snapshots not used in ``fit``."""
        self.offsets = {h: 0.0 for h in self.HORIZONS}
        raw = self.predict(panel)
        for h in self.HORIZONS:
            m = panel[f"delta_{h}"].notna().to_numpy()
            if m.sum() >= 50:
                self.offsets[h] = _conformal_offset(panel.loc[m, f"delta_{h}"].to_numpy(),
                                                    raw.loc[m, f"delta_{h}_low"].to_numpy(), raw.loc[m, f"delta_{h}_high"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columns delta_{h}_{low|mid|high} on the log scale."""
        x = self.enc.frame(df)
        out = {}
        for h in self.HORIZONS:
            q = np.sort(np.column_stack([self.models[(h, a)].predict(x) for a in QUANTILES]), axis=1)
            q[:, 0] = np.minimum(q[:, 0] - self.offsets[h], q[:, 1])
            q[:, 2] = np.maximum(q[:, 2] + self.offsets[h], q[:, 1])
            out[f"delta_{h}_low"], out[f"delta_{h}_mid"], out[f"delta_{h}_high"] = q[:, 0], q[:, 1], q[:, 2]
        return pd.DataFrame(out, index=df.index)
