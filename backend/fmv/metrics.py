"""Evaluation metrics (section 8 of the brief). All errors are in euros."""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_BUCKETS = [0, 1e6, 5e6, 20e6, np.inf]
BUCKET_LABELS = ["<1M", "1-5M", "5-20M", ">20M"]


def regression_metrics(actual, predicted, low=None, high=None) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    err = p - a
    out = {
        "n": int(len(a)),
        "mae_eur": float(np.mean(np.abs(err))),
        "rmse_eur": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / np.sum(a)),
        "median_ape": float(np.median(np.abs(err) / a)),
        "bias_log": float(np.mean(np.log(p) - np.log(a))),
    }
    if low is not None and high is not None:
        lo, hi = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        out["interval_coverage"] = float(np.mean((a >= lo) & (a <= hi)))
        out["interval_rel_width"] = float(np.median((hi - lo) / p))
    return out


def breakdown(df: pd.DataFrame, actual: str, predicted: str, low: str | None = None, high: str | None = None) -> dict:
    """Overall metrics plus the slices the brief asks for: position, league, value band."""
    def m(g: pd.DataFrame) -> dict:
        return regression_metrics(g[actual], g[predicted], g[low] if low else None, g[high] if high else None)

    bucket = pd.cut(df[actual], VALUE_BUCKETS, labels=BUCKET_LABELS, right=False)
    return {
        "overall": m(df),
        "by_position_group": {str(k): m(g) for k, g in df.groupby("position_group")},
        "by_league": {str(k): m(g) for k, g in df.groupby("league")},
        "by_value_bucket": {str(k): m(g) for k, g in df.groupby(bucket, observed=True)},
    }
