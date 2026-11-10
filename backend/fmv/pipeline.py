"""Training, evaluation and batch valuation."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import JSON, delete, insert, select
from sqlalchemy.engine import Engine

from . import config, db
from .engines import DriftCorrector, MarketValueEngine, PotentialEngine, TransferFeeEngine, target
from .features import FACTOR_GROUPS, Data, add_deal_features, build_player_features, data_quality
from .league_adjustment import LeagueAdjuster
from .metrics import breakdown, regression_metrics


WARMUP_YEARS = 3


class Valuer:
    """The trained engines behind one interface that returns euros."""

    def __init__(self, models: dict):
        self.adj: LeagueAdjuster = models["adjuster"]
        self.mv: MarketValueEngine = models["market_value"]
        self.tf: TransferFeeEngine = models["transfer_fee"]
        self.pot: PotentialEngine = models["potential"]
        self.drift_mv: DriftCorrector = models["drift_market_value"]
        self.drift_tf: DriftCorrector = models["drift_transfer_fee"]

    def market(self, feats: pd.DataFrame, drift: bool = True) -> pd.DataFrame:
        """market_value_{,low_,high_}eur plus ``mv_rel`` (raw log value relative to the market index)."""
        rel = self.mv.predict_rel(feats)
        shift = feats["market_index"].to_numpy() + (self.drift_mv.at(feats["as_of"]) if drift else 0.0)
        return pd.DataFrame({"market_value_eur": np.exp(rel["rel"] + shift), "market_value_low_eur": np.exp(rel["rel_low"] + shift),
                             "market_value_high_eur": np.exp(rel["rel_high"] + shift), "mv_rel": rel["rel"]}, index=feats.index)

    def transfer(self, deal_feats: pd.DataFrame, drift: bool = True) -> pd.DataFrame:
        """transfer_value_{,low_,high_}eur; ``deal_feats`` needs deal features and ``mv_rel``."""
        rel = self.tf.predict_rel(deal_feats)
        shift = deal_feats["market_index"].to_numpy() + (self.drift_tf.at(deal_feats["as_of"]) if drift else 0.0)
        return pd.DataFrame({"transfer_value_eur": np.exp(rel["rel"] + shift), "transfer_value_low_eur": np.exp(rel["rel_low"] + shift),
                             "transfer_value_high_eur": np.exp(rel["rel_high"] + shift)}, index=deal_feats.index)


# --------------------------------------------------------------------------- data sets
def fit_adjuster(data: Data, train_end: pd.Timestamp) -> LeagueAdjuster:
    known = data.seasons.loc[data.seasons["end_date"] < train_end, "id"]
    return LeagueAdjuster().fit(data.stats[data.stats["season_id"].isin(known)], data.players, data.seasons)


def deal_frame(data: Data, adj: LeagueAdjuster) -> pd.DataFrame:
    """One row per labelled transfer, features as of the day before the deal."""
    t = data.fee_transfers()
    snaps = pd.DataFrame({
        "transfer_id": t["id"].to_numpy(), "player_id": t["player_id"].to_numpy(),
        "as_of": (t["date"] - pd.Timedelta(days=1)).to_numpy(), "deal_date": t["date"].to_numpy(),
        "club_id": t["from_club_id"].to_numpy(), "buyer_club_id": t["to_club_id"].to_numpy(),
        "fee_eur": t["fee_eur"].to_numpy(),
    })
    return add_deal_features(data, build_player_features(data, snaps, adj), adj)


def snapshot_dates(data: Data) -> list[pd.Timestamp]:
    return [pd.Timestamp(d.year, config.SNAPSHOT_MONTH, config.SNAPSHOT_DAY) for d in sorted(data.seasons["end_date"])]


def panel_frame(data: Data, adj: LeagueAdjuster, dates: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Yearly snapshots of every player who played in the season that just ended."""
    dates = dates or snapshot_dates(data)
    ends = data.seasons.set_index("id")["end_date"]
    played = data.stats.assign(end=data.stats["season_id"].map(ends))
    frames = []
    for d in dates:
        last = played[(played["end"] < d) & (played["end"] >= d - pd.Timedelta(days=365))]
        frames.append(pd.DataFrame({"player_id": sorted(last["player_id"].unique()), "as_of": d}))
    return build_player_features(data, pd.concat(frames, ignore_index=True), adj)


def _with_deltas(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    key = panel.set_index(["player_id", "as_of"])["mv_rel"]
    for h in PotentialEngine.HORIZONS:
        future = pd.MultiIndex.from_arrays([panel["player_id"], panel["as_of"] + pd.DateOffset(months=h)])
        out[f"delta_{h}"] = key.reindex(future).to_numpy() - panel["mv_rel"].to_numpy()
    return out


# --------------------------------------------------------------------------- training
def train(engine: Engine, name: str | None = None, artifacts_dir: Path | None = None,
          train_end: pd.Timestamp | None = None, val_end: pd.Timestamp | None = None, seed: int = 0) -> dict:
    train_end = train_end or config.TRAIN_END
    val_end = val_end or config.VAL_END
    data = Data.from_db(engine)
    adj = fit_adjuster(data, train_end)
    deals = deal_frame(data, adj)
    # the first seasons of the history only warm up the three-season feature windows
    warm = data.seasons["start_date"].min() + pd.DateOffset(years=WARMUP_YEARS)
    deals = deals[deals["deal_date"] >= warm]
    tr = deals[deals["deal_date"] < train_end]
    va = deals[(deals["deal_date"] >= train_end) & (deals["deal_date"] < val_end)]
    te = deals[deals["deal_date"] >= val_end].copy()
    if min(len(tr), len(va), len(te)) < 50:
        raise ValueError(f"not enough labelled transfers: train={len(tr)} val={len(va)} test={len(te)}")

    mv = MarketValueEngine(seed).fit(tr, va)
    deals["mv_rel"] = mv.predict_rel(deals)["rel"].to_numpy()
    tr = tr.assign(mv_rel=mv.oof_)
    va, te = deals.loc[va.index], deals.loc[te.index].copy()
    tf = TransferFeeEngine(seed).fit(tr, va)
    # drift correctors see every labelled deal; at() only ever uses deals before the valuation date
    y_all = target(deals)
    models = {"adjuster": adj, "market_value": mv, "transfer_fee": tf, "potential": None,
              "drift_market_value": DriftCorrector().fit(deals["deal_date"], y_all - deals["mv_rel"].to_numpy()),
              "drift_transfer_fee": DriftCorrector().fit(deals["deal_date"], y_all - tf.predict_rel(deals)["rel"].to_numpy())}
    valuer = Valuer(models)

    # ---- evaluation on transfers the models have never seen ---------------------
    raw = te.join(valuer.market(te, drift=False).drop(columns="mv_rel")).join(valuer.transfer(te, drift=False))
    te = te.join(valuer.market(te).drop(columns="mv_rel")).join(valuer.transfer(te))
    comp = mv.components(te)
    idx = te["market_index"]
    naive_rel = pd.Series(target(tr), index=tr.index).groupby(tr["position_group"]).median()
    te["naive_eur"] = np.exp(te["position_group"].map(naive_rel).fillna(naive_rel.median()) + idx)
    ref = te[te["ref_value_eur"] > 0]
    report = {
        "split": {"train_end": str(train_end.date()), "val_end": str(val_end.date()),
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "test_from": str(te["deal_date"].min().date()), "test_to": str(te["deal_date"].max().date())},
        "market_value": breakdown(te, "fee_eur", "market_value_eur", "market_value_low_eur", "market_value_high_eur"),
        "transfer_value": breakdown(te, "fee_eur", "transfer_value_eur", "transfer_value_low_eur", "transfer_value_high_eur"),
        "without_drift_correction": {
            "market_value": regression_metrics(raw["fee_eur"], raw["market_value_eur"], raw["market_value_low_eur"], raw["market_value_high_eur"]),
            "transfer_value": regression_metrics(raw["fee_eur"], raw["transfer_value_eur"], raw["transfer_value_low_eur"], raw["transfer_value_high_eur"]),
        },
        "components": {
            "catboost": regression_metrics(te["fee_eur"], np.exp(comp["catboost"] + idx)),
            "lightgbm": regression_metrics(te["fee_eur"], np.exp(comp["lightgbm"] + idx)),
        },
        "baselines": {
            "position_median": regression_metrics(te["fee_eur"], te["naive_eur"]),
            "reference_valuation": regression_metrics(ref["fee_eur"], ref["ref_value_eur"]),
            "market_value_same_rows_as_reference": regression_metrics(ref["fee_eur"], ref["market_value_eur"]),
        },
        "ensemble_weights": mv.weights,
        "conformal_offset_log": {"market_value": mv.offset, "transfer_value": tf.offset},
        "league_difficulty": {str(data.leagues.set_index("id")["name"][k]): round(v, 3) for k, v in adj.difficulty_.items()},
        "league_movers_used": adj.n_movers_,
    }

    # ---- potential -----------------------------------------------------------------
    panel = panel_frame(data, adj)
    panel["mv_rel"] = mv.predict_rel(panel)["rel"].to_numpy()  # real terms: no index, no drift
    panel = _with_deltas(panel)
    panel = panel[panel["as_of"] >= warm]
    fit_panel, cal_panel = panel.copy(), panel.copy()
    for h in PotentialEngine.HORIZONS:
        due = panel["as_of"] + pd.DateOffset(months=h)
        fit_panel.loc[due > train_end, f"delta_{h}"] = np.nan
        cal_panel.loc[(due <= train_end) | (due > val_end), f"delta_{h}"] = np.nan
    pot = PotentialEngine(seed).fit(fit_panel).calibrate(cal_panel)
    pred = pot.predict(panel)
    report["potential"] = {}
    for h in PotentialEngine.HORIZONS:
        m = panel[f"delta_{h}"].notna() & (panel["as_of"] + pd.DateOffset(months=h) > val_end)
        y = panel.loc[m, f"delta_{h}"]
        p = pred.loc[m]
        report["potential"][f"{h}m"] = {
            "n": int(m.sum()),
            "mae_log_model": float((p[f"delta_{h}_mid"] - y).abs().mean()),
            "mae_log_no_change_baseline": float(y.abs().mean()),
            "interval_coverage": float(((y >= p[f"delta_{h}_low"]) & (y <= p[f"delta_{h}_high"])).mean()),
        } if m.any() else {"n": 0}

    # ---- persist --------------------------------------------------------------------
    name = name or f"fmv-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = Path(artifacts_dir or config.ARTIFACTS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as fh:
        pickle.dump({**models, "potential": pot}, fh)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    params = {"seed": seed, "interval_coverage": config.INTERVAL_COVERAGE,
              "market_value": "CatBoost + LightGBM, NNLS blend, locally adaptive conformal interval",
              "transfer_value": "XGBoost multi-quantile premium over market value, CQR interval",
              "drift_correction": "trailing 365-day mean residual of closed deals",
              "potential": "LightGBM quantile, horizons 12/24 months"}
    db.create_schema(engine)
    with engine.begin() as conn:
        version_id = conn.execute(insert(db.ModelVersion).values(
            name=name, created_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), train_end=train_end.date(),
            val_end=val_end.date(), params=params, metrics=report, artifact_path=str(out_dir))).inserted_primary_key[0]
    return {"version_id": version_id, "name": name, "artifact_path": str(out_dir), "report": report}


def load_models(engine: Engine, version: str | None = None) -> tuple[dict, dict]:
    """Returns (model_versions row, unpickled models). Latest version by default."""
    q = select(db.ModelVersion).order_by(db.ModelVersion.id.desc())
    if version:
        q = q.where(db.ModelVersion.name == version)
    with engine.connect() as conn:
        row = conn.execute(q).mappings().first()
    if row is None:
        raise LookupError("no trained model version found - run `fmv train` first")
    with open(Path(row["artifact_path"]) / "model.pkl", "rb") as fh:
        return dict(row), pickle.load(fh)


# --------------------------------------------------------------------------- valuation
def explain(mv: MarketValueEngine, feats: pd.DataFrame) -> list[list[dict]]:
    """SHAP contributions of the CatBoost component, summed into readable factor groups."""
    shap = mv.shap(feats)
    grouped = pd.DataFrame({g: shap[cols].sum(axis=1) for g, cols in FACTOR_GROUPS.items()})
    out = []
    for values in grouped.to_numpy():
        factors = [{"factor": g, "effect_pct": round(float(np.expm1(v)) * 100, 1),
                    "direction": "positive" if v > 0.03 else "negative" if v < -0.03 else "neutral"}
                   for g, v in zip(grouped.columns, values)]
        out.append(sorted(factors, key=lambda f: -abs(f["effect_pct"])))
    return out


def _eur(v: float) -> str:
    return f"€{v / 1e6:.1f} mln" if v >= 1e6 else f"€{v / 1e3:.0f} tys."


def summary_text(row: pd.Series, factors: list[dict]) -> str:
    """Deterministic explanation built from the model output (no LLM involved)."""
    up = [f for f in factors if f["direction"] == "positive"][:2]
    down = [f for f in factors if f["direction"] == "negative"][:2]
    parts = [f"Wartość rynkowa: {_eur(row['market_value_eur'])} "
             f"(przedział 80%: {_eur(row['market_value_low_eur'])}–{_eur(row['market_value_high_eur'])})."]
    if up:
        parts.append("Wycenę podnoszą: " + ", ".join(f"{f['factor'].lower()} ({f['effect_pct']:+.0f}%)" for f in up) + ".")
    if down:
        parts.append("Wycenę obniżają: " + ", ".join(f"{f['factor'].lower()} ({f['effect_pct']:+.0f}%)" for f in down) + ".")
    if pd.notna(row.get("forecast_12m_eur")):
        change = row["forecast_12m_eur"] / row["market_value_eur"] - 1
        trend = "wzrost" if change > 0.05 else "spadek" if change < -0.05 else "stabilizację"
        parts.append(f"Prognoza na 12 miesięcy wskazuje na {trend} ({change:+.0%}).")
    if row["data_quality"] < 0.6:
        parts.append("Uwaga: mało danych o zawodniku, wycena jest mniej pewna.")
    return " ".join(parts)


N_BUYERS = 10


def transfer_scenarios(data: Data, valuer: Valuer, feats: pd.DataFrame) -> pd.DataFrame:
    """Default Transfer Value when no buyer is named.

    The Transfer Fee Engine is run for every other club as the buyer. The
    published value is the median over the ``N_BUYERS`` clubs whose typical
    purchase of the last three years (median fee paid) is closest to the
    player's market value - i.e. the clubs that actually shop at this price
    level. A much richer or much poorer buyer is a different question, answered
    by the transfer simulator.
    """
    as_of = feats["as_of"].iloc[0]
    clubs = data.clubs["id"].to_numpy()
    fees = data.fee_transfers()
    fees = fees[(fees["date"] < as_of) & (fees["date"] >= as_of - pd.Timedelta(days=1095))]
    typical = np.log(fees.groupby("to_club_id")["fee_eur"].median().reindex(clubs)).to_numpy()
    mats = {k: np.empty((len(feats), len(clubs))) for k in ("transfer_value_eur", "transfer_value_low_eur", "transfer_value_high_eur")}
    for j, club in enumerate(clubs):
        p = valuer.transfer(add_deal_features(data, feats.assign(buyer_club_id=club), valuer.adj))
        for k in mats:
            mats[k][:, j] = p[k].to_numpy()
    dist = np.abs(typical[None, :] - np.log(feats["market_value_eur"].to_numpy())[:, None])
    dist = np.where(np.isnan(dist) | (clubs[None, :] == feats["club_id"].to_numpy()[:, None]), np.inf, dist)
    plausible = np.zeros_like(dist, dtype=bool)
    np.put_along_axis(plausible, np.argsort(dist, axis=1)[:, :N_BUYERS], True, axis=1)
    plausible &= np.isfinite(dist)
    out = {k: np.nanmedian(np.where(plausible, m, np.nan), axis=1) for k, m in mats.items()}
    out["n_plausible_buyers"] = plausible.sum(axis=1)
    return pd.DataFrame(out, index=feats.index)


def similar_players(feats: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    cols = ["age", "l1_idx_adj", "l1_minutes_share", "l1_xg90", "l1_xa90", "l1_def_actions90",
            "l1_progressive_passes90", "log_mv"]
    f = feats.assign(log_mv=np.log(feats["market_value_eur"]))
    rows = []
    for _, g in f.groupby("position_group"):
        x = g[cols].astype(float)
        z = ((x - x.mean()) / x.std().replace(0, 1)).fillna(0).to_numpy()
        ids = g["player_id"].to_numpy()
        for i in range(len(g)):
            d = np.sqrt(((z - z[i]) ** 2).sum(axis=1))
            d[i] = np.inf
            for rank, j in enumerate(np.argsort(d)[:k], start=1):
                rows.append({"player_id": int(ids[i]), "similar_player_id": int(ids[j]), "rank": rank,
                             "similarity": round(float(1 / (1 + d[j])), 4)})
    return pd.DataFrame(rows)


def value_all(engine: Engine, version: str | None = None) -> dict:
    """Valuations for every yearly snapshot (history) and a full valuation for the latest one."""
    row, models = load_models(engine, version)
    valuer = Valuer(models)
    adj, mv, pot = valuer.adj, valuer.mv, valuer.pot
    data = Data.from_db(engine)
    panel = panel_frame(data, adj)
    panel = panel.join(valuer.market(panel))
    panel["data_quality"] = data_quality(panel)
    panel["uncertainty"] = mv.uncertainty_level(panel)
    latest_date = panel["as_of"].max()
    latest = panel[panel["as_of"] == latest_date].reset_index(drop=True)

    latest = latest.join(transfer_scenarios(data, valuer, latest))
    d = pot.predict(latest)
    bands = {}
    for months, h, scale in ((6, 12, 0.5), (12, 12, 1.0), (24, 24, 1.0)):
        for q in ("low", "mid", "high"):
            bands[(months, q)] = latest["market_value_eur"] * np.exp(d[f"delta_{h}_{q}"] * scale)
        latest[f"forecast_{months}m_eur"] = bands[(months, "mid")]
    factors = explain(mv, latest)
    details = []
    for i, r in latest.iterrows():
        details.append({
            "factors": factors[i],
            "summary": summary_text(r, factors[i]),
            "forecast": {str(m): {q: round(float(bands[(m, q)].iloc[i]), -3) for q in ("low", "mid", "high")} for m in (6, 12, 24)},
            "n_plausible_buyers": int(r["n_plausible_buyers"]),
            "profile": {k: (None if pd.isna(r[k]) else round(float(r[k]), 3)) for k in
                        ("age", "contract_months", "l1_minutes", "l1_goals90", "l1_assists90", "l1_xg90", "l1_xa90",
                         "l1_idx_adj", "inj_days_365", "league_difficulty")},
        })
    latest["details"] = details

    cols = ["player_id", "as_of", "club_id", "league_id", "market_value_eur", "market_value_low_eur", "market_value_high_eur",
            "ref_value_eur", "data_quality", "uncertainty"]
    extra = ["transfer_value_eur", "transfer_value_low_eur", "transfer_value_high_eur",
             "forecast_6m_eur", "forecast_12m_eur", "forecast_24m_eur", "details"]
    hist = panel.loc[panel["as_of"] != latest_date, cols]
    out = pd.concat([hist, latest[cols + extra]], ignore_index=True)
    out = out.rename(columns={"as_of": "as_of_date", "ref_value_eur": "reference_value_eur"})
    money = [c for c in out.columns if c.endswith("_eur")]
    out[money] = out[money].round(-3)
    out["as_of_date"] = out["as_of_date"].dt.date
    out["model_version_id"] = row["id"]
    out["created_at"] = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = out.astype(object).where(out.notna(), None)

    sim = similar_players(latest)
    sim["model_version_id"] = row["id"]
    sim["as_of_date"] = latest_date.date()
    with engine.begin() as conn:
        conn.execute(delete(db.ModelPrediction).where(db.ModelPrediction.model_version_id == row["id"]))
        conn.execute(delete(db.PlayerComparison).where(db.PlayerComparison.model_version_id == row["id"]))
        out.to_sql("model_predictions", conn, if_exists="append", index=False, chunksize=1000, dtype={"details": JSON})
        sim.to_sql("player_comparisons", conn, if_exists="append", index=False, chunksize=1000)
    latest.drop(columns=["details"]).to_pickle(Path(row["artifact_path"]) / "latest_features.pkl")
    return {"version": row["name"], "as_of": str(latest_date.date()), "n_latest": len(latest), "n_history": len(hist)}
