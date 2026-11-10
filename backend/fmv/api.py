"""Valuation API (FastAPI). Run with ``uvicorn fmv.api:app``.

Reads valuations stored by ``fmv value`` and runs the Transfer Fee Engine live
for the transfer simulator.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select

from . import db
from .features import Data, add_deal_features
from .pipeline import Valuer, load_models

P, MP, C, L = db.Player.__table__, db.ModelPrediction.__table__, db.Club.__table__, db.League.__table__

app = FastAPI(title="Football Market Value AI", version="0.1.0",
              description="Market Value, Transfer Value and 6/12/24-month forecasts for football players.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("FMV_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class State:
    """Everything the API needs, loaded once."""

    def __init__(self) -> None:
        self.engine = db.get_engine()
        self.version, models = load_models(self.engine, os.environ.get("FMV_MODEL_VERSION"))
        self.valuer = Valuer(models)
        with self.engine.connect() as conn:
            self.as_of = conn.execute(select(func.max(MP.c.as_of_date)).where(MP.c.model_version_id == self.version["id"])).scalar()
        if self.as_of is None:
            raise LookupError("model version has no valuations - run `fmv value`")
        self.latest = pd.read_pickle(Path(self.version["artifact_path"]) / "latest_features.pkl").set_index("player_id", drop=False)
        self._data: Data | None = None

    @property
    def data(self) -> Data:
        if self._data is None:
            self._data = Data.from_db(self.engine)
        return self._data

    def rows(self, query) -> list[dict]:
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]


@lru_cache(maxsize=1)
def get_state() -> State:
    try:
        return State()
    except LookupError as exc:  # no model / no valuations yet
        raise HTTPException(503, str(exc)) from exc


def _latest(state: State):
    return and_(MP.c.model_version_id == state.version["id"], MP.c.as_of_date == state.as_of)


_LIST_COLS = [P.c.id, P.c.name, P.c.birth_date, P.c.nationality, P.c.position, P.c.position_group,
              C.c.name.label("club"), L.c.name.label("league"), MP.c.league_id, MP.c.club_id,
              MP.c.market_value_eur, MP.c.market_value_low_eur, MP.c.market_value_high_eur,
              MP.c.transfer_value_eur, MP.c.forecast_12m_eur, MP.c.reference_value_eur,
              MP.c.data_quality, MP.c.uncertainty]


def _listing(state: State):
    return (select(*_LIST_COLS).select_from(
        MP.join(P, P.c.id == MP.c.player_id).outerjoin(C, C.c.id == MP.c.club_id).outerjoin(L, L.c.id == MP.c.league_id))
        .where(_latest(state)))


def _filters(q, league_id: int | None, position_group: str | None):
    if league_id:
        q = q.where(MP.c.league_id == league_id)
    if position_group:
        q = q.where(P.c.position_group == position_group.upper())
    return q


def _age(rows: list[dict], as_of) -> list[dict]:
    for r in rows:
        r["age"] = round((as_of - r["birth_date"]).days / 365.25, 1)
    return rows


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/summary")
def summary(state: State = Depends(get_state)) -> dict:
    with state.engine.connect() as conn:
        count = lambda table: conn.execute(select(func.count()).select_from(table)).scalar()  # noqa: E731
        sources = [r[0] for r in conn.execute(select(db.Transfer.source).distinct())]
        out = {
            "players": count(P), "clubs": count(C), "leagues": count(L), "transfers": count(db.Transfer.__table__),
            "valued_players": conn.execute(select(func.count()).select_from(MP).where(_latest(state))).scalar(),
        }
    m = state.version["metrics"]
    out.update({
        "as_of": str(state.as_of), "model_version": state.version["name"],
        "synthetic_data": all(s.startswith("synthetic") for s in sources),
        "test_metrics": {"market_value": m["market_value"]["overall"], "transfer_value": m["transfer_value"]["overall"],
                         "reference_valuation": m["baselines"]["reference_valuation"], "split": m["split"]},
    })
    return out


@app.get("/model")
def model(state: State = Depends(get_state)) -> dict:
    v = state.version
    return {"name": v["name"], "created_at": str(v["created_at"]), "train_end": str(v["train_end"]),
            "val_end": str(v["val_end"]), "params": v["params"], "metrics": v["metrics"]}


@app.get("/leagues")
def leagues(state: State = Depends(get_state)) -> list[dict]:
    return state.rows(select(L).order_by(L.c.id))


@app.get("/clubs")
def clubs(league_id: int | None = None, state: State = Depends(get_state)) -> list[dict]:
    q = select(C.c.id, C.c.name, C.c.league_id, L.c.name.label("league")).join(L, L.c.id == C.c.league_id)
    if league_id:
        q = q.where(C.c.league_id == league_id)
    return state.rows(q.order_by(L.c.id, C.c.name))


@app.get("/players")
def players(q: str | None = None, league_id: int | None = None, position_group: str | None = None,
            sort: str = Query("market_value", pattern="^(market_value|transfer_value|name)$"),
            limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0), state: State = Depends(get_state)) -> dict:
    query = _filters(_listing(state), league_id, position_group)
    if q:
        like = f"%{q.lower()}%"
        query = query.where(or_(func.lower(P.c.name).like(like), func.lower(C.c.name).like(like)))
    order = {"market_value": MP.c.market_value_eur.desc(), "transfer_value": MP.c.transfer_value_eur.desc(), "name": P.c.name.asc()}[sort]
    with state.engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(query.subquery())).scalar()
    return {"total": total, "items": _age(state.rows(query.order_by(order, P.c.id).limit(limit).offset(offset)), state.as_of)}


@app.get("/rankings/undervalued")
def undervalued(league_id: int | None = None, position_group: str | None = None, max_age: float | None = None,
                min_value_eur: float = 1e6, min_data_quality: float = 0.7, limit: int = Query(25, ge=1, le=200),
                state: State = Depends(get_state)) -> list[dict]:
    """Players whose model Market Value is furthest above the external reference valuation.

    Only confident valuations are ranked: enough data and the reference value
    below the lower end of the model's 80% interval.
    """
    ratio = (MP.c.market_value_eur / MP.c.reference_value_eur).label("value_ratio")
    query = _filters(_listing(state).add_columns(ratio), league_id, position_group).where(
        MP.c.reference_value_eur > 0, MP.c.market_value_eur >= min_value_eur, MP.c.data_quality >= min_data_quality,
        MP.c.reference_value_eur < MP.c.market_value_low_eur)
    rows = _age(state.rows(query.order_by(ratio.desc()).limit(limit * 4 if max_age else limit)), state.as_of)
    if max_age:
        rows = [r for r in rows if r["age"] <= max_age]
    return rows[:limit]


def _profile(state: State, player_id: int) -> dict:
    found = state.rows(_listing(state).add_columns(
        P.c.foot, P.c.height_cm, MP.c.transfer_value_low_eur, MP.c.transfer_value_high_eur,
        MP.c.forecast_6m_eur, MP.c.forecast_24m_eur, MP.c.details).where(P.c.id == player_id))
    if not found:
        raise HTTPException(404, "player not found or not valued in the latest snapshot")
    return _age(found, state.as_of)[0]


@app.get("/players/{player_id}")
def player(player_id: int, state: State = Depends(get_state)) -> dict:
    out = _profile(state, player_id)
    out["as_of"] = str(state.as_of)
    out["history"] = state.rows(
        select(MP.c.as_of_date, MP.c.market_value_eur, MP.c.market_value_low_eur, MP.c.market_value_high_eur, MP.c.reference_value_eur)
        .where(MP.c.model_version_id == state.version["id"], MP.c.player_id == player_id).order_by(MP.c.as_of_date))
    # valuations dated before the training cut-off were produced by a model that has seen those seasons
    for h in out["history"]:
        h["in_sample"] = h["as_of_date"] < state.version["train_end"]
    PC, T, S, SE = (db.PlayerComparison.__table__, db.Transfer.__table__, db.PlayerStatistic.__table__, db.Season.__table__)
    out["similar"] = _age(state.rows(
        _listing(state).add_columns(PC.c.similarity, PC.c.rank).join(PC, PC.c.similar_player_id == P.c.id)
        .where(PC.c.player_id == player_id, PC.c.model_version_id == state.version["id"]).order_by(PC.c.rank)), state.as_of)
    frm, to = C.alias("frm"), C.alias("to")
    out["transfers"] = state.rows(
        select(T.c.date, T.c.type, T.c.fee_eur, T.c.addons_eur, T.c.fee_confirmed, frm.c.name.label("from_club"), to.c.name.label("to_club"))
        .select_from(T.outerjoin(frm, frm.c.id == T.c.from_club_id).outerjoin(to, to.c.id == T.c.to_club_id))
        .where(T.c.player_id == player_id, T.c.date <= state.as_of).order_by(T.c.date.desc()))
    out["seasons"] = state.rows(
        select(SE.c.name.label("season"), C.c.name.label("club"), L.c.name.label("league"), S.c.appearances, S.c.minutes,
               S.c.goals, S.c.assists, S.c.xg, S.c.xa)
        .select_from(S.join(SE, SE.c.id == S.c.season_id).join(C, C.c.id == S.c.club_id).join(L, L.c.id == S.c.league_id))
        .where(S.c.player_id == player_id, SE.c.end_date < state.as_of).order_by(SE.c.id.desc(), S.c.stint.desc()))
    return out


@app.get("/compare")
def compare(ids: str = Query(..., description="comma separated player ids (2-4)"), state: State = Depends(get_state)) -> list[dict]:
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError as exc:
        raise HTTPException(422, "ids must be integers") from exc
    if not 2 <= len(wanted) <= 4:
        raise HTTPException(422, "give 2 to 4 player ids")
    return [_profile(state, pid) for pid in wanted]


class SimulationRequest(BaseModel):
    player_id: int
    buyer_club_id: int
    window: str = Field("summer", pattern="^(summer|winter)$")
    contract_months: float | None = Field(None, ge=0, le=84, description="override the remaining contract length")


@app.post("/simulate")
def simulate(req: SimulationRequest, state: State = Depends(get_state)) -> dict:
    """Transfer Value for one concrete deal: this player to this buyer in this window."""
    if req.player_id not in state.latest.index:
        raise HTTPException(404, "player not valued in the latest snapshot")
    row = state.latest.loc[[req.player_id]].reset_index(drop=True)
    club = state.rows(select(C.c.id, C.c.name, L.c.name.label("league")).join(L, L.c.id == C.c.league_id).where(C.c.id == req.buyer_club_id))
    if not club:
        raise HTTPException(404, "buyer club not found")
    if int(row["club_id"].iloc[0]) == req.buyer_club_id:
        raise HTTPException(422, "the buyer is the player's current club")
    if req.contract_months is not None:
        row["contract_months"] = req.contract_months
    as_of = pd.Timestamp(state.as_of)
    deal_date = pd.Timestamp(as_of.year, 7, 1) if req.window == "summer" else pd.Timestamp(as_of.year + 1, 1, 15)
    market = state.valuer.market(row)
    row = row.drop(columns=[c for c in market.columns if c in row.columns]).join(market)
    deal = add_deal_features(state.data, row.assign(buyer_club_id=req.buyer_club_id, deal_date=deal_date), state.valuer.adj)
    tv = state.valuer.transfer(deal).iloc[0]
    money = lambda v: float(np.round(v, -3))  # noqa: E731
    return {
        "player_id": req.player_id, "buyer": club[0], "window": req.window,
        "contract_months": None if pd.isna(row["contract_months"].iloc[0]) else round(float(row["contract_months"].iloc[0]), 1),
        "market_value_eur": money(market["market_value_eur"].iloc[0]),
        "transfer_value_eur": money(tv["transfer_value_eur"]),
        "transfer_value_low_eur": money(tv["transfer_value_low_eur"]),
        "transfer_value_high_eur": money(tv["transfer_value_high_eur"]),
        "premium_pct": round(float(tv["transfer_value_eur"] / market["market_value_eur"].iloc[0] - 1) * 100, 1),
    }
