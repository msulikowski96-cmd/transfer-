"""Database schema (SQLAlchemy 2.0). Runs on PostgreSQL in production and on
SQLite for local development and tests.

Twelve tables, as in the project brief. Two of them are easy to confuse:

* ``market_valuations``  - *external* reference valuations published by portals.
  Never used as a training label or as a model feature, only as a benchmark.
* ``model_predictions``  - valuations produced by this system, one row per
  (player, as-of date, model version).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from . import config


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    country: Mapped[str] = mapped_column(String(60))


class Club(Base):
    __tablename__ = "clubs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)


class Season(Base):
    __tablename__ = "seasons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(20))
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    birth_date: Mapped[dt.date] = mapped_column(Date)
    nationality: Mapped[str] = mapped_column(String(60))
    position: Mapped[str] = mapped_column(String(4))  # GK CB FB DM CM AM W ST
    position_group: Mapped[str] = mapped_column(String(4))  # GK DEF MID FWD
    foot: Mapped[str | None] = mapped_column(String(6), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PlayerStatistic(Base):
    """One row per player / season / club stint (a winter transfer gives two).

    Event statistics (xG, tackles, ...) are NULL when the source does not publish them.
    """

    __tablename__ = "player_statistics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    club_id: Mapped[int] = mapped_column(ForeignKey("clubs.id"))
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    stint: Mapped[int] = mapped_column(Integer, default=1)
    appearances: Mapped[int] = mapped_column(Integer)
    minutes: Mapped[int] = mapped_column(Integer)
    goals: Mapped[int] = mapped_column(Integer)
    assists: Mapped[int] = mapped_column(Integer)
    xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    xa: Mapped[float | None] = mapped_column(Float, nullable=True)
    shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    key_passes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progressive_passes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tackles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    interceptions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aerials_won: Mapped[int | None] = mapped_column(Integer, nullable=True)
    saves: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals_conceded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clean_sheets: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_minutes: Mapped[int] = mapped_column(Integer)  # minutes the team played in the stint
    team_points_per_game: Mapped[float | None] = mapped_column(Float, nullable=True)


class Transfer(Base):
    __tablename__ = "transfers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    from_club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"), nullable=True)
    to_club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"), nullable=True)
    type: Mapped[str] = mapped_column(String(16))  # permanent | loan | loan_return | free
    fee_eur: Mapped[float | None] = mapped_column(Float, nullable=True)  # fixed part only
    addons_eur: Mapped[float | None] = mapped_column(Float, nullable=True)  # conditional bonuses
    fee_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(40))


class Contract(Base):
    """Contract terms as they were signed; ``end_date`` is never rewritten later."""

    __tablename__ = "contracts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    club_id: Mapped[int] = mapped_column(ForeignKey("clubs.id"))
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(40))


class Injury(Base):
    __tablename__ = "injuries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    days_missed: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(40))


class MarketValuation(Base):
    __tablename__ = "market_valuations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    value_eur: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(40))


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)
    train_end: Mapped[dt.date] = mapped_column(Date)
    val_end: Mapped[dt.date] = mapped_column(Date)
    params: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String(300))


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    as_of_date: Mapped[dt.date] = mapped_column(Date)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)
    club_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    league_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    market_value_eur: Mapped[float] = mapped_column(Float)
    market_value_low_eur: Mapped[float] = mapped_column(Float)
    market_value_high_eur: Mapped[float] = mapped_column(Float)
    transfer_value_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    transfer_value_low_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    transfer_value_high_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_6m_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_12m_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_24m_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_value_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_quality: Mapped[float] = mapped_column(Float)  # 0..1
    uncertainty: Mapped[str] = mapped_column(String(10))  # low | medium | high
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # forecast bands, factors, text

    __table_args__ = (
        Index("ix_pred_player_version_date", "player_id", "model_version_id", "as_of_date"),
        Index("ix_pred_version_date", "model_version_id", "as_of_date"),
    )


class PlayerComparison(Base):
    __tablename__ = "player_comparisons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    similar_player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    as_of_date: Mapped[dt.date] = mapped_column(Date)
    rank: Mapped[int] = mapped_column(Integer)
    similarity: Mapped[float] = mapped_column(Float)


SOURCE_TABLES = [
    "leagues", "clubs", "seasons", "players", "player_statistics",
    "transfers", "contracts", "injuries", "market_valuations",
]
DATE_COLUMNS = {
    "seasons": ["start_date", "end_date"],
    "players": ["birth_date"],
    "transfers": ["date"],
    "contracts": ["start_date", "end_date"],
    "injuries": ["start_date", "end_date"],
    "market_valuations": ["date"],
}


def get_engine(url: str | None = None) -> Engine:
    url = url or config.DATABASE_URL
    if url.startswith("sqlite:///"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def create_schema(engine: Engine, drop: bool = False) -> None:
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def write_tables(engine: Engine, tables: dict[str, pd.DataFrame]) -> None:
    """Bulk-insert source tables in foreign-key order."""
    for name in SOURCE_TABLES:
        if name not in tables:
            continue
        df = tables[name].copy()
        for col in DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col]).dt.date
        df = df.astype(object).where(df.notna(), None)
        df.to_sql(name, engine, if_exists="append", index=False, chunksize=1000, method="multi")


def read_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    out = {}
    with engine.connect() as conn:
        for name in SOURCE_TABLES:
            out[name] = pd.read_sql_table(name, conn, parse_dates=DATE_COLUMNS.get(name))
    return out
