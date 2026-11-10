import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select

from fmv import db


def test_split_is_chronological(trained):
    s = trained["result"]["report"]["split"]
    assert s["train_end"] <= s["val_end"] <= s["test_from"]
    assert min(s["n_train"], s["n_val"], s["n_test"]) >= 50


def test_models_beat_the_naive_baseline_on_unseen_transfers(trained):
    r = trained["result"]["report"]
    naive = r["baselines"]["position_median"]["median_ape"]
    assert r["market_value"]["overall"]["median_ape"] < 0.7 * naive
    assert r["transfer_value"]["overall"]["median_ape"] < 0.7 * naive


def test_intervals_are_roughly_calibrated(trained):
    r = trained["result"]["report"]
    for key in ("market_value", "transfer_value"):
        assert 0.70 <= r[key]["overall"]["interval_coverage"] <= 0.90
    for h in ("12m", "24m"):
        assert 0.65 <= r["potential"][h]["interval_coverage"] <= 0.92
        assert r["potential"][h]["mae_log_model"] < r["potential"][h]["mae_log_no_change_baseline"]


def test_report_has_required_breakdowns(trained):
    mv = trained["result"]["report"]["market_value"]
    assert set(mv["by_position_group"]) == {"GK", "DEF", "MID", "FWD"}
    assert len(mv["by_league"]) == 8
    assert {"mae_eur", "rmse_eur", "wape", "median_ape", "interval_coverage"} <= set(mv["overall"])


def test_valuations_track_the_hidden_value(trained, world):
    """Synthetic-only check: the model never sees the latent value, but should rank players like it."""
    with trained["engine"].connect() as conn:
        pred = pd.read_sql(select(db.ModelPrediction).where(db.ModelPrediction.as_of_date == pd.Timestamp(trained["valued"]["as_of"]).date()), conn)
    truth = world.truth[world.truth["season_id"] == world.truth["season_id"].max()]
    m = pred.merge(truth, on="player_id")
    assert len(m) > 500
    assert spearmanr(m["market_value_eur"], m["latent_value_eur"]).statistic > 0.8


def test_stored_valuation_is_complete(trained):
    with trained["engine"].connect() as conn:
        pred = pd.read_sql(select(db.ModelPrediction).where(db.ModelPrediction.as_of_date == pd.Timestamp(trained["valued"]["as_of"]).date()), conn)
        sim = pd.read_sql(select(db.PlayerComparison), conn)
    need = ["market_value_eur", "market_value_low_eur", "market_value_high_eur", "transfer_value_eur",
            "forecast_6m_eur", "forecast_12m_eur", "forecast_24m_eur", "data_quality", "uncertainty", "model_version_id"]
    assert pred[need].notna().all().all()
    assert (pred["market_value_low_eur"] <= pred["market_value_eur"]).all()
    assert (pred["market_value_eur"] <= pred["market_value_high_eur"]).all()
    assert (pred["transfer_value_low_eur"] <= pred["transfer_value_high_eur"]).all()
    assert set(pred["uncertainty"]) <= {"low", "medium", "high"}
    assert pred["data_quality"].between(0, 1).all()
    assert sim.groupby("player_id").size().eq(5).all()
    assert (sim["player_id"] != sim["similar_player_id"]).all()
