import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fmv import db
from fmv.engines import DriftCorrector
from fmv.features import DEAL_CAT, DEAL_NUM, PLAYER_CAT, PLAYER_NUM, Data, add_deal_features, build_player_features, market_index
from fmv.metrics import regression_metrics
from fmv.pipeline import deal_frame, fit_adjuster
from fmv.synthetic import _STRENGTH


def _truncate(world, cutoff):
    """The world as it was known on ``cutoff``: every later record removed."""
    t = {k: v.copy() for k, v in world.tables.items()}
    date_col = {"transfers": "date", "contracts": "start_date", "injuries": "start_date", "market_valuations": "date"}
    for name, col in date_col.items():
        t[name] = t[name][pd.to_datetime(t[name][col]) < cutoff]
    ended = t["seasons"].loc[pd.to_datetime(t["seasons"]["end_date"]) < cutoff, "id"]
    t["player_statistics"] = t["player_statistics"][t["player_statistics"]["season_id"].isin(ended)]
    # injuries still running on the cut-off date: the end is not known yet
    inj = t["injuries"].copy()
    running = pd.to_datetime(inj["end_date"]) > cutoff
    inj.loc[running, "end_date"] = cutoff.date()
    t["injuries"] = inj
    return Data.from_tables(t)


def test_features_do_not_use_the_future(world, data, train_end):
    """Deleting everything that happened after the valuation date must not change any feature."""
    cutoff = pd.Timestamp("2019-08-10")
    adj = fit_adjuster(data, pd.Timestamp("2018-06-20"))
    deals = data.fee_transfers()
    deals = deals[(deals["date"] > cutoff - pd.Timedelta(days=500)) & (deals["date"] <= cutoff)]
    assert len(deals) > 100
    snaps = pd.DataFrame({"player_id": deals["player_id"].to_numpy(), "as_of": (deals["date"] - pd.Timedelta(days=1)).to_numpy(),
                          "club_id": deals["from_club_id"].to_numpy(), "buyer_club_id": deals["to_club_id"].to_numpy(),
                          "deal_date": deals["date"].to_numpy()})
    past = _truncate(world, cutoff)
    full = add_deal_features(data, build_player_features(data, snaps, adj), adj)
    known = add_deal_features(past, build_player_features(past, snaps, adj), adj)
    cols = PLAYER_NUM + DEAL_NUM + ["market_index", "ref_value_eur"]
    pd.testing.assert_frame_equal(full[cols], known[cols], check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(full[PLAYER_CAT + DEAL_CAT], known[PLAYER_CAT + DEAL_CAT])


def test_club_is_derived_from_history_before_the_deal(data):
    adj = fit_adjuster(data, pd.Timestamp("2018-06-20"))
    deals = data.fee_transfers().head(400)
    snaps = pd.DataFrame({"player_id": deals["player_id"].to_numpy(), "as_of": (deals["date"] - pd.Timedelta(days=1)).to_numpy()})
    feats = build_player_features(data, snaps, adj)
    assert (feats["club_id"].to_numpy() == deals["from_club_id"].to_numpy()).all()


def test_labels_exclude_loans_free_and_undisclosed(data):
    t = data.transfers
    labels = data.fee_transfers()
    assert set(labels["type"]) == {"permanent"}
    assert (labels["fee_eur"] > 0).all() and labels["fee_confirmed"].all()
    assert {"loan", "loan_return", "free"} <= set(t["type"])
    assert t[(t["type"] == "permanent") & t["fee_eur"].isna()].shape[0] > 0  # undisclosed fees exist and are dropped
    assert len(labels) < (t["type"] == "permanent").sum()


def test_league_adjustment_recovers_hidden_league_order(data, train_end):
    adj = fit_adjuster(data, train_end)
    learned = [adj.difficulty_[i + 1] for i in range(len(_STRENGTH))]
    assert spearmanr(learned, _STRENGTH).statistic > 0.9
    assert adj.n_movers_ > 100


def test_adjuster_only_sees_training_seasons(data, train_end):
    early = fit_adjuster(data, pd.Timestamp("2017-06-20"))
    late = fit_adjuster(data, train_end)
    assert early.n_movers_ < late.n_movers_


def test_market_index_uses_only_earlier_fees(data):
    dates = pd.Series(pd.to_datetime(["2018-07-01", "2022-07-01"]))
    idx = market_index(data, dates)
    fees = data.fee_transfers()
    for d, value in zip(dates, idx):
        window = fees[(fees["date"] < d) & (fees["date"] >= d - pd.Timedelta(days=730))]
        assert np.isclose(value, np.log(window["fee_eur"]).mean())


def test_drift_corrector_ignores_deals_on_or_after_the_date():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]))
    drift = DriftCorrector(days=365, prior=0.0).fit(dates, np.array([0.2, 0.4, 9.0]))
    out = drift.at(pd.Series(pd.to_datetime(["2020-01-01", "2020-12-31", "2021-03-01"])))
    assert out[0] == 0.0  # nothing before the first deal
    assert np.isclose(out[1], 0.3)
    assert np.isclose(out[2], (0.4 + 9.0) / 2)


def test_metrics():
    m = regression_metrics([100.0, 200.0], [110.0, 180.0], [105.0, 170.0], [120.0, 210.0])
    assert np.isclose(m["mae_eur"], 15.0)
    assert np.isclose(m["wape"], 30 / 300)
    assert np.isclose(m["median_ape"], 0.1)
    assert m["interval_coverage"] == 0.5


def test_schema_has_the_twelve_tables():
    assert set(db.Base.metadata.tables) == {
        "players", "clubs", "leagues", "seasons", "player_statistics", "transfers", "contracts", "injuries",
        "market_valuations", "model_predictions", "player_comparisons", "model_versions"}
