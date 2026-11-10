import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained, monkeypatch_module):
    from fmv import api, config

    monkeypatch_module.setattr(config, "DATABASE_URL", str(trained["engine"].url))
    api.get_state.cache_clear()
    yield TestClient(api.app)
    api.get_state.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_summary(client):
    r = client.get("/summary").json()
    assert r["leagues"] == 8 and r["valued_players"] > 500
    assert r["synthetic_data"] is True
    assert "median_ape" in r["test_metrics"]["market_value"]


def test_search_filter_and_profile(client):
    top = client.get("/players", params={"limit": 5}).json()
    assert top["total"] > 500 and len(top["items"]) == 5
    values = [p["market_value_eur"] for p in top["items"]]
    assert values == sorted(values, reverse=True)
    first = top["items"][0]
    found = client.get("/players", params={"q": first["name"].split()[1].lower()}).json()
    assert any(p["id"] == first["id"] for p in found["items"])
    gk = client.get("/players", params={"position_group": "gk", "league_id": 1}).json()
    assert gk["items"] and all(p["position_group"] == "GK" and p["league_id"] == 1 for p in gk["items"])

    p = client.get(f"/players/{first['id']}").json()
    assert p["details"]["summary"].startswith("Wartość rynkowa")
    assert len(p["details"]["factors"]) == 10 and set(p["details"]["forecast"]) == {"6", "12", "24"}
    assert len(p["similar"]) == 5 and p["history"] and p["seasons"]
    assert p["history"][-1]["market_value_eur"] == p["market_value_eur"]


def test_simulator(client):
    player = client.get("/players", params={"limit": 1}).json()["items"][0]
    clubs = [c for c in client.get("/clubs").json() if c["id"] != player["club_id"]]
    base = {"player_id": player["id"], "buyer_club_id": clubs[0]["id"]}
    r = client.post("/simulate", json=base).json()
    assert r["transfer_value_low_eur"] <= r["transfer_value_eur"] <= r["transfer_value_high_eur"]
    assert r["market_value_eur"] == player["market_value_eur"]
    short = client.post("/simulate", json={**base, "contract_months": 6}).json()
    long = client.post("/simulate", json={**base, "contract_months": 48}).json()
    assert short["transfer_value_eur"] < long["transfer_value_eur"]  # a running-down contract is cheaper
    assert client.post("/simulate", json={**base, "buyer_club_id": player["club_id"]}).status_code == 422
    assert client.post("/simulate", json={**base, "buyer_club_id": 99999}).status_code == 404


def test_rankings_and_compare(client):
    rows = client.get("/rankings/undervalued", params={"limit": 10}).json()
    assert rows and all(r["value_ratio"] > 1 and r["reference_value_eur"] < r["market_value_low_eur"]
                        and r["uncertainty"] != "high" for r in rows)
    ids = ",".join(str(r["id"]) for r in rows[:2])
    assert len(client.get("/compare", params={"ids": ids}).json()) == 2
    assert client.get("/compare", params={"ids": "1"}).status_code == 422


def test_unknown_player(client):
    assert client.get("/players/9999999").status_code == 404


def test_model_report(client):
    r = client.get("/model").json()
    assert r["metrics"]["split"]["n_test"] > 0 and r["params"]["interval_coverage"] == 0.8
