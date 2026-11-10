import json

import pandas as pd

from fmv.importers import tm_live


def _put(cache, endpoint, pid, data):
    (cache / endpoint).mkdir(parents=True, exist_ok=True)
    (cache / endpoint / f"{pid}.json").write_text(json.dumps(data))


def _cache(tmp_path):
    _put(tmp_path, "transfers", 7, {"id": "7", "transfers": [
        {"clubFrom": {"id": "16"}, "clubTo": {"id": "281"}, "date": "2022-07-01", "upcoming": False,
         "fee": 60000000, "contractUntil": "2034-06-30", "remainingContractDays": 730},
        {"clubFrom": {"id": "409"}, "clubTo": {"id": "16"}, "date": "2020-01-01", "upcoming": False,
         "fee": 20000000, "contractUntil": "2024-06-30", "remainingContractDays": 1276},
        {"clubFrom": {"id": "281"}, "clubTo": {"id": "5"}, "date": "2027-07-01", "upcoming": True, "remainingContractDays": 10},
    ]})
    _put(tmp_path, "transfers", 8, {"id": "8", "missing": True})
    _put(tmp_path, "injuries", 7, {"id": "7", "injuries": [
        {"injury": "Ankle injury", "fromDate": "2025-03-31", "untilDate": "2025-05-01", "days": 32},
        {"injury": "Knock", "fromDate": "2026-02-26", "untilDate": None, "days": 4}]})
    _put(tmp_path, "stats", 9, {"id": "9", "stats": [
        {"competitionId": "PL1", "seasonId": "2025", "clubId": "100", "appearances": 30, "minutesPlayed": 2500,
         "goals": 10, "assists": 4, "goalsConceded": 30, "cleanSheets": 8},
        {"competitionId": "POCP", "seasonId": "2025", "clubId": "100", "appearances": 3, "minutesPlayed": 270, "goals": 1}]})
    return tmp_path


def test_contract_at_transfer_is_point_in_time(tmp_path):
    c = tm_live.contracts(_cache(tmp_path)).sort_values("start_date").reset_index(drop=True)
    assert len(c) == 2  # the upcoming transfer is ignored, the missing player has no rows
    first = c.iloc[0]
    assert first["club_id"] == 409 and first["end_date"] == pd.Timestamp("2020-01-01") + pd.Timedelta(days=1276)
    assert first["start_date"] == pd.Timestamp("2019-12-30")  # visible to the deal, not to earlier snapshots
    # contractUntil (today's renewed contract with the buyer) must not leak into the past
    assert pd.Timestamp("2034-06-30") not in set(c["end_date"])


def test_current_contract_comes_from_latest_past_transfer(tmp_path):
    cache = _cache(tmp_path)
    cur = tm_live.current_contracts(cache, pd.DataFrame({"id": [7, 8]}), pd.Timestamp("2026-06-14"))
    assert len(cur) == 1
    row = cur.iloc[0]
    assert row["club_id"] == 281 and row["end_date"] == pd.Timestamp("2034-06-30") and row["start_date"] == pd.Timestamp("2026-06-14")


def test_injuries_and_league_stats(tmp_path):
    cache = _cache(tmp_path)
    inj = tm_live.injuries(cache).sort_values("start_date")
    assert list(inj["days_missed"]) == [32, 4]
    assert inj.iloc[1]["end_date"] == pd.Timestamp("2026-03-02")  # open end: start + days
    st = tm_live.league_stats(cache, "PL1")
    assert len(st) == 1 and st.iloc[0]["minutes"] == 2500 and st.iloc[0]["season"] == 2025
