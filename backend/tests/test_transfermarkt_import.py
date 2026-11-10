import gzip
from pathlib import Path

import pandas as pd

from fmv.importers import transfermarkt as tm


def _t(rows):
    return pd.DataFrame(rows, columns=["player_id", "date", "from_club_tm", "to_club_tm", "fee_eur"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_loans_and_returns_are_detected_even_with_a_loan_fee():
    t = tm.classify_transfers(_t([
        (1, "2020-08-01", 10, 20, 2_000_000.0),   # paid loan
        (1, "2021-06-30", 20, 10, 0.0),           # end of loan
        (1, "2021-08-01", 10, 30, 15_000_000.0),  # permanent sale
        (2, "2020-07-01", 10, 40, 0.0),           # free transfer
        (3, "2020-07-01", 10, 50, None),          # undisclosed
        (3, "2023-07-01", 50, 10, 5_000_000.0),   # sold back years later: permanent
    ]))
    by = {(r.player_id, str(r.date.date())): r.type for r in t.itertuples()}
    assert by[(1, "2020-08-01")] == "loan" and by[(1, "2021-06-30")] == "loan_return"
    assert by[(1, "2021-08-01")] == "permanent"
    assert by[(2, "2020-07-01")] == "free"
    assert by[(3, "2020-07-01")] == "permanent" and by[(3, "2023-07-01")] == "permanent"


def _write(dir_: Path, name: str, df: pd.DataFrame) -> None:
    with gzip.open(dir_ / f"{name}.csv.gz", "wt") as fh:
        df.to_csv(fh, index=False)


def test_load_builds_schema_tables(tmp_path):
    _write(tmp_path, "competitions", pd.DataFrame({"competition_id": ["GB1"]}))
    _write(tmp_path, "clubs", pd.DataFrame({"club_id": [1, 2, 3], "name": ["A", "B", "Abroad"],
                                            "domestic_competition_id": ["GB1", "GB1", "TR1"]}))
    games = pd.DataFrame({"game_id": [100, 101], "competition_id": "GB1", "season": 2020,
                          "date": ["2020-09-01", "2021-03-01"], "home_club_id": [1, 2], "away_club_id": [2, 1],
                          "home_club_goals": [2, 0], "away_club_goals": [0, 0]})
    _write(tmp_path, "games", pd.concat([games, pd.DataFrame({"game_id": [999], "competition_id": "GB1", "season": 2021,
                                                              "date": ["2022-03-01"], "home_club_id": [1], "away_club_id": [2],
                                                              "home_club_goals": [1], "away_club_goals": [1]})]))
    _write(tmp_path, "appearances", pd.DataFrame({
        "game_id": [100, 101, 100], "player_id": [7, 7, 8], "player_club_id": [1, 1, 2], "competition_id": "GB1",
        "date": ["2020-09-01", "2021-03-01", "2020-09-01"], "goals": [1, 0, 0], "assists": [0, 1, 0], "minutes_played": [90, 90, 90]}))
    _write(tmp_path, "players", pd.DataFrame({
        "player_id": [7, 8], "name": ["Seven", "Eight"], "date_of_birth": ["1999-01-01", "2000-01-01"],
        "sub_position": ["Centre-Forward", "Goalkeeper"], "position": ["Attack", "Goalkeeper"], "foot": ["right", None],
        "height_in_cm": [180, 0], "country_of_citizenship": ["Spain", None]}))
    _write(tmp_path, "transfers", pd.DataFrame({
        "player_id": [7, 7], "transfer_date": ["2021-07-01", "2030-06-30"], "from_club_id": [1, 1], "to_club_id": [3, 2],
        "transfer_fee": [12_000_000, 0]}))
    _write(tmp_path, "player_valuations", pd.DataFrame({"player_id": [7], "date": ["2021-01-01"], "market_value_in_eur": [9_000_000]}))

    t = tm.load(tmp_path)
    assert list(t["seasons"]["id"]) == [2020]
    s = t["player_statistics"].set_index("player_id")
    assert s.loc[7, "minutes"] == 180 and s.loc[7, "goals"] == 1 and s.loc[7, "assists"] == 1
    assert s.loc[8, "clean_sheets"] == 0 and s.loc[8, "goals_conceded"] == 2
    assert s["xg"].isna().all()  # not published by the source
    assert s.loc[7, "team_minutes"] == 180 and s.loc[7, "team_points_per_game"] == 2.0
    tr = t["transfers"]
    assert len(tr) == 1  # the 2030 "future" row is dropped
    assert tr.iloc[0]["fee_confirmed"] and tr.iloc[0]["type"] == "permanent" and tr.iloc[0]["to_club_id"] == 3
    clubs = t["clubs"].set_index("id")
    assert clubs.loc[3, "league_id"] == tm.OTHER_LEAGUE[0]
    assert t["contracts"].empty and t["injuries"].empty
    assert pd.isna(t["players"].set_index("id").loc[8, "height_cm"])
