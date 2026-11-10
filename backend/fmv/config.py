"""Central configuration. Everything is overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATABASE_URL = os.environ.get("FMV_DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'fmv.db'}")
ARTIFACTS_DIR = Path(os.environ.get("FMV_ARTIFACTS_DIR", ROOT / "artifacts"))

# Scope of the first version (section 10 of the project brief).
LEAGUES = [
    ("Premier League", "England"),
    ("La Liga", "Spain"),
    ("Bundesliga", "Germany"),
    ("Serie A", "Italy"),
    ("Ligue 1", "France"),
    ("Liga Portugal", "Portugal"),
    ("Eredivisie", "Netherlands"),
    ("Ekstraklasa", "Poland"),
]

# Chronological split. Transfers strictly before TRAIN_END train the models, the
# window up to VAL_END is used for early stopping, ensemble weights and conformal
# calibration, everything after VAL_END is the untouched test set.
TRAIN_END = pd.Timestamp(os.environ.get("FMV_TRAIN_END", "2021-06-20"))
VAL_END = pd.Timestamp(os.environ.get("FMV_VAL_END", "2022-06-20"))

# Nominal coverage of the published value interval (q10-q90).
INTERVAL_COVERAGE = 0.80

# Day of the year used for periodic valuations: after the season ends,
# before the summer window opens.
SNAPSHOT_MONTH, SNAPSHOT_DAY = 6, 15
