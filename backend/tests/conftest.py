import pandas as pd
import pytest

from fmv import config, db
from fmv.features import Data
from fmv.synthetic import generate


@pytest.fixture(scope="session")
def world():
    return generate(seed=3, n_players=2500)


@pytest.fixture(scope="session")
def data(world):
    return Data.from_tables(world.tables)


@pytest.fixture(scope="session")
def trained(world, tmp_path_factory):
    """Seeded database + trained model version + stored valuations."""
    from fmv import pipeline

    root = tmp_path_factory.mktemp("fmv")
    engine = db.get_engine(f"sqlite:///{root / 'fmv.db'}")
    db.create_schema(engine, drop=True)
    db.write_tables(engine, world.tables)
    result = pipeline.train(engine, name="test", artifacts_dir=root / "artifacts")
    valued = pipeline.value_all(engine, "test")
    return {"engine": engine, "result": result, "valued": valued, "root": root}


@pytest.fixture(scope="session")
def train_end():
    return pd.Timestamp(config.TRAIN_END)
