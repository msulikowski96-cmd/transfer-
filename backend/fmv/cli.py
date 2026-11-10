"""Command line entry point: ``python -m fmv.cli <command>``."""
from __future__ import annotations

import argparse
import json

from . import config, db


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fmv", description="Football Market Value AI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="(re)create the schema and load SYNTHETIC development data")
    seed.add_argument("--players", type=int, default=6000)
    seed.add_argument("--seed", type=int, default=7)
    train = sub.add_parser("train", help="train all engines and store a new model version")
    train.add_argument("--name")
    value = sub.add_parser("value", help="value all players with a model version (latest by default)")
    value.add_argument("--version")
    imp = sub.add_parser("import-tm", help="(re)create the schema and load the open transfermarkt-datasets collection")
    imp.add_argument("--dir", default=str(config.ROOT / "data" / "raw" / "transfermarkt"),
                     help="where the CSV files are (downloaded there if missing)")
    imp.add_argument("--no-download", action="store_true")
    imp.add_argument("--live-cache", default=str(config.ROOT / "data" / "raw" / "tm-api"),
                     help="responses from fetch-tm-live; used when the directory exists")
    live = sub.add_parser("fetch-tm-live", help="fetch contracts, injuries and Ekstraklasa stats from Transfermarkt's API (cached, resumable)")
    live.add_argument("--dir", default=str(config.ROOT / "data" / "raw" / "transfermarkt"))
    live.add_argument("--cache", default=str(config.ROOT / "data" / "raw" / "tm-api"))
    live.add_argument("--concurrency", type=int, default=3)
    sub.add_parser("all", help="seed + train + value")
    sub.add_parser("bootstrap", help="run `all` only if the database has no trained model yet (used by Docker)")
    report = sub.add_parser("report", help="print the evaluation report of a model version")
    report.add_argument("--version")
    args = parser.parse_args(argv)

    engine = db.get_engine()
    if args.cmd == "bootstrap":
        from sqlalchemy import func, inspect, select

        has_model = inspect(engine).has_table("model_predictions")
        if has_model:
            with engine.connect() as conn:
                has_model = bool(conn.execute(select(func.count()).select_from(db.ModelPrediction)).scalar())
        if has_model:
            print("model and valuations already present - nothing to do")
            return
        args.cmd = "all"
    if args.cmd in ("seed", "all"):
        from .synthetic import generate

        world = generate(seed=getattr(args, "seed", 7), n_players=getattr(args, "players", 6000))
        db.create_schema(engine, drop=True)
        db.write_tables(engine, world.tables)
        print(f"seeded synthetic data into {config.DATABASE_URL.split('@')[-1]}: "
              + ", ".join(f"{k}={len(v)}" for k, v in world.tables.items()))
    if args.cmd == "import-tm":
        from pathlib import Path

        from .importers import transfermarkt as tm

        src = Path(args.dir)
        if not args.no_download:
            tm.download(src)
        live_cache = Path(args.live_cache)
        tables = tm.load(src, live_cache if live_cache.exists() else None)
        db.create_schema(engine, drop=True)
        db.write_tables(engine, tables)
        print("imported transfermarkt-datasets: " + ", ".join(f"{k}={len(v)}" for k, v in tables.items()))
    if args.cmd == "fetch-tm-live":
        from pathlib import Path

        from .importers import tm_live

        print(tm_live.fetch(Path(args.dir), Path(args.cache), concurrency=args.concurrency))
    if args.cmd in ("train", "all"):
        from .pipeline import train as run_train

        result = run_train(engine, name=getattr(args, "name", None))
        r = result["report"]
        print(f"trained {result['name']}: test n={r['split']['n_test']}")
        for key in ("market_value", "transfer_value"):
            o = r[key]["overall"]
            print(f"  {key:15s} MAE €{o['mae_eur'] / 1e6:.2f}M  WAPE {o['wape']:.1%}  MdAPE {o['median_ape']:.1%}  "
                  f"coverage {o['interval_coverage']:.1%}")
        for key, o in r["baselines"].items():
            print(f"  baseline {key}: WAPE {o['wape']:.1%}  MdAPE {o['median_ape']:.1%} (n={o['n']})")
    if args.cmd in ("value", "all"):
        from .pipeline import value_all

        print("valued:", value_all(engine, getattr(args, "version", None)))
    if args.cmd == "report":
        from .pipeline import load_models

        row, _ = load_models(engine, args.version)
        print(json.dumps(row["metrics"], indent=2))


if __name__ == "__main__":
    main()
