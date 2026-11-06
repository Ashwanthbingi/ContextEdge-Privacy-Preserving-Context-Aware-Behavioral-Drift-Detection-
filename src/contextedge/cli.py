from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from contextedge.config import Context
from contextedge.engine import ContextEdgeEngine


def main(argv=None):
    parser = argparse.ArgumentParser(description="ContextEdge local behavioural intelligence")
    parser.add_argument(
        "--data-dir", type=Path, help="Private local data directory (default outside source tree)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor")
    monitor.add_argument("--seconds", type=float, default=60)
    monitor.add_argument("--context", choices=[str(c) for c in Context])
    monitor.add_argument(
        "--calibrate",
        action="store_true",
        help="Explicitly label this manual context as representative normal",
    )
    train = sub.add_parser("train")
    train.add_argument("context", choices=[str(c) for c in Context])
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output", type=Path, default=Path("reports/synthetic"))
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--epochs", type=int, default=18)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--seconds", type=float, default=30)
    benchmark.add_argument("--output", type=Path, default=Path("reports/laptop_benchmark.json"))
    imported = sub.add_parser("import-evaluation")
    imported.add_argument("report", type=Path)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        from contextedge.evaluation.runner import run_evaluation

        results = run_evaluation(args.output, args.seed, args.epochs)
        print(
            json.dumps(
                {"source": results["source"], "ai": results["ai"], "drift": results["drift"]}, indent=2
            )
        )
        return
    engine = ContextEdgeEngine(args.data_dir)
    if args.command == "train":
        profile = engine.train(args.context)
        print(json.dumps(profile.metadata, indent=2))
    elif args.command == "monitor":
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        engine.set_context(args.context)
        engine.set_calibration(args.calibrate)
        engine.start()
        try:
            until = time.monotonic() + args.seconds
            while time.monotonic() < until and engine.running:
                time.sleep(min(1, max(0, until - time.monotonic())))
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop()
        if engine.error:
            raise RuntimeError(engine.error)
        print(json.dumps(engine.latest, indent=2))
    elif args.command == "benchmark":
        from contextedge.evaluation.benchmark import run_benchmark

        print(json.dumps(run_benchmark(engine, args.seconds, args.output), indent=2))
    elif args.command == "import-evaluation":
        results = json.loads(args.report.read_text(encoding="utf-8"))
        engine.repo.add_evaluation(results["source"], results["name"], results)
        print("Executed evaluation report imported into local store")
    elif args.command == "status":
        print(
            json.dumps(
                {
                    "data_dir": str(engine.data_dir),
                    "profiles": engine.repo.versions(),
                    "last_observations": engine.repo.recent(1),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
