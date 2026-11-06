import argparse
import json
from pathlib import Path

from contextedge.engine import ContextEdgeEngine
from contextedge.evaluation.labelled import evaluate_labelled


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate independent labelled numerical features on frozen local profiles"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/labelled_local.json"))
    args = parser.parse_args()
    engine = ContextEdgeEngine(args.data_dir)
    results = evaluate_labelled(args.input, engine.profiles, engine.settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    engine.repo.add_evaluation(results["source"], results["name"], results)
    print(json.dumps(results["ai"], indent=2))


if __name__ == "__main__":
    main()
