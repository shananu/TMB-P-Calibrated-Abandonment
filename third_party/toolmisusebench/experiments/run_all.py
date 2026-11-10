from __future__ import annotations

import argparse
import json

from toolmisusebench.experiment_pipeline import load_experiment_config, run_experiment_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.run_all")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    payload = run_experiment_pipeline(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
