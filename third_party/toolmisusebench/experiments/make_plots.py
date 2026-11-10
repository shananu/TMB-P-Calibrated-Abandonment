from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolmisusebench.experiment_pipeline import build_plot_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.make_plots")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    payload = build_plot_data(results, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
