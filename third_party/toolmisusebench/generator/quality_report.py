from __future__ import annotations

import argparse
import json

from toolmisusebench.generator import analyze_dataset
from toolmisusebench.version2 import analyze_dataset_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m generator.quality_report")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", default="", help="Comma-separated splits, e.g. train,dev,test_public")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    parser.add_argument("--v2", action="store_true", help="Use v2 expanded quality diagnostics")
    args = parser.parse_args(argv)

    splits = [part.strip() for part in args.splits.split(",") if part.strip()] or None
    payload = analyze_dataset_v2(args.dataset, splits) if args.v2 else analyze_dataset(args.dataset, splits)

    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
