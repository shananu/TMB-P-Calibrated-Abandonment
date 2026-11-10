from __future__ import annotations

import argparse
import json

from toolmisusebench.generator import SIZE_PROFILES, generate_dataset
from toolmisusebench.version2.generator import V2_SIZE_PROFILES, generate_dataset_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m generator.generate")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--size-profile", default="default", choices=sorted(set(SIZE_PROFILES) | set(V2_SIZE_PROFILES)))
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--dev-size", type=int)
    parser.add_argument("--test-size", type=int)
    args = parser.parse_args(argv)

    split_overrides = None
    if args.train_size is not None or args.dev_size is not None or args.test_size is not None:
        if args.train_size is None or args.dev_size is None or args.test_size is None:
            raise SystemExit("set all of --train-size, --dev-size, and --test-size together")
        split_overrides = {
            "train": args.train_size,
            "dev": args.dev_size,
            "test_public": args.test_size,
        }

    if args.version in {"v0.2", "v2", "version2"}:
        payload = generate_dataset_v2(
            version="v0.2",
            out_dir=args.out,
            seed=args.seed,
            split_sizes=split_overrides,
            size_profile=args.size_profile,
        )
    else:
        payload = generate_dataset(
            version=args.version,
            out_dir=args.out,
            seed=args.seed,
            write_parquet=not args.no_parquet,
            split_sizes=split_overrides,
            size_profile=args.size_profile,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
