from __future__ import annotations

import argparse
import json
from pathlib import Path

from .steering import fit_directions


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit paired CAA and low-rank direction bundles from discovery activations.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    report = fit_directions(args.prompts, args.activations, args.run_manifest, args.output, tuple(args.ranks))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
