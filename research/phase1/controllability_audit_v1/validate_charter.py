from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_charter, preflight_open_data, validate_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen causal-controllability audit charter.")
    parser.add_argument("charter", type=Path)
    parser.add_argument("--inventory-open-data", action="store_true")
    args = parser.parse_args()

    charter = load_charter(args.charter)
    validate_charter(charter)
    output = {"status": "valid", "study_id": charter.get("study_id")}
    if args.inventory_open_data:
        output["open_data_preflight"] = preflight_open_data(charter)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

