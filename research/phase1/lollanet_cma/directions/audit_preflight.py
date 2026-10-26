from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import audit_direction_preflight


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit readiness for LollaNet-CMA direction fitting.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("direction_charter.yaml"))
    parser.add_argument(
        "--qualification-charter", type=Path,
        default=root / "qualification" / "qualification_charter.yaml",
    )
    parser.add_argument("--access-view", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_direction_preflight(
        args.charter, args.qualification_charter, args.access_view, args.selection, args.output,
    )
    print(result["direction_gate_status"])


if __name__ == "__main__":
    main()
