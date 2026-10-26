from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import audit_development_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LollaNet-CMA development-selection readiness.")
    parser.add_argument("--charter", type=Path, default=Path(__file__).with_name("development_charter.yaml"))
    parser.add_argument("--access-view", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_development_preflight(
        args.charter, args.access_view, args.direction_manifest, args.output,
    )
    print(result["development_gate_status"])


if __name__ == "__main__":
    main()

