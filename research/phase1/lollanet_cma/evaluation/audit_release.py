from __future__ import annotations

import argparse
from pathlib import Path

from .preflight import audit_sealed_release_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit readiness to unseal LollaNet-CMA evaluation data.")
    parser.add_argument(
        "--charter", type=Path, default=Path(__file__).with_name("sealed_release_charter.yaml"),
    )
    parser.add_argument("--stage0-bundle", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_sealed_release_preflight(
        args.charter, args.stage0_bundle, args.development_manifest, args.output,
    )
    print(result["sealed_release_gate_status"])


if __name__ == "__main__":
    main()
