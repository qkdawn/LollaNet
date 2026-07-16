#!/usr/bin/env python3
"""Create or verify a mapping-free delivery bundle for one frozen blind-review wave.

The bundle contains only the frozen blind packet, a reviewer instruction, and a
small delivery manifest. It intentionally excludes the randomization mapping.
Creating or verifying a bundle does not make the recipient independent, does
not perform a review, and does not change any model state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "private" / "evaluations" / "model-test-v2-2026-07-16" / "28-independent-review-integrity-manifest.json"
BUNDLE_MANIFEST = "delivery-manifest.json"
PACKET_NAME = "blind-packet.md"
INSTRUCTIONS_NAME = "README.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"manifest path must be relative: {raw_path}")
    resolved_base = base.resolve()
    resolved_path = (base / path).resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise ValueError(f"manifest path escapes its directory: {raw_path}")
    return resolved_path


def read_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read integrity manifest: {exc}"]
    if manifest.get("type") != "independent_review_integrity_manifest":
        errors.append("unexpected integrity manifest type")
    if not isinstance(manifest.get("campaign"), str) or not manifest["campaign"]:
        errors.append("integrity manifest missing campaign")
    if not isinstance(manifest.get("waves"), list):
        errors.append("integrity manifest must contain waves list")
    return manifest, errors


def select_wave(manifest_path: Path, wave_name: str) -> tuple[dict[str, Any] | None, list[str]]:
    manifest, errors = read_manifest(manifest_path)
    if manifest is None:
        return None, errors
    matches = [wave for wave in manifest.get("waves", []) if isinstance(wave, dict) and wave.get("wave") == wave_name]
    if len(matches) != 1:
        errors.append(f"expected exactly one wave named {wave_name!r}, found {len(matches)}")
        return None, errors
    wave = matches[0]
    for key in ("blind_packet_path", "blind_packet_sha256", "packet_bytes"):
        if key not in wave:
            errors.append(f"{wave_name}: missing {key}")
    if not isinstance(wave.get("blind_packet_sha256"), str) or len(wave.get("blind_packet_sha256", "")) != 64:
        errors.append(f"{wave_name}: invalid blind_packet_sha256")
    if not isinstance(wave.get("packet_bytes"), int) or wave.get("packet_bytes", -1) < 0:
        errors.append(f"{wave_name}: invalid packet_bytes")
    if not isinstance(wave.get("blind_packet_path"), str) or not wave.get("blind_packet_path"):
        errors.append(f"{wave_name}: invalid blind_packet_path")
    return wave, errors


def reviewer_instructions(campaign: str, wave: str, packet_sha256: str) -> str:
    return f"""# Independent blind-review delivery — {wave}

This bundle is a **frozen blind packet only** for campaign `{campaign}`. Its packet SHA-256 is:

```text
{packet_sha256}
```

## Before reviewing

1. Verify the hash of `blind-packet.md` against the value above.
2. Do not seek, infer, or request the hidden A/B randomization mapping, model-card identities, previous results, or reviewer conclusions.
3. Do not use private user data, real task data, or information outside the packet that would reveal the mapping.
4. If you have previously authored LollaNet cards, generated the task/answers, reviewed this wave, seen the mapping, or have a conflict, do not score it as independent; disclose that to the coordinator.

## Submit

Score every unit in `blind-packet.md`, preserve ties, invalid runs, abstentions and uncertainty, then submit one frozen JSON record using the private contract:

```text
61-independent-blind-review-submission-contract.md
```

The coordinator, not the reviewer, controls any later unblinding. A completed form or a passing format check does not itself prove independence, correctness, model efficacy, or authorize model changes.
"""


def prepare_bundle(manifest_path: Path, wave_name: str, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    wave, errors = select_wave(manifest_path, wave_name)
    if wave is None or errors:
        return {"output_dir": str(output_dir), "passed": False, "errors": errors}
    integrity_manifest, manifest_errors = read_manifest(manifest_path)
    errors.extend(manifest_errors)
    if errors or integrity_manifest is None:
        return {"output_dir": str(output_dir), "passed": False, "errors": errors}
    try:
        source_packet = safe_relative_path(manifest_path.parent, wave["blind_packet_path"])
    except ValueError as exc:
        return {"output_dir": str(output_dir), "passed": False, "errors": [str(exc)]}
    if not source_packet.is_file():
        return {"output_dir": str(output_dir), "passed": False, "errors": [f"blind packet missing: {source_packet}"]}
    actual_hash = sha256(source_packet)
    actual_bytes = source_packet.stat().st_size
    if actual_hash != wave["blind_packet_sha256"]:
        errors.append("frozen blind packet SHA-256 mismatch")
    if actual_bytes != wave["packet_bytes"]:
        errors.append("frozen blind packet byte-count mismatch")
    if errors:
        return {"output_dir": str(output_dir), "passed": False, "errors": errors}

    output_dir = output_dir.resolve()
    if output_dir.exists():
        return {"output_dir": str(output_dir), "passed": False, "errors": [f"output directory already exists: {output_dir}"]}
    try:
        output_dir.mkdir(parents=True)
        packet_path = output_dir / PACKET_NAME
        shutil.copyfile(source_packet, packet_path)
        (output_dir / INSTRUCTIONS_NAME).write_text(
            reviewer_instructions(integrity_manifest["campaign"], wave_name, actual_hash), encoding="utf-8"
        )
        delivery_manifest = {
            "type": "blind_review_delivery_manifest",
            "campaign": integrity_manifest["campaign"],
            "wave": wave_name,
            "privacy": "private",
            "purpose": "mapping-free reviewer delivery; not evidence of completed independent review",
            "artifacts": [
                {"path": PACKET_NAME, "sha256": actual_hash, "bytes": actual_bytes},
                {"path": INSTRUCTIONS_NAME, "sha256": sha256(output_dir / INSTRUCTIONS_NAME), "bytes": (output_dir / INSTRUCTIONS_NAME).stat().st_size},
            ],
            "prohibited_artifacts": ["randomization mappings", "model cards", "previous review results", "private user data"],
        }
        (output_dir / BUNDLE_MANIFEST).write_text(json.dumps(delivery_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return {"output_dir": str(output_dir), "passed": False, "errors": [f"cannot create delivery bundle: {exc}"]}
    return verify_bundle(output_dir)


def verify_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    errors: list[str] = []
    manifest_path = output_dir / BUNDLE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"output_dir": str(output_dir), "passed": False, "errors": [f"cannot read delivery manifest: {exc}"]}
    if manifest.get("type") != "blind_review_delivery_manifest":
        errors.append("unexpected delivery manifest type")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        errors.append("delivery manifest must describe exactly two artifacts")
        artifacts = []
    expected_names = {PACKET_NAME, INSTRUCTIONS_NAME}
    names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("invalid delivery artifact entry")
            continue
        name = item["path"]
        names.add(name)
        if name not in expected_names or Path(name).name != name:
            errors.append(f"unexpected delivery artifact path: {name}")
            continue
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing delivery artifact: {name}")
            continue
        if sha256(path) != item.get("sha256"):
            errors.append(f"delivery artifact SHA-256 mismatch: {name}")
        if path.stat().st_size != item.get("bytes"):
            errors.append(f"delivery artifact byte-count mismatch: {name}")
    if names != expected_names:
        errors.append("delivery manifest artifact set differs from required packet and instructions")
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_names | {BUNDLE_MANIFEST}:
        errors.append("delivery bundle contains unexpected files")
    forbidden = [name for name in actual_files if "mapping" in name.lower() or "randomization" in name.lower()]
    if forbidden:
        errors.append("delivery bundle contains a forbidden mapping-like filename")
    return {"output_dir": str(output_dir), "wave": manifest.get("wave"), "artifacts_checked": len(artifacts), "passed": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a mapping-free frozen blind-review delivery bundle.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Frozen blind-review integrity manifest.")
    parser.add_argument("--wave", help="Wave name to package; required unless --verify is used.")
    parser.add_argument("--output-dir", type=Path, help="New output directory for a delivery bundle, or bundle directory with --verify.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing delivery bundle instead of creating one.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args()
    if args.verify:
        if args.output_dir is None:
            parser.error("--verify requires --output-dir")
        report = verify_bundle(args.output_dir)
    else:
        if not args.wave or args.output_dir is None:
            parser.error("creation requires --wave and --output-dir")
        report = prepare_bundle(args.manifest, args.wave, args.output_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"PASS: mapping-free blind-review delivery bundle verified ({report['artifacts_checked']} artifacts).")
        print("Distribution and verification do not prove reviewer independence or model effectiveness.")
    else:
        print("FAIL: blind-review delivery bundle")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
