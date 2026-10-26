from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lollanet_db.domain import DomainError, validate_change, validate_version  # noqa: E402
from lollanet_db.legacy_import import load_legacy_library  # noqa: E402
from lollanet_db.models import ModelRef  # noqa: E402


@pytest.fixture(scope="module")
def legacy():
    return load_legacy_library(ROOT)


def _future(base, model_id: str, version_no: int, status: str):
    claims = tuple(replace(item, legacy_key=None) for item in base.claims)
    evidence = tuple(replace(item, legacy_key=None) for item in base.evidence)
    digest = hashlib.sha256(f"{model_id}@{version_no}:{status}".encode()).hexdigest()
    return replace(
        base,
        ref=ModelRef(model_id, version_no),
        status=status,
        name=f"Test {model_id}",
        english_name=f"Test {model_id}",
        content_hash=digest,
        claims=claims,
        evidence=evidence,
        relations=(),
        legacy_source_path=None,
    )


def test_legacy_import_audit_matches_public_library(legacy):
    library, audit = legacy
    assert audit.source_files == 88
    assert audit.models == 88
    assert audit.versions == 88
    assert audit.statuses == {
        "candidate": 58,
        "active": 28,
        "contested": 2,
        "deprecated": 0,
    }
    assert audit.claims == 792
    assert audit.evidence == audit.claim_evidence == 266
    assert audit.formal_relations == 2
    assert audit.discovery_hints_skipped == 278
    assert all(version.legacy_source_path.startswith("01-Models/") for version in library.versions)
    assert not any("private" in version.legacy_source_path.lower() for version in library.versions)


def test_every_imported_version_has_typed_model_contract(legacy):
    library, _ = legacy
    for version in library.versions:
        validate_version(version)
        assert {claim.kind for claim in version.claims} == {
            "definition",
            "mechanism",
            "input",
            "transformation",
            "output",
            "premise",
            "boundary",
            "falsifier",
            "prediction",
        }


def test_domain_rejects_empty_core_claim(legacy):
    base = legacy[0].versions[0]
    invalid = replace(base, claims=tuple(item for item in base.claims if item.kind != "mechanism"))
    with pytest.raises(DomainError, match="mechanism"):
        validate_version(invalid)


def test_domain_validates_all_change_shapes(legacy):
    candidates = [version for version in legacy[0].versions if version.status == "candidate"]
    first, second = candidates[:2]
    new = _future(first, "mm-test-new", 1, "candidate")
    validate_change("new", [], {}, [new])
    amended = _future(first, first.ref.model_id, first.ref.version_no + 1, "candidate")
    validate_change("amend", [first.ref], {first.ref: first.status}, [amended])
    contested = _future(first, first.ref.model_id, first.ref.version_no + 1, "contested")
    validate_change("contest", [first.ref], {first.ref: first.status}, [contested])
    deprecated = _future(first, first.ref.model_id, first.ref.version_no + 1, "deprecated")
    validate_change("deprecate", [first.ref], {first.ref: first.status}, [deprecated])
    split_a = _future(first, "mm-test-split-a", 1, "candidate")
    split_b = _future(first, "mm-test-split-b", 1, "candidate")
    validate_change("split", [first.ref], {first.ref: first.status}, [deprecated, split_a, split_b])
    second_deprecated = _future(second, second.ref.model_id, second.ref.version_no + 1, "deprecated")
    merged = _future(first, "mm-test-merged", 1, "candidate")
    validate_change(
        "merge",
        [first.ref, second.ref],
        {first.ref: first.status, second.ref: second.status},
        [deprecated, second_deprecated, merged],
    )


def test_change_shapes_reject_implicit_or_partial_evolution(legacy):
    first = next(version for version in legacy[0].versions if version.status == "candidate")
    successor = _future(first, first.ref.model_id, first.ref.version_no + 1, "candidate")
    with pytest.raises(DomainError, match="new requires"):
        validate_change("new", [first.ref], {first.ref: first.status}, [successor])
    with pytest.raises(DomainError, match="successor"):
        validate_change("deprecate", [first.ref], {first.ref: first.status}, [])


def test_change_rejects_disallowed_status_transition(legacy):
    active = next(version for version in legacy[0].versions if version.status == "active")
    demoted = _future(active, active.ref.model_id, active.ref.version_no + 1, "candidate")
    with pytest.raises(DomainError, match="invalid status transition"):
        validate_change("amend", [active.ref], {active.ref: active.status}, [demoted])

    deprecated = replace(active, status="deprecated")
    contested = _future(active, active.ref.model_id, active.ref.version_no + 1, "contested")
    with pytest.raises(DomainError, match="invalid status transition"):
        validate_change("contest", [deprecated.ref], {deprecated.ref: deprecated.status}, [contested])


@pytest.mark.parametrize(
    ("script", "arguments", "forbidden_paths"),
    [
        (
            "create_model.py",
            ["Test", "--proposal", "mp-test", "--root", "{root}"],
            ["01-Models"],
        ),
        (
            "generate_model_index.py",
            ["--root", "{root}", "--apply"],
            ["00-Start/Model Index.md"],
        ),
        (
            "model_graph.py",
            ["--root", "{root}", "build", "--apply"],
            ["05-Maps/model-graph.json", "05-Maps/Model Graph.md"],
        ),
    ],
)
def test_retired_legacy_write_clis_fail_closed(tmp_path, script, arguments, forbidden_paths):
    command = [
        sys.executable,
        str(ROOT / "06-Scripts" / script),
        *(argument.format(root=tmp_path) for argument in arguments),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "retired" in result.stderr
    assert not any((tmp_path / path).exists() for path in forbidden_paths)
