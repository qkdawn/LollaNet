from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any

import yaml

from .common import atomic_write_json, normalized_local_checkpoint, sha256_file, stable_hash


REQUIRED_PATTERNS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _selected_files(siblings: list[Any]) -> list[Any]:
    from fnmatch import fnmatch

    return sorted(
        [item for item in siblings if any(fnmatch(item.rfilename, pattern) for pattern in REQUIRED_PATTERNS)],
        key=lambda item: item.rfilename,
    )


def _verify_snapshot(checkpoint: Path, siblings: list[Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for sibling in _selected_files(siblings):
        path = checkpoint / sibling.rfilename
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {sibling.rfilename}")
        size = path.stat().st_size
        if sibling.size is not None and size != int(sibling.size):
            raise ValueError(f"checkpoint file size mismatch: {sibling.rfilename}")
        actual_sha256 = sha256_file(path)
        lfs = sibling.lfs
        expected_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint LFS SHA-256 mismatch: {sibling.rfilename}")
        verified.append({
            "path": sibling.rfilename,
            "size": size,
            "sha256": actual_sha256,
            "lfs_sha256_expected": expected_sha256,
            "blob_id": sibling.blob_id,
        })
    names = {row["path"] for row in verified}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise ValueError("checkpoint metadata files are incomplete")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("checkpoint contains no safetensors weights")
    return verified


def _run_git(arguments: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, env=env, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git command failed: {detail}")
    return completed.stdout.strip()


def _selected_local_files(checkpoint: Path) -> list[Path]:
    return sorted(
        item for item in checkpoint.iterdir()
        if item.is_file() and any(__import__("fnmatch").fnmatch(item.name, pattern) for pattern in REQUIRED_PATTERNS)
    )


def _freeze_git_candidate(
    charter_path: Path, candidate: dict[str, Any], checkpoint: Path,
) -> dict[str, Any]:
    revision = str(candidate["revision"])
    repository_url = str(candidate.get("repository_url", ""))
    if not repository_url:
        raise ValueError("git-backed candidate is missing repository_url")
    clone_env = dict(os.environ)
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if checkpoint.exists() and any(checkpoint.iterdir()):
        if not (checkpoint / ".git").is_dir():
            raise ValueError(f"git checkpoint destination is not resumable: {checkpoint}")
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--no-checkout", repository_url, str(checkpoint)], env=clone_env)
    _run_git(["checkout", "--detach", revision], cwd=checkpoint, env=clone_env)
    _run_git(["lfs", "pull"], cwd=checkpoint)
    observed_revision = _run_git(["rev-parse", "HEAD"], cwd=checkpoint)
    if observed_revision != revision:
        raise ValueError(f"git repository returned a different revision: {observed_revision}")
    lfs_rows = _run_git(["lfs", "ls-files", "-l"], cwd=checkpoint).splitlines()
    lfs_sha_by_path: dict[str, str] = {}
    for row in lfs_rows:
        fields = row.split(maxsplit=2)
        if len(fields) == 3:
            lfs_sha_by_path[fields[2].lstrip("*- ")] = fields[0]
    selected = _selected_local_files(checkpoint)
    names = {item.name for item in selected}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise ValueError("checkpoint metadata files are incomplete")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("checkpoint contains no safetensors weights")
    files: list[dict[str, Any]] = []
    for path in selected:
        actual_sha256 = sha256_file(path)
        expected_sha256 = lfs_sha_by_path.get(path.name)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint LFS SHA-256 mismatch: {path.name}")
        files.append({
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": actual_sha256,
            "lfs_sha256_expected": expected_sha256,
            "blob_id": None,
        })
    (checkpoint / "REVISION").write_text(revision + "\n", encoding="utf-8")
    return {
        "source_provider": candidate["source_provider"],
        "repository_url": repository_url,
        "files": files,
    }


def _remove_git_transport_cache(checkpoint: Path) -> None:
    git_dir = checkpoint / ".git"
    if not git_dir.is_dir():
        return
    for item in git_dir.rglob("*"):
        try:
            os.chmod(item, 0o700)
        except OSError:
            pass
    shutil.rmtree(git_dir)


def freeze_candidate(charter_path: Path, candidate_id: str, endpoint: str, max_workers: int = 4) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    charter = yaml.safe_load(charter_path.read_text(encoding="utf-8"))
    candidates = {row["id"]: row for row in charter["candidates"]}
    if candidate_id not in candidates:
        raise ValueError(f"candidate is not preregistered: {candidate_id}")
    candidate = candidates[candidate_id]
    revision = candidate.get("revision")
    checkpoint = normalized_local_checkpoint(charter_path, candidate.get("local_checkpoint"))
    if not revision or checkpoint is None:
        raise ValueError("candidate revision and local checkpoint must be frozen before download")
    source_provider = candidate.get("source_provider", "huggingface")
    if source_provider == "modelscope_git_lfs":
        frozen = _freeze_git_candidate(charter_path, candidate, checkpoint)
        files = frozen["files"]
    elif source_provider == "huggingface":
        api = HfApi(endpoint=endpoint)
        info = api.model_info(candidate["model_id"], revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError(f"repository returned a different revision: {info.sha}")
        selected = _selected_files(info.siblings)
        if not selected:
            raise ValueError("repository returned no required checkpoint files")
        snapshot_download(
            repo_id=candidate["model_id"],
            revision=revision,
            endpoint=endpoint,
            local_dir=checkpoint,
            allow_patterns=list(REQUIRED_PATTERNS),
            max_workers=max_workers,
        )
        files = _verify_snapshot(checkpoint, info.siblings)
        (checkpoint / "REVISION").write_text(str(revision) + "\n", encoding="utf-8")
        frozen = {"source_provider": source_provider, "repository_url": endpoint}
    else:
        raise ValueError(f"unsupported candidate source provider: {source_provider}")
    manifest = {
        "type": "lollanet_cma_candidate_checkpoint_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "model_id": candidate["model_id"],
        "revision": revision,
        "endpoint": frozen["repository_url"] if source_provider == "modelscope_git_lfs" else endpoint,
        "source_provider": frozen["source_provider"],
        "repository_url": frozen["repository_url"],
        "checkpoint": str(checkpoint),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
        "charter_sha256": sha256_file(charter_path),
    }
    manifest["checkpoint_content_sha256"] = stable_hash(manifest)
    atomic_write_json(checkpoint / "checkpoint_manifest.json", manifest)
    if source_provider == "modelscope_git_lfs":
        _remove_git_transport_cache(checkpoint)
    return manifest
