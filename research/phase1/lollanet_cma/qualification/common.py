from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def normalized_local_checkpoint(charter_path: Path, value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = charter_path.resolve().parent / path
    return path.resolve()


def runtime_code_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = sorted(
        [*root.glob("*.py"), *[path for path in (root / "pyproject.toml",) if path.is_file()]],
        key=lambda path: path.name,
    )
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    if not hashes:
        raise ValueError(f"qualification code root contains no execution files: {root}")
    return {"root": str(root), "files": hashes, "sha256": stable_hash(hashes)}


def runtime_environment() -> dict[str, Any]:
    import torch
    value = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "accelerate": version("accelerate"),
        "bitsandbytes": version("bitsandbytes"),
        "huggingface_hub": version("huggingface-hub"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    return {"values": value, "sha256": stable_hash(value)}


def dependency_lock() -> dict[str, str]:
    path = Path(__file__).with_name("uv.lock").resolve()
    return {"path": str(path), "sha256": sha256_file(path)}
