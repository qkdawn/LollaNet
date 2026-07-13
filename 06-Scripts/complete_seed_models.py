"""Read-only audit for the retired seed-card bulk writer.

The original script regenerated every public model card from embedded copy. That
workflow bypassed review, erased manual revisions, and mixed personal usage
notes into the public framework. Model changes now go through proposals and
reviewed, card-by-card edits.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_NAMES = frozenset(
    {
        "生态位",
        "代理问题",
        "规模经济",
        "护城河",
        "网络效应",
        "复利",
        "机会成本",
        "安全边际",
        "反馈回路",
        "瓶颈",
        "冗余",
        "路径依赖",
        "反向思考",
        "能力圈",
        "基础概率",
        "非线性",
        "均值回归",
        "贝叶斯更新",
        "临界点",
        "期望值",
        "过度自信",
        "社会认同",
        "确认偏误",
        "激励机制",
        "激励导致的偏见",
        "权威偏差",
        "损失厌恶",
        "承诺一致性",
        "嫉妒与比较",
        "可得性偏差",
    }
)


def audit_seed_models(vault_root: Path) -> list[str]:
    """Return missing-seed errors without rejecting later accepted models."""
    model_root = Path(vault_root) / "01-Models"
    names = {path.stem for path in model_root.rglob("*.md")} if model_root.exists() else set()
    return [f"missing seed model: {name}" for name in sorted(EXPECTED_MODEL_NAMES - names)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the 30 legacy seed cards. Bulk rewriting is permanently disabled."
    )
    parser.add_argument("--root", default=str(ROOT), help="Vault root path")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Refused: retained only to explain how the retired command changed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rewrite:
        print("Refusing to rewrite model cards. Submit and approve a model-change proposal instead.")
        return 2

    errors = audit_seed_models(Path(args.root))
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Seed inventory audit failed with {len(errors)} error(s).")
        return 1

    print("Seed inventory is complete. Bulk rewriting is retired; no files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
