from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_config
from .model_runner import run_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline or oracle linear steering on LollaSteer-Q prompts.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pilot.yaml"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("baseline", "prompt_only", "caa", "low_rank"), default="baseline")
    parser.add_argument("--directions", type=Path)
    parser.add_argument("--factor", choices=("reward", "social", "certainty"))
    parser.add_argument("--layer", type=int)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("all", "discovery", "pilot_holdout"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    model = config["model"]
    report = run_model(
        prompts_path=args.prompts,
        output_dir=args.output,
        checkpoint=model["primary_checkpoint"],
        fallback_checkpoint=model.get("fallback_checkpoint"),
        fallback_policy=model["fallback_policy"],
        layer_fractions=model["layer_fractions"],
        max_new_tokens=int(model["max_new_tokens"]),
        dtype_name=model["dtype"],
        direction_path=args.directions,
        method=args.method,
        selected_factor=args.factor,
        selected_layer=args.layer,
        strength=args.strength,
        rank=args.rank,
        limit=args.limit,
        checkpoint_revision=model.get("checkpoint_revision"),
        local_checkpoint=model.get("local_checkpoint"),
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
