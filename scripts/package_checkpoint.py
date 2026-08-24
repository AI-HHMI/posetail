#!/usr/bin/env python3
"""Package an inference-ready posetail checkpoint.

The training checkpoints contain optimizer state and both raw and schedule-free averaged
weights. This script keeps only the averaged inference weights and the model constructor
configuration, reducing the artifact substantially while leaving it self-contained.

Run through the project environment, for example:

    pixi run python scripts/package_checkpoint.py \
        --run-dir /path/to/wandb/run-20260727_113457-ukt14i7c \
        --output /path/to/posetail-models/posetail-static-animal/model.pth
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import toml
import torch


_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth$")


def _resolve_run_files(run_dir: Path) -> tuple[Path, Path]:
    """Return ``(config.toml, checkpoints directory)`` for a W&B run directory."""
    run_dir = run_dir.expanduser().resolve()

    candidates = (
        (run_dir / "files" / "config.toml", run_dir / "files" / "checkpoints"),
        (run_dir / "config.toml", run_dir / "checkpoints"),
    )
    for config_path, checkpoint_dir in candidates:
        if config_path.is_file() and checkpoint_dir.is_dir():
            return config_path, checkpoint_dir

    raise FileNotFoundError(
        f"Could not find config.toml and checkpoints/ below W&B run directory {run_dir}. "
        "Expected either <run>/files/config.toml + <run>/files/checkpoints/ or the "
        "equivalent paths directly below <run>."
    )


def _checkpoint_iteration(path: Path) -> int:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a posetail checkpoint filename: {path}")
    return int(match.group(1))


def resolve_checkpoint(checkpoint_dir: Path, checkpoint: str | int | None = None) -> Path:
    """Resolve a checkpoint number/name, defaulting to the numerically latest checkpoint."""
    if checkpoint is not None:
        checkpoint_text = str(checkpoint)
        if checkpoint_text.endswith(".pth"):
            candidate = checkpoint_dir / Path(checkpoint_text).name
        else:
            candidate = checkpoint_dir / f"checkpoint_{int(checkpoint):08d}.pth"
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    checkpoints = [
        path for path in checkpoint_dir.glob("checkpoint_*.pth")
        if _CHECKPOINT_RE.fullmatch(path.name)
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pth files found in {checkpoint_dir}")
    return max(checkpoints, key=_checkpoint_iteration)


def package_checkpoint(
    run_dir: str | Path,
    output: str | Path,
    checkpoint: str | int | None = None,
) -> Path:
    """Write an inference-only checkpoint and return its output path.

    The output contains the schedule-free averaged weights as ``model_state`` and the
    complete ``[model]`` TOML section as ``model_config``. Training and optimizer state are
    deliberately omitted.
    """
    run_dir = Path(run_dir)
    output = Path(output).expanduser().resolve()
    config_path, checkpoint_dir = _resolve_run_files(run_dir)
    checkpoint_path = resolve_checkpoint(checkpoint_dir, checkpoint)

    config = toml.load(config_path)
    model_config = config.get("model")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"{config_path} does not contain a non-empty [model] section")

    print(f"Loading {checkpoint_path} ...", flush=True)
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError(f"Training checkpoint must be a dictionary, got {type(source).__name__}")

    eval_state = source.get("model_state_eval")
    if not isinstance(eval_state, dict) or not eval_state:
        raise ValueError(
            f"{checkpoint_path} does not contain model_state_eval; refusing to package raw "
            "training weights as inference weights"
        )

    # map_location='cpu' already places tensors on CPU. Rebuild the mapping so no optimizer
    # state or other objects from the 5+ GB source checkpoint can be retained by the output.
    model_state = {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in eval_state.items()
    }
    iteration = int(source.get("iteration", _checkpoint_iteration(checkpoint_path)))
    source_run = config.get("wandb", {}).get("run_id") or run_dir.name

    packaged: dict[str, Any] = {
        "format_version": 1,
        "model_state": model_state,
        "model_config": model_config,
        "iteration": iteration,
        "source_run": source_run,
        "source_checkpoint": checkpoint_path.name,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output} ...", flush=True)
    torch.save(packaged, output)
    print(
        f"Packaged iteration {iteration} ({len(model_state)} tensors) "
        f"to {output} ({output.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="W&B run directory containing files/config.toml and files/checkpoints/",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .pth path",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint iteration or filename; defaults to the numerically latest checkpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package_checkpoint(args.run_dir, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
