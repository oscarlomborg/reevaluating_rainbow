#!/usr/bin/env python3
"""Convert Dopamine legacy log_* pickles to evaluation.npz for evaluate.py.

Input, for example:
    runs/seaquest/rainbow/seed_0/logs/log_1
Output, for example:
    experiments/minatar/setup1/rainbow/MinAtar_Seaquest-v0/seed_0/evaluation.npz

A Dopamine log_<N> is usually a cumulative dictionary of completed iterations.
This script merges all readable snapshots; it can run while training continues.
It never modifies the source log files.

Only unpickle logs produced by code you trust. Python pickle is not secure
against untrusted files.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

GAMES = {
    "asterix": "MinAtar_Asterix-v0",
    "breakout": "MinAtar_Breakout-v0",
    "freeway": "MinAtar_Freeway-v0",
    "seaquest": "MinAtar_Seaquest-v0",
    "invaders": "MinAtar_SpaceInvaders-v0",
}

ITER_RE = re.compile(r"^(?:iteration_)?(\d+)$")
LOG_RE = re.compile(r"^log_(\d+)$")


def text_key(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_scalar(value):
    """Dopamine stores a list of values for each statistic in an iteration."""
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("empty statistic")
    return float(arr.reshape(-1)[-1])


def log_files(log_directory: Path):
    files = []
    for path in log_directory.glob("log_*"):
        match = LOG_RE.fullmatch(path.name)
        if match and path.is_file():
            files.append((int(match.group(1)), path))
    return [path for _, path in sorted(files)]


def read_evaluations(log_directory: Path):
    """Read all saved iterations, keeping newer snapshots where duplicated.

    Returns a mapping from zero-based iteration number to evaluation return.
    """
    paths = log_files(log_directory)
    if not paths:
        raise FileNotFoundError(f"No log_N files in {log_directory}")

    by_iteration = {}
    for path in paths:
        try:
            with path.open("rb") as stream:
                log = pickle.load(stream)
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            # A file may be in the process of being written; earlier snapshots
            # may still provide valid, completed iteration results.
            print(f"[warn] Cannot read {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(log, dict):
            print(f"[warn] {path}: expected a dict, got {type(log).__name__}",
                  file=sys.stderr)
            continue

        for key, statistics in log.items():
            match = ITER_RE.fullmatch(text_key(key))
            if not match:
                continue
            if not isinstance(statistics, dict):
                print(f"[warn] {path}: {key} is not a statistics dictionary",
                      file=sys.stderr)
                continue
            normalized = {text_key(k): v for k, v in statistics.items()}
            if "eval_average_return" not in normalized:
                continue
            try:
                value = get_scalar(normalized["eval_average_return"])
                if not np.isfinite(value):
                    raise ValueError(f"non-finite return: {value}")
            except (ValueError, TypeError) as exc:
                print(f"[warn] {path}: {key}: {exc}", file=sys.stderr)
                continue
            by_iteration[int(match.group(1))] = value

    if not by_iteration:
        raise ValueError(
            f"No valid iteration_N/eval_average_return entries in {log_directory}. "
            "Check that your Dopamine version writes the legacy pickle logs."
        )
    return by_iteration


def write_npz(destination: Path, frames: np.ndarray, mean_returns: np.ndarray):
    """Replace only evaluation.npz, atomically (safe during ongoing training)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=".evaluation-",
            dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, frames=frames, mean_returns=mean_returns)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert_one(source_seed: Path, output_root: Path, game: str,
                ablation: str, steps_per_iteration: int, dry_run: bool):
    entries = read_evaluations(source_seed / "logs")
    iterations = sorted(entries)
    # Nominal budgets match evaluate.py's exact 1M-step milestones. Dopamine
    # completes whole episodes, so actual environment steps can exceed them.
    frames = np.array(
        [(iteration + 1) * steps_per_iteration for iteration in iterations],
        dtype=np.int64,
    )
    mean_returns = np.array(
        [entries[iteration] for iteration in iterations], dtype=np.float64
    )
    destination = output_root / ablation / GAMES[game] / source_seed.name / "evaluation.npz"
    if not dry_run:
        write_npz(destination, frames, mean_returns)
    print(f"{'[dry-run] ' if dry_run else '[saved] '}{destination} | "
          f"{len(frames)} evaluations | last step {frames[-1]:,} | "
          f"last return {mean_returns[-1]:.3f}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("runs"),
                        help="Root containing <env>/<ablation>/seed_<N>/logs/log_* files")
    parser.add_argument("--output-root", type=Path,
                        default=Path("experiments/minatar/dopamine"),
                        help="Directory to create <ablation>/<game>/seed_<N>/evaluation.npz")
    parser.add_argument("--steps-per-iteration", type=int, default=1_000_000,
                        help="Runner.training_steps used in each iteration (default 1000000)")
    parser.add_argument("--env", nargs="+", choices=GAMES, default=None,
                        help="Optional environment filter, e.g. --env seaquest breakout")
    parser.add_argument("--ablation", nargs="+", default=None,
                        help="Optional ablation filter, e.g. --ablation rainbow no_per")
    parser.add_argument("--seed", nargs="+", type=int, default=None,
                        help="Optional seed filter, e.g. --seed 0 1 2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned outputs without creating files")
    args = parser.parse_args()

    if args.steps_per_iteration <= 0:
        parser.error("--steps-per-iteration must be greater than zero")
    if not args.input_root.is_dir():
        parser.error(f"Input directory does not exist: {args.input_root}")

    environments = args.env if args.env is not None else list(GAMES)
    converted = 0
    failed = 0
    for game in environments:
        game_dir = args.input_root / game
        if not game_dir.is_dir():
            continue
        for ablation_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            ablation = ablation_dir.name
            if args.ablation is not None and ablation not in args.ablation:
                continue
            for seed_dir in sorted(p for p in ablation_dir.glob("seed_*") if p.is_dir()):
                try:
                    seed_number = int(seed_dir.name.removeprefix("seed_"))
                except ValueError:
                    continue
                if args.seed is not None and seed_number not in args.seed:
                    continue
                if not (seed_dir / "logs").is_dir():
                    continue
                try:
                    convert_one(seed_dir, args.output_root, game, ablation,
                                args.steps_per_iteration, args.dry_run)
                    converted += 1
                except (FileNotFoundError, ValueError, OSError) as exc:
                    print(f"[skip] {seed_dir}: {exc}", file=sys.stderr)
                    failed += 1
    print(f"Finished: {converted} converted, {failed} skipped/failed.")
    if converted == 0:
        parser.exit(1, "No evaluation files created. Check paths and log format.\n")


if __name__ == "__main__":
    main()
