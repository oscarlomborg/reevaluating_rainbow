
import argparse
import csv
from pathlib import Path
import numpy as np

COLUMNS = [
    "env_type", "setup", "experiment",
    "environment", "seed", "frame", "mean_return"
]


def iter_rows(path):
    parts = path.parts
    i = len(parts) - 1 - parts[::-1].index("experiments")

    env_type, setup, experiment, environment, seed_folder = parts[i+1:i+6]

    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"]
        means = data["mean_returns"]

        if len(frames) != len(means):
            raise ValueError(f"Mismatched array lengths: {path}")

        for frame, value in zip(frames, means):
            yield {
                "env_type": env_type,
                "setup": setup,
                "experiment": experiment,
                "environment": environment,
                "seed": seed_folder.removeprefix("seed_"),
                "frame": int(frame),
                "mean_return": float(value),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/minatar/setup1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/minatar/setup1/mean_returns.csv")
    )
    args = parser.parse_args()

    files = sorted(args.root.rglob("evaluation.npz"))

    if not files:
        raise SystemExit("No evaluation.npz files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for path in files:
            for row in iter_rows(path):
                writer.writerow(row)
                count += 1

    print(f"Exported {count:,} rows to {args.output}")


if __name__ == "__main__":
    main()

