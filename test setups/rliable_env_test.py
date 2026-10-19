# %%
"""Per-environment analysis of Rainbow / DQN evaluation results.

For each selected environment, generates:
  1. Mean validation return over training with a pointwise, non-bootstrapped,
     normal-approximation 90% confidence interval across seeds.
  2. Final per-seed validation returns and final mean/median/IQM statistics.
  3. A per-environment empirical performance profile (final return survival).

No metrics are aggregated across environments; no bootstrap resampling is used.
Dependencies: numpy, matplotlib (rliable and scipy are not required).

Run from the repository root:
    python rliable_per_environment.py

NOTE: evaluation.npz files are only produced once EVAL_EVERY is reached.
"""

import csv
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

# %%
# ============================================================
# Configuration: update these values to match your experiment
# ============================================================

EXPERIMENT_ROOT = Path("./experiments/minatar_rainbow-test")
OUTPUT_DIR = Path("./analysis_plots")

GAMES = [
    "MinAtar_Breakout-v1",
    "MinAtar_Asterix-v1",
    # "MinAtar_Freeway-v1",
    # "MinAtar_Seaquest-v1",
    # "MinAtar_SpaceInvaders-v1",
]

SEEDS = list(range(10))
ALGORITHM_NAME = "Rainbow"
SHOW_PLOTS = True  # Set False for non-interactive / HPC runs

# Two-sided 90% NORMAL confidence interval: z_(0.95) = 1.64485...
# The interval is mean +/- z * sample_standard_deviation / sqrt(num_seeds).
CONFIDENCE_LEVEL = 0.90
Z_CRITICAL = NormalDist().inv_cdf((1.0 + CONFIDENCE_LEVEL) / 2.0)


# %%
# ============================================================
# Helpers
# ============================================================

def load_evaluation_file(game: str, seed: int):
    path = EXPERIMENT_ROOT / game / f"seed_{seed}" / "evaluation.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Has the seed reached its first EVAL_EVERY step?"
        )

    # Only plain numeric arrays are needed; no pickled objects are loaded.
    with np.load(path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"], dtype=np.int64)
        mean_returns = np.asarray(data["mean_returns"], dtype=np.float64)

    if frames.ndim != 1 or mean_returns.ndim != 1:
        raise ValueError(f"Expected 1-D frames and mean_returns in {path}")
    if len(frames) == 0 or frames.shape != mean_returns.shape:
        raise ValueError(f"Missing or inconsistent evaluation points in {path}")
    if not np.isfinite(mean_returns).all():
        raise ValueError(f"Nonfinite validation mean in {path}")
    return frames, mean_returns


def load_game_scores(game: str):
    """Return frames (evaluations,) and returns (seeds, evaluations)."""
    reference_frames = None
    per_seed = []
    for seed in SEEDS:
        frames, returns = load_evaluation_file(game, seed)
        if reference_frames is None:
            reference_frames = frames
        elif not np.array_equal(frames, reference_frames):
            raise ValueError(
                f"Evaluation frames don't match within {game}, seed {seed}. "
                "Compare the eval schedule and whether all seeds have finished."
            )
        per_seed.append(returns)

    if len(per_seed) < 2:
        raise ValueError("At least 2 seeds are required for a confidence interval.")
    return reference_frames, np.stack(per_seed, axis=0)


def mean_ci90(scores: np.ndarray):
    """Compute pointwise normal-approximation 90% CI across seeds (axis 0)."""
    n = scores.shape[0]
    if n < 2:
        raise ValueError("At least 2 seeds are required for a confidence interval.")
    mean = np.mean(scores, axis=0)
    sample_sd = np.std(scores, axis=0, ddof=1)
    margin = Z_CRITICAL * sample_sd / np.sqrt(n)
    return mean, mean - margin, mean + margin, sample_sd


def interquartile_mean(values: np.ndarray):
    """Exact 25%-trimmed mean of an empirical sample (fractional edges)."""
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = len(x)
    if n == 0:
        raise ValueError("Cannot calculate IQM for an empty sample.")
    start, stop = 0.25 * n, 0.75 * n
    edges = np.arange(n + 1, dtype=float)
    weights = np.maximum(0.0, np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start))
    return float(np.dot(weights, x) / (stop - start))


def save_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def analyze_game(game: str):
    frames, scores = load_game_scores(game)
    n_seeds, n_eval = scores.shape
    means, lows, highs, stds = mean_ci90(scores)
    final_scores = scores[:, -1]
    final_mean, final_low, final_high, final_std = mean_ci90(final_scores)
    final_mean = float(final_mean)
    final_low = float(final_low)
    final_high = float(final_high)
    final_std = float(final_std)
    final_median = float(np.median(final_scores))
    final_iqm = interquartile_mean(final_scores)

    game_dir = OUTPUT_DIR / game
    game_dir.mkdir(parents=True, exist_ok=True)

    # Print independent statistics for this environment only.
    print(f"\n{'=' * 66}")
    print(f"{ALGORITHM_NAME} | {game}")
    print(f"Seeds: {n_seeds} | Evaluations per seed: {n_eval}")
    print(f"Final evaluation step: {frames[-1]:,}")
    print(f"Final mean validation return: {final_mean:.4f}")
    print(f"90% normal CI of the mean: [{final_low:.4f}, {final_high:.4f}]")
    print(f"Final sample standard deviation: {final_std:.4f}")
    print(f"Final median: {final_median:.4f}")
    print(f"Final IQM: {final_iqm:.4f}")
    for seed, value in zip(SEEDS, final_scores):
        print(f"  seed {seed:2d}: {value:.4f}")

    # Machine-readable numerical outputs, one directory per environment.
    save_csv(
        game_dir / "validation_mean_ci90.csv",
        ["steps", "mean", "ci90_lower", "ci90_upper", "seed_std", "n_seeds"],
        zip(frames, means, lows, highs, stds, [n_seeds] * n_eval),
    )
    save_csv(
        game_dir / "final_seed_returns.csv",
        ["seed", "final_validation_mean_return"],
        zip(SEEDS, final_scores),
    )
    save_csv(
        game_dir / "final_summary.csv",
        ["environment", "algorithm", "n_seeds", "final_step", "mean", "ci90_lower", "ci90_upper", "std", "median", "iqm"],
        [[game, ALGORITHM_NAME, n_seeds, frames[-1], final_mean, final_low,
          final_high, final_std, final_median, final_iqm]],
    )

    # Figure 1: each environment's mean learning curve with NORMAL 90% CI.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames, means, label=f"Mean of {n_seeds} seeds", linewidth=2)
    ax.fill_between(frames, lows, highs, alpha=0.25, label="90% normal CI (no bootstrap)")
    ax.set(title=f"{ALGORITHM_NAME} — {game}",
           xlabel="Training environment steps", ylabel="Mean validation return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(game_dir / "validation_mean_ci90.png", dpi=180)

    # Figure 2: final per-seed means and the across-seed mean with normal CI.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(SEEDS, final_scores, label="Final validation mean per seed", zorder=3)
    ax.axhline(final_mean, label=f"Across-seed mean: {final_mean:.2f}", linestyle="--")
    ax.axhspan(final_low, final_high, alpha=0.17, label="90% normal CI of mean")
    ax.set(title=f"{ALGORITHM_NAME} — {game} — Final validation",
           xlabel="Seed", ylabel="Final validation mean return")
    ax.set_xticks(SEEDS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(game_dir / "final_scores.png", dpi=180)

    # Figure 3: performance profile for THIS environment only,
    # based on the empirical fraction of seeds reaching a score threshold.
    thresholds = np.linspace(float(final_scores.min()), float(final_scores.max()), 100)
    fraction_at_least = np.mean(final_scores[:, None] >= thresholds[None, :], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(thresholds, fraction_at_least, where="post")
    ax.set(title=f"{ALGORITHM_NAME} — {game} — Final performance profile",
           xlabel="Final validation return threshold",
           ylabel="Fraction of seeds at or above threshold", ylim=(-0.02, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(game_dir / "performance_profile.png", dpi=180)

    if not SHOW_PLOTS:
        plt.close("all")
    print(f"Saved results to {game_dir}")


def main():
    if not GAMES:
        raise ValueError("GAMES must contain at least one environment.")
    for game in GAMES:
        analyze_game(game)
    if SHOW_PLOTS:
        plt.show()
    print("\nDone. All plots/statistics are per environment, with no bootstrapping.")


if __name__ == "__main__":
    main()
