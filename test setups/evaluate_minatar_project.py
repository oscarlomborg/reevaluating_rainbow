
"""
evaluate_minatar_project.py

Evaluation pipeline for the MinAtar stage of the Rainbow project.

Main ideas
----------
A) Descriptive learning curves:
   - raw return, separately for each environment
   - mean across seeds
   - pointwise 90% NORMAL confidence interval
   - no bootstrap for these curves

B) Final per-environment results:
   - raw returns
   - mean, 90% CI, median, IQM across seeds

C) Cross-environment RLiable-style analysis:
   - raw probability of improvement (scale-safe)
   - aggregate IQM/mean/median and performance profiles on a
     clearly labelled Rainbow-standardized score:
         z = (score - Rainbow_full_mean_game) / Rainbow_full_sd_game
   - this is NOT human normalization

D) Seed-count calibration for RQ1:
   - subsample n = 5, 10, 15, 20, 25, 30 runs
   - for each subsample, bootstrap Rainbow - ablation
   - effect detected when the 95% bootstrap CI excludes zero
   - report detection probability and CI width
   - do this both:
       1) per environment on raw returns
       2) across environments on Rainbow-standardized IQM

Expected files
--------------
experiments/<configuration>/<game>/seed_<seed>/evaluation.npz

Each evaluation.npz must contain:
    frames        : shape (num_eval_points,)
    mean_returns  : shape (num_eval_points,)

Run:
    python evaluate_minatar_project.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

# IMPORTANT: Edit these paths/names to match your experiment folders.
# "Rainbow" must be the full reference configuration.
EXPERIMENTS = {
    "DQN": Path("./experiments/minatar_dqn"),
    "Rainbow": Path("./experiments/minatar_rainbow"),
    "No PER": Path("./experiments/minatar_no-per"),
    "No n-step": Path("./experiments/minatar_no-nstep"),
    "No dueling": Path("./experiments/minatar_no-duel"),
    # "No distributional": Path("./experiments/minatar_no-distrib"),
    "No noisy": Path("./experiments/minatar_no-noisy"),
    # "No double": Path("./experiments/minatar_no-double"),
}

REFERENCE_CONFIG = "Rainbow"

GAMES = [
    "MinAtar_Asterix-v1",
    "MinAtar_Breakout-v1",
    "MinAtar_Freeway-v1",
    # "MinAtar_Seaquest-v1",
    "MinAtar_SpaceInvaders-v1",
]

# Full MinAtar calibration target from the project plan.
SEEDS = list(range(10))

# Candidate run counts for the retrospective seed-count study.
SEED_COUNTS = [3, 5, 10]

OUTPUT_DIR = Path("./minatar_evaluation")

# Revisiting-Rainbow-style descriptive curve CI.
CURVE_CI_LEVEL = 0.90

# Project-plan detection rule uses bootstrap CI for the difference.
DIFFERENCE_CI_LEVEL = 0.95

# Number of random smaller experiments drawn from the full seed pool.
# 300 is convenient while developing. For final analysis, 1000 is reasonable.
N_SUBSAMPLES = 300

# Bootstrap repetitions inside each sampled smaller experiment.
# Increase to 2000 for final figures if runtime is acceptable.
SUBSAMPLE_BOOTSTRAP_REPS = 1000

# RLiable aggregate/bootstrap settings.
RLIABLE_PROFILE_REPS = 2000
RLIABLE_AGGREGATE_REPS = 50_000

# Reproducibility of the evaluation itself.
ANALYSIS_SEED = 12345

# If False, figures are only saved and no interactive window is opened.
SHOW_PLOTS = False


# ============================================================
# Small numerical helpers
# ============================================================

CURVE_Z = NormalDist().inv_cdf((1.0 + CURVE_CI_LEVEL) / 2.0)


def safe_name(text: str) -> str:
    """Make a string safe for filenames."""
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def save_csv(path: Path, header, rows):
    """Save rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def exact_iqm_1d(values: np.ndarray) -> float:
    """Exact 25%-trimmed mean, including fractional boundary weights."""
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(x)

    if n == 0:
        raise ValueError("Cannot compute IQM of an empty array.")

    start = 0.25 * n
    stop = 0.75 * n
    edges = np.arange(n + 1, dtype=np.float64)

    weights = np.maximum(
        0.0,
        np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start),
    )

    return float(np.dot(x, weights) / (stop - start))


def exact_iqm_rows(values: np.ndarray) -> np.ndarray:
    """
    IQM for every row of a 2-D matrix.

    values shape:
        (num_bootstrap_reps, num_scores)
    """
    x = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    n = x.shape[1]

    start = 0.25 * n
    stop = 0.75 * n
    edges = np.arange(n + 1, dtype=np.float64)

    weights = np.maximum(
        0.0,
        np.minimum(edges[1:], stop) - np.maximum(edges[:-1], start),
    )

    return (x @ weights) / (stop - start)


def percentile_ci(samples: np.ndarray, level: float):
    """Two-sided percentile confidence interval."""
    alpha = 1.0 - level
    return np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])


def mean_normal_ci(scores: np.ndarray):
    """
    Mean +/- normal-approximation CI across axis 0.

    This is deliberately NOT bootstrapped.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]

    if n < 2:
        raise ValueError("At least two seeds are required for a confidence interval.")

    mean = np.mean(scores, axis=0)
    sd = np.std(scores, axis=0, ddof=1)
    margin = CURVE_Z * sd / np.sqrt(n)

    return mean, mean - margin, mean + margin, sd


# ============================================================
# Loading
# ============================================================

def load_evaluation_file(root: Path, game: str, seed: int):
    """Load one seed's evaluation curve."""
    path = root / game / f"seed_{seed}" / "evaluation.npz"

    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation file: {path}")

    with np.load(path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"], dtype=np.int64)
        returns = np.asarray(data["mean_returns"], dtype=np.float64)

    if frames.ndim != 1 or returns.ndim != 1:
        raise ValueError(f"Expected 1-D frames and mean_returns in {path}")

    if len(frames) == 0 or frames.shape != returns.shape:
        raise ValueError(f"Inconsistent evaluation arrays in {path}")

    if not np.isfinite(returns).all():
        raise ValueError(f"Non-finite evaluation return in {path}")

    return frames, returns


def load_one_curve(root: Path, game: str):
    """Load all seeds for one configuration/game."""
    reference_frames = None
    per_seed = []

    for seed in SEEDS:
        frames, returns = load_evaluation_file(root, game, seed)

        if reference_frames is None:
            reference_frames = frames
        elif not np.array_equal(reference_frames, frames):
            raise ValueError(
                f"Evaluation steps differ within {game}: seed {seed} does not "
                "match the first seed."
            )

        per_seed.append(returns)

    return reference_frames, np.stack(per_seed, axis=0)


def load_all_data():
    """
    Returns
    -------
    curves[label][game] = (frames, scores)
        scores shape = (num_seeds, num_eval_points)

    final_scores[label]
        shape = (num_seeds, num_games)
    """
    if REFERENCE_CONFIG not in EXPERIMENTS:
        raise ValueError(
            f"REFERENCE_CONFIG={REFERENCE_CONFIG!r} must be present in EXPERIMENTS."
        )

    curves = {}
    final_scores = {}

    for label, root in EXPERIMENTS.items():
        curves[label] = {}
        game_final_scores = []

        for game in GAMES:
            frames, scores = load_one_curve(root, game)
            curves[label][game] = (frames, scores)

            # Final evaluation point only; never take max-over-training.
            game_final_scores.append(scores[:, -1])

        # (games, seeds) -> (seeds, games)
        final_scores[label] = np.stack(game_final_scores, axis=0).T

    return curves, final_scores


# ============================================================
# 1. Descriptive learning curves: raw score + 90% normal CI
# ============================================================

def plot_per_environment(curves):
    """One figure per environment comparing all configurations."""
    out_dir = OUTPUT_DIR / "learning_curves_by_environment"
    out_dir.mkdir(parents=True, exist_ok=True)

    for game in GAMES:
        fig, ax = plt.subplots(figsize=(9, 5))

        for label in EXPERIMENTS:
            frames, scores = curves[label][game]
            mean, low, high, _ = mean_normal_ci(scores)

            line = ax.plot(
                frames,
                mean,
                linewidth=2,
                label=label,
            )[0]

            ax.fill_between(
                frames,
                low,
                high,
                color=line.get_color(),
                alpha=0.20,
            )

        title = game.replace("MinAtar_", "").replace("-v1", "")
        ax.set_title(title)
        ax.set_xlabel("Training environment steps")
        ax.set_ylabel("Mean evaluation return")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            out_dir / f"{safe_name(game)}_mean_ci90.png",
            dpi=200,
        )

        if not SHOW_PLOTS:
            plt.close(fig)


def plot_per_configuration(curves):
    """One figure per configuration, with one subplot for each environment."""
    out_dir = OUTPUT_DIR / "learning_curves_by_configuration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in EXPERIMENTS:
        fig, axes = plt.subplots(
            1,
            len(GAMES),
            figsize=(4.1 * len(GAMES), 4.2),
            squeeze=False,
        )

        for j, game in enumerate(GAMES):
            ax = axes[0, j]
            frames, scores = curves[label][game]
            mean, low, high, _ = mean_normal_ci(scores)

            line = ax.plot(frames, mean, linewidth=2)[0]
            ax.fill_between(
                frames,
                low,
                high,
                color=line.get_color(),
                alpha=0.20,
            )

            title = game.replace("MinAtar_", "").replace("-v1", "")
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ax.grid(alpha=0.3)

            if j == 0:
                ax.set_ylabel("Mean evaluation return")

        fig.suptitle(
            f"{label}: mean evaluation return with "
            f"{int(CURVE_CI_LEVEL * 100)}% normal CI"
        )
        fig.tight_layout()

        fig.savefig(
            out_dir / f"{safe_name(label)}_all_games.png",
            dpi=200,
        )

        if not SHOW_PLOTS:
            plt.close(fig)


# ============================================================
# 2. Final raw per-game summaries
# ============================================================

def save_final_raw_summaries(final_scores):
    """
    Save raw final results separately per environment.

    Raw MinAtar returns are NOT pooled across environments here.
    """
    summary_rows = []
    seed_rows = []

    for label, scores in final_scores.items():
        for game_index, game in enumerate(GAMES):
            values = scores[:, game_index]
            mean, low, high, sd = mean_normal_ci(values)

            summary_rows.append([
                label,
                game,
                len(values),
                float(mean),
                float(low),
                float(high),
                float(sd),
                float(np.median(values)),
                exact_iqm_1d(values),
            ])

            for seed, value in zip(SEEDS, values):
                seed_rows.append([
                    label,
                    game,
                    seed,
                    float(value),
                ])

    save_csv(
        OUTPUT_DIR / "tables" / "final_raw_summary.csv",
        [
            "configuration",
            "environment",
            "n_seeds",
            "mean",
            "ci90_lower",
            "ci90_upper",
            "std",
            "median",
            "iqm_across_seeds",
        ],
        summary_rows,
    )

    save_csv(
        OUTPUT_DIR / "tables" / "final_raw_scores.csv",
        ["configuration", "environment", "seed", "final_mean_return"],
        seed_rows,
    )


# ============================================================
# 3. Rainbow-standardized scores for cross-game aggregation
# ============================================================

def rainbow_standardization(final_scores):
    """
    Standardize each environment using FULL Rainbow data:

        z = (score - Rainbow_mean_game) / Rainbow_sd_game

    Interpretation:
        z =  0   -> Rainbow full-seed mean for that game
        z = -1   -> one Rainbow SD below that mean
        z = +1   -> one Rainbow SD above that mean

    This is an effect-size scale for THIS experiment.
    It is NOT human normalization and should be labelled accordingly.
    """
    rainbow = final_scores[REFERENCE_CONFIG]

    centers = np.mean(rainbow, axis=0)
    scales = np.std(rainbow, axis=0, ddof=1)

    if np.any(~np.isfinite(scales)) or np.any(scales <= 1e-12):
        bad = [
            GAMES[i]
            for i, scale in enumerate(scales)
            if (not np.isfinite(scale)) or scale <= 1e-12
        ]
        raise ValueError(
            "Cannot standardize because Rainbow has zero/nearly-zero "
            f"seed variation for: {bad}"
        )

    standardized = {
        label: (scores - centers[None, :]) / scales[None, :]
        for label, scores in final_scores.items()
    }

    rows = [
        [game, float(centers[i]), float(scales[i])]
        for i, game in enumerate(GAMES)
    ]

    save_csv(
        OUTPUT_DIR / "tables" / "rainbow_standardization.csv",
        ["environment", "rainbow_full_mean", "rainbow_full_sample_sd"],
        rows,
    )

    return standardized, centers, scales


# ============================================================
# 4. RLiable full-data analysis
# ============================================================

def run_rliable_full_analysis(final_scores, standardized_scores):
    """
    RLiable outputs:
      - probability of improvement from RAW returns
      - IQM/mean/median from Rainbow-standardized returns
      - performance profiles on Rainbow-standardized returns
    """
    try:
        from rliable import library as rly
        from rliable import metrics
        from rliable import plot_utils
    except ImportError as exc:
        print(
            "\n[RLiable skipped] Install it with:\n"
            "    pip install -U rliable\n"
            f"Reason: {exc}\n"
        )
        return

    out_dir = OUTPUT_DIR / "rliable"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = {}

    for label, scores in final_scores.items():
        if label == REFERENCE_CONFIG:
            continue

        comparisons[f"{REFERENCE_CONFIG} > {label}"] = (
            final_scores[REFERENCE_CONFIG],
            scores,
        )

    if comparisons:
        probabilities, probability_cis = rly.get_interval_estimates(
            comparisons,
            metrics.probability_of_improvement,
            reps=RLIABLE_PROFILE_REPS,
        )

        rows = []
        for name in comparisons:
            estimate = float(np.asarray(probabilities[name]).squeeze())
            ci = np.asarray(probability_cis[name]).reshape(-1)

            rows.append([
                name,
                estimate,
                float(ci[0]),
                float(ci[1]),
            ])

        save_csv(
            out_dir / "probability_of_improvement.csv",
            ["comparison", "probability", "ci95_lower", "ci95_upper"],
            rows,
        )

        plot_utils.plot_probability_of_improvement(
            probabilities,
            probability_cis,
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(
            out_dir / "probability_of_improvement.png",
            dpi=200,
            bbox_inches="tight",
        )

        if not SHOW_PLOTS:
            plt.close(fig)

    def aggregate_metrics(x):
        return np.array([
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
        ])

    estimates, cis = rly.get_interval_estimates(
        standardized_scores,
        aggregate_metrics,
        reps=RLIABLE_AGGREGATE_REPS,
    )

    plot_utils.plot_interval_estimates(
        estimates,
        cis,
        metric_names=["IQM", "Mean", "Median"],
        algorithms=list(EXPERIMENTS.keys()),
        xlabel="Rainbow-standardized final return (SD units)",
    )

    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(
        out_dir / "standardized_aggregate_metrics.png",
        dpi=200,
        bbox_inches="tight",
    )

    if not SHOW_PLOTS:
        plt.close(fig)

    all_values = np.concatenate(
        [scores.reshape(-1) for scores in standardized_scores.values()]
    )

    thresholds = np.linspace(
        float(np.min(all_values)),
        float(np.max(all_values)),
        151,
    )

    profiles, profile_cis = rly.create_performance_profile(
        standardized_scores,
        thresholds,
        reps=RLIABLE_PROFILE_REPS,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_utils.plot_performance_profiles(
        profiles,
        thresholds,
        performance_profile_cis=profile_cis,
        xlabel="Rainbow-standardized final return (SD units)",
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "standardized_performance_profile.png",
        dpi=200,
        bbox_inches="tight",
    )

    if not SHOW_PLOTS:
        plt.close(fig)


# ============================================================
# 5. Bootstrap difference helpers
# ============================================================

def bootstrap_mean_difference(
    rainbow_values: np.ndarray,
    ablation_values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
):
    """
    Bootstrap CI for:
        mean(Rainbow) - mean(ablation)

    Used for one environment, so raw returns are directly comparable.
    """
    rainbow_values = np.asarray(rainbow_values, dtype=np.float64)
    ablation_values = np.asarray(ablation_values, dtype=np.float64)

    nr = len(rainbow_values)
    na = len(ablation_values)

    r_idx = rng.integers(0, nr, size=(reps, nr))
    a_idx = rng.integers(0, na, size=(reps, na))

    r_boot = rainbow_values[r_idx].mean(axis=1)
    a_boot = ablation_values[a_idx].mean(axis=1)

    boot_delta = r_boot - a_boot
    estimate = float(np.mean(rainbow_values) - np.mean(ablation_values))
    low, high = percentile_ci(boot_delta, DIFFERENCE_CI_LEVEL)

    return estimate, float(low), float(high)


def stratified_bootstrap_iqm_difference(
    rainbow_scores: np.ndarray,
    ablation_scores: np.ndarray,
    rng: np.random.Generator,
    reps: int,
):
    """
    Stratified bootstrap across environments for:

        IQM(Rainbow) - IQM(ablation)

    Inputs are already on the Rainbow-standardized common scale.

    Each environment's runs are independently resampled with replacement.
    """
    rainbow_scores = np.asarray(rainbow_scores, dtype=np.float64)
    ablation_scores = np.asarray(ablation_scores, dtype=np.float64)

    nr, num_games = rainbow_scores.shape
    na, num_games_a = ablation_scores.shape

    if num_games != num_games_a:
        raise ValueError("Rainbow and ablation must contain the same games.")

    r_boot = np.empty((reps, nr, num_games), dtype=np.float64)
    a_boot = np.empty((reps, na, num_games), dtype=np.float64)

    for game_index in range(num_games):
        r_idx = rng.integers(0, nr, size=(reps, nr))
        a_idx = rng.integers(0, na, size=(reps, na))

        r_boot[:, :, game_index] = rainbow_scores[:, game_index][r_idx]
        a_boot[:, :, game_index] = ablation_scores[:, game_index][a_idx]

    r_iqm = exact_iqm_rows(r_boot.reshape(reps, -1))
    a_iqm = exact_iqm_rows(a_boot.reshape(reps, -1))

    boot_delta = r_iqm - a_iqm

    estimate = (
        exact_iqm_1d(rainbow_scores.reshape(-1))
        - exact_iqm_1d(ablation_scores.reshape(-1))
    )

    low, high = percentile_ci(boot_delta, DIFFERENCE_CI_LEVEL)

    return float(estimate), float(low), float(high)


# ============================================================
# 6. Seed-count calibration
# ============================================================

def seed_calibration(final_scores, standardized_scores):
    """
    Retrospective seed-count study.

    For smaller n:
      repeat many random subsets of n runs.

    For n == all available runs:
      use the full data once.

    Detection rule:
      95% bootstrap CI for Rainbow - ablation excludes zero.
    """
    rng = np.random.default_rng(ANALYSIS_SEED)

    ref_raw = final_scores[REFERENCE_CONFIG]
    ref_std = standardized_scores[REFERENCE_CONFIG]

    aggregate_rows = []
    per_game_rows = []

    ablations = [
        label
        for label in EXPERIMENTS
        if label != REFERENCE_CONFIG
    ]

    for ablation in ablations:
        abl_raw = final_scores[ablation]
        abl_std = standardized_scores[ablation]

        for n in SEED_COUNTS:
            if n > len(SEEDS):
                continue

            repetitions = 1 if n == len(SEEDS) else N_SUBSAMPLES

            aggregate_detected = []
            aggregate_rainbow_better = []
            aggregate_delta = []
            aggregate_width = []

            game_detected = {game: [] for game in GAMES}
            game_rainbow_better = {game: [] for game in GAMES}
            game_delta = {game: [] for game in GAMES}
            game_width = {game: [] for game in GAMES}

            for _ in range(repetitions):
                if n == len(SEEDS):
                    ref_raw_sub = ref_raw
                    abl_raw_sub = abl_raw
                    ref_std_sub = ref_std
                    abl_std_sub = abl_std
                else:
                    ref_indices = rng.choice(
                        ref_raw.shape[0], size=n, replace=False
                    )
                    abl_indices = rng.choice(
                        abl_raw.shape[0], size=n, replace=False
                    )

                    ref_raw_sub = ref_raw[ref_indices]
                    abl_raw_sub = abl_raw[abl_indices]
                    ref_std_sub = ref_std[ref_indices]
                    abl_std_sub = abl_std[abl_indices]

                delta, low, high = stratified_bootstrap_iqm_difference(
                    ref_std_sub,
                    abl_std_sub,
                    rng,
                    SUBSAMPLE_BOOTSTRAP_REPS,
                )

                aggregate_delta.append(delta)
                aggregate_width.append(high - low)
                aggregate_detected.append((low > 0.0) or (high < 0.0))
                aggregate_rainbow_better.append(low > 0.0)

                for game_index, game in enumerate(GAMES):
                    delta_g, low_g, high_g = bootstrap_mean_difference(
                        ref_raw_sub[:, game_index],
                        abl_raw_sub[:, game_index],
                        rng,
                        SUBSAMPLE_BOOTSTRAP_REPS,
                    )

                    game_delta[game].append(delta_g)
                    game_width[game].append(high_g - low_g)
                    game_detected[game].append(
                        (low_g > 0.0) or (high_g < 0.0)
                    )
                    game_rainbow_better[game].append(low_g > 0.0)

            aggregate_rows.append([
                ablation,
                n,
                repetitions,
                float(np.mean(aggregate_delta)),
                float(np.median(aggregate_delta)),
                float(np.mean(aggregate_width)),
                float(np.median(aggregate_width)),
                float(np.mean(aggregate_detected)),
                float(np.mean(aggregate_rainbow_better)),
            ])

            for game in GAMES:
                per_game_rows.append([
                    ablation,
                    game,
                    n,
                    repetitions,
                    float(np.mean(game_delta[game])),
                    float(np.median(game_delta[game])),
                    float(np.mean(game_width[game])),
                    float(np.median(game_width[game])),
                    float(np.mean(game_detected[game])),
                    float(np.mean(game_rainbow_better[game])),
                ])

            print(
                f"[calibration] {ablation:20s} n={n:2d} "
                f"detect={np.mean(aggregate_detected):.3f} "
                f"Rainbow-better={np.mean(aggregate_rainbow_better):.3f}"
            )

    calibration_dir = OUTPUT_DIR / "seed_calibration"

    save_csv(
        calibration_dir / "aggregate_iqm_detection.csv",
        [
            "ablation",
            "n_seeds",
            "num_subsamples",
            "mean_delta_iqm",
            "median_delta_iqm",
            "mean_ci_width",
            "median_ci_width",
            "detection_probability",
            "rainbow_better_probability",
        ],
        aggregate_rows,
    )

    save_csv(
        calibration_dir / "per_game_mean_detection.csv",
        [
            "ablation",
            "environment",
            "n_seeds",
            "num_subsamples",
            "mean_delta_return",
            "median_delta_return",
            "mean_ci_width",
            "median_ci_width",
            "detection_probability",
            "rainbow_better_probability",
        ],
        per_game_rows,
    )

    plot_seed_calibration(aggregate_rows, per_game_rows)


# ============================================================
# 7. Seed-calibration plots
# ============================================================

def plot_seed_calibration(aggregate_rows, per_game_rows):
    """Create the main RQ1 figures."""
    out_dir = OUTPUT_DIR / "seed_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        label
        for label in EXPERIMENTS
        if label != REFERENCE_CONFIG
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for ablation in ablations:
        rows = [row for row in aggregate_rows if row[0] == ablation]
        rows.sort(key=lambda row: row[1])

        x = [row[1] for row in rows]
        y = [row[7] for row in rows]

        ax.plot(x, y, marker="o", linewidth=2, label=ablation)

    ax.axhline(
        0.80,
        linestyle="--",
        linewidth=1,
        label="80% detection",
    )
    ax.set_xlabel("Number of seeds per configuration/game")
    ax.set_ylabel("Probability bootstrap CI excludes zero")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        "Seed calibration: detecting Rainbow component effects\n"
        "(aggregate Rainbow-standardized IQM difference)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "aggregate_detection_probability.png",
        dpi=200,
    )

    if not SHOW_PLOTS:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for ablation in ablations:
        rows = [row for row in aggregate_rows if row[0] == ablation]
        rows.sort(key=lambda row: row[1])

        x = [row[1] for row in rows]
        y = [row[6] for row in rows]

        ax.plot(x, y, marker="o", linewidth=2, label=ablation)

    ax.set_xlabel("Number of seeds per configuration/game")
    ax.set_ylabel("Median 95% bootstrap CI width")
    ax.set_title(
        "Uncertainty versus number of seeds\n"
        "(Rainbow-standardized IQM difference)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "aggregate_ci_width.png",
        dpi=200,
    )

    if not SHOW_PLOTS:
        plt.close(fig)

    for ablation in ablations:
        fig, ax = plt.subplots(figsize=(9, 5))

        for game in GAMES:
            rows = [
                row
                for row in per_game_rows
                if row[0] == ablation and row[1] == game
            ]
            rows.sort(key=lambda row: row[2])

            x = [row[2] for row in rows]
            y = [row[8] for row in rows]

            short_game = game.replace("MinAtar_", "").replace("-v1", "")
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=short_game,
            )

        ax.axhline(
            0.80,
            linestyle="--",
            linewidth=1,
            label="80% detection",
        )
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Probability bootstrap CI excludes zero")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{ablation}: per-environment detection probability")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            out_dir / f"{safe_name(ablation)}_per_game_detection.png",
            dpi=200,
        )

        if not SHOW_PLOTS:
            plt.close(fig)


# ============================================================
# 8. Simple full-seed effect table
# ============================================================

def save_full_seed_effect_table(final_scores, standardized_scores):
    """
    Compact table for the report:
      - aggregate standardized IQM difference
      - per-game raw mean differences
    """
    rows = []

    ref_raw = final_scores[REFERENCE_CONFIG]
    ref_std = standardized_scores[REFERENCE_CONFIG]

    for label in EXPERIMENTS:
        if label == REFERENCE_CONFIG:
            continue

        abl_raw = final_scores[label]
        abl_std = standardized_scores[label]

        aggregate_delta = (
            exact_iqm_1d(ref_std.reshape(-1))
            - exact_iqm_1d(abl_std.reshape(-1))
        )

        raw_differences = (
            np.mean(ref_raw, axis=0)
            - np.mean(abl_raw, axis=0)
        )

        rows.append(
            [label, aggregate_delta]
            + [float(x) for x in raw_differences]
        )

    save_csv(
        OUTPUT_DIR / "tables" / "full_seed_effects.csv",
        [
            "ablation",
            "rainbow_minus_ablation_standardized_iqm",
            *[
                f"raw_mean_difference_{safe_name(game)}"
                for game in GAMES
            ],
        ],
        rows,
    )


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation results...")
    curves, final_scores = load_all_data()

    print("Creating raw learning-curve plots...")
    plot_per_environment(curves)
    plot_per_configuration(curves)

    print("Saving raw final-score summaries...")
    save_final_raw_summaries(final_scores)

    print("Creating Rainbow-standardized effect scores...")
    standardized_scores, _, _ = rainbow_standardization(final_scores)

    print("Saving full-seed effect table...")
    save_full_seed_effect_table(
        final_scores,
        standardized_scores,
    )

    print("Running full-data RLiable analysis...")
    run_rliable_full_analysis(
        final_scores,
        standardized_scores,
    )

    if len(EXPERIMENTS) > 1:
        print("Running seed-count calibration...")
        seed_calibration(
            final_scores,
            standardized_scores,
        )
    else:
        print(
            "\nSeed calibration skipped: add at least one ablation "
            "to EXPERIMENTS."
        )

    if SHOW_PLOTS:
        plt.show()

    print(
        f"\nDone. Results saved to:\n{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
