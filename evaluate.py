"""Plot MinAtar ablations using the reference screenshot's exact color palette.

Run from your project root:
    python evaluate_million_steps.py

Input for each seed:
    ./experiments/minatar/<setup>/<experiment>/<game>/seed_<N>/evaluation.npz
with arrays named "frames" and "mean_returns".

The main output is:
    ./minatar_evaluation/setup2/minatar_reference_style.png

The RGB colors below exactly match Seaborn's 8-color ``husl`` palette,
in the order shown in the supplied figure. No seaborn dependency required.
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


# Order matches the reference figure's legend, NOT filesystem order.
EXPERIMENTS = {
    # "DQN": Path("./experiments/minatar/setup1/dqn"),
    "Rainbow": Path("./experiments/minatar/setup1/rainbow"),
    "Rainbow 2": Path("./experiments/minatar/setup2/rainbow"),
    "Rainbow 3": Path("./experiments/minatar/setup3/rainbow"),
    "Rainbow 4": Path("./experiments/minatar/setup4/rainbow"),
    # "No Multi-step": Path("./experiments/minatar/setup1/no_nstep"),
    # "No Noisy": Path("./experiments/minatar/setup1/no-noisy"),
    # "No Dueling": Path("./experiments/minatar/setup1/no-duel"),
    # "No Double": Path("./experiments/minatar/setup1/no_double"),
    # "No Prioritized": Path("./experiments/minatar/setup1/no_per"),
    # "No Distributional": Path("./experiments/minatar/setup1/no_distrib"),
}

# EXACT Seaborn color_palette('husl', 8) colors (RGB rounded to 8-bit).
COLORS = {
    "DQN": "#F77189",
    "Rainbow": "#CE9032",
    "Rainbow 2": "#828A1D",
    "Rainbow 3": "#1D7D8A",
    "Rainbow 4": "#8A1D85",
    "No Multi-step": "#97A431",
    "No Noisy": "#32B166",
    "No Dueling": "#36ADA4",
    "No Double": "#39A7D0",
    "No Prioritized": "#A48CF4",
    "No Distributional": "#F561DD",
}

# Match the marker family in the reference figure.
MARKERS = {
    "DQN": "o",                # circle
    "Rainbow": "X",            # filled X
    "Rainbow 2": "X",            # filled X
    "Rainbow 3": "X",            # filled X
    "Rainbow 4": "X",            # filled X
    "No Multi-step": "s",     # square
    "No Noisy": "P",           # filled plus
    "No Dueling": "D",         # diamond
    "No Double": "d",          # thin diamond
    "No Prioritized": "^",     # upward triangle
    "No Distributional": "x",  # thin x
}

GAMES = [
    # "MinAtar_Asterix-v0",
    # "MinAtar_Breakout-v0",
    # "MinAtar_Freeway-v0",
    "MinAtar_Seaquest-v0",
#     "MinAtar_SpaceInvaders-v0",
]

SEEDS = list(range(10))
OUTPUT_DIR = Path("./minatar_evaluation/setup3")
CI_LEVEL = 0.90
SHOW_PLOTS = False

# Plot data at 1M, 2M, 3M, ... environment steps, regardless of EVAL_EVERY.
# If a milestone falls between saved evaluations, linearly interpolate returns.
PLOT_EVERY_STEPS = 1_000_000
SHOW_GAME_TITLES = False  # screenshot is cropped above the subplot titles
SAVE_DPI = 200

# These reproduce the approximate panel ranges visible in the screenshot.
# They are applied only when *all* observed means and CIs fit, to avoid
# inadvertently cutting off your own experiment results.
REFERENCE_Y_LIMITS = {
    "MinAtar_Asterix-v0": (-1.0, 54.0),
    "MinAtar_Breakout-v0": (0.0, 14.5),
    "MinAtar_Freeway-v0": (16.5, 62.5),
    "MinAtar_Seaquest-v0": (-1.0, 36.0),
    "MinAtar_SpaceInvaders-v0": (12.0, 107.0),
}
REFERENCE_Y_TICKS = {
    "MinAtar_Asterix-v0": np.arange(0, 51, 10),
    "MinAtar_Breakout-v0": np.arange(4, 15, 2),
    "MinAtar_Freeway-v0": np.arange(20, 61, 10),
    "MinAtar_Seaquest-v0": np.arange(5, 36, 5),
    "MinAtar_SpaceInvaders-v0": np.arange(20, 101, 20),
}

CI_Z = NormalDist().inv_cdf((1.0 + CI_LEVEL) / 2.0)


def safe_name(value: str) -> str:
    return (value.replace("/", "_").replace("\\", "_")
            .replace(" ", "_").replace(":", "_"))


def short_game_name(game: str) -> str:
    return game.replace("MinAtar_", "").replace("-v0", "")


def mean_normal_ci(scores: np.ndarray):
    """Pointwise, two-sided 90% normal-approximation CI across seeds."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n < 2:
        raise ValueError("At least two seeds are required for a confidence interval.")
    mean = np.mean(scores, axis=0)
    sd = np.std(scores, axis=0, ddof=1)
    margin = CI_Z * sd / np.sqrt(n)
    return mean, mean - margin, mean + margin


def load_one_curve(root: Path, game: str):
    """Align every seed on the same 1M, 2M, ... environment-step grid.

    Values between recorded evaluations are linearly interpolated, never
    extrapolated.  Only 1M milestones covered by *all* seeds are included.
    """
    seed_curves = []
    for seed in SEEDS:
        path = root / game / f"seed_{seed}" / "evaluation.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation file: {path.resolve()}")

        with np.load(path, allow_pickle=False) as data:
            frames = np.asarray(data["frames"], dtype=np.int64)
            returns = np.asarray(data["mean_returns"], dtype=np.float64)

        if (frames.ndim != 1 or returns.ndim != 1
                or not len(frames) or frames.shape != returns.shape):
            raise ValueError(f"Invalid evaluation arrays in {path}")
        if not np.isfinite(returns).all():
            raise ValueError(f"Non-finite evaluation return in {path}")
        if len(frames) > 1 and not np.all(np.diff(frames) > 0):
            raise ValueError(f"Evaluation frames must be strictly increasing: {path}")
        seed_curves.append((frames, returns))

    start_step = max(int(frames[0]) for frames, _ in seed_curves)
    end_step = min(int(frames[-1]) for frames, _ in seed_curves)
    # Always start at a positive, exact multiple of one million.
    first_milestone = max(1, (start_step + PLOT_EVERY_STEPS - 1) // PLOT_EVERY_STEPS)
    last_milestone = end_step // PLOT_EVERY_STEPS
    if first_milestone > last_milestone:
        raise ValueError(
            f"No shared 1M-step checkpoint for {root / game} "
            f"(all seeds overlap only between {start_step} and {end_step} steps)"
        )
    common_frames = np.arange(
        first_milestone, last_milestone + 1, dtype=np.int64
    ) * PLOT_EVERY_STEPS

    aligned = np.stack([
        np.interp(common_frames, frames, values)
        for frames, values in seed_curves
    ])
    print(f"[aligned] {root.name}/{game}: {len(common_frames)} one-million-step points")
    return common_frames, aligned


def available_experiments():
    """Skip entirely absent ablations but flag incomplete experiment results."""
    found = {}
    for label, root in EXPERIMENTS.items():
        first = root / GAMES[0] / f"seed_{SEEDS[0]}" / "evaluation.npz"
        if first.is_file():
            found[label] = root
        else:
            print(f"[skip] {label}: {first} does not exist")
    if not found:
        raise FileNotFoundError(
            "No evaluation data found. Check the EXPERIMENTS paths. "
            "Expected evaluation.npz for each experiment/game/seed."
        )
    return found


def load_all_curves(experiments):
    return {
        label: {game: load_one_curve(root, game) for game in GAMES}
        for label, root in experiments.items()
    }


def x_values(frames):
    """Plot x in millions of environment steps (1.0 means 1M)."""
    return np.asarray(frames, dtype=np.float64) / PLOT_EVERY_STEPS


def style_step_axis(ax):
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}M"))
    ax.set_xlabel("Training environment steps")


def line_and_band(ax, label, frames, scores, *, show_label=False):
    mean, low, high = mean_normal_ci(scores)
    x = x_values(frames)
    color = COLORS[label]
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(
        x, mean,
        color=color,
        linewidth=1.55,
        marker=MARKERS[label],
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=label if show_label else None,
        zorder=2,
    )
    return low.min(), high.max()


def set_reference_axis_style(ax, game, *, first=False):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=17, width=0.8, length=4, pad=0)
    ax.tick_params(axis="x", labelsize=10, labelrotation=45)
    style_step_axis(ax)
    ax.xaxis.label.set_fontsize(18)
    if first:
        ax.set_ylabel("Return", fontsize=26, labelpad=2)
    if SHOW_GAME_TITLES:
        ax.set_title(short_game_name(game).replace("SpaceInvaders", "Space Invaders"),
                     fontsize=18, pad=12)


def plot_reference_grid(curves):
    """One panel per selected game, with all ablations and 1M-step x-axis."""
    n_games = len(GAMES)
    fig_width = 18.57 if n_games == 5 else max(9.0, 3.7 * n_games)
    fig, axes = plt.subplots(1, n_games, figsize=(fig_width, 3.88), squeeze=False)
    axes = axes[0]
    if n_games == 5:
        fig.subplots_adjust(left=0.083, right=0.955, bottom=0.28, top=0.88,
                            wspace=0.20)
    else:
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.30, top=0.88,
                            wspace=0.27)
    labels = list(curves)
    legend_index = n_games // 2

    for j, game in enumerate(GAMES):
        ax = axes[j]
        lowest, highest = np.inf, -np.inf
        min_step, max_step = np.inf, -np.inf
        for label in labels:
            frames, scores = curves[label][game]
            low, high = line_and_band(
                ax, label, frames, scores, show_label=(j == legend_index)
            )
            lowest = min(lowest, low)
            highest = max(highest, high)
            min_step = min(min_step, frames[0])
            max_step = max(max_step, frames[-1])

        set_reference_axis_style(ax, game, first=(j == 0))
        ax.set_xlim(min_step / PLOT_EVERY_STEPS - 0.25,
                    max_step / PLOT_EVERY_STEPS + 0.25)
        ylim = REFERENCE_Y_LIMITS[game]
        if ylim[0] <= lowest and highest <= ylim[1]:
            ax.set_ylim(*ylim)
            ax.set_yticks(REFERENCE_Y_TICKS[game])
        else:
            ax.margins(y=0.06)
            print(f"[scale] {short_game_name(game)}: using automatic y-limits "
                  f"because data/CI extend beyond {ylim}.")

    # Keep the reference-style legend in the middle panel, even for one game.
    axes[legend_index].legend(
        loc="lower right", fontsize=12.0,
        frameon=True, fancybox=False, framealpha=0.93,
        edgecolor="#BBBBBB", facecolor="white",
        labelspacing=0.40, handlelength=2.1,
        borderpad=0.35, borderaxespad=0.38,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "minatar_reference_style.png"
    fig.savefig(destination, dpi=SAVE_DPI, facecolor="white")
    print(f"Saved {destination}")
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_per_environment(curves):
    """One plot for each game, using precisely the same color/marker mapping."""
    out_dir = OUTPUT_DIR / "learning_curves_by_environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    for game in GAMES:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label in curves:
            frames, scores = curves[label][game]
            line_and_band(ax, label, frames, scores, show_label=True)
        ax.set_title(f"{short_game_name(game)} (90% CI)")
        style_step_axis(ax)
        ax.set_ylabel("Return")
        ax.grid(alpha=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name(game)}_mean_ci90.png", dpi=SAVE_DPI)
        if not SHOW_PLOTS:
            plt.close(fig)


def plot_per_configuration(curves):
    """Original one-image-per-configuration output, with fixed scheme."""
    out_dir = OUTPUT_DIR / "learning_curves_by_configuration"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in curves:
        fig, axes = plt.subplots(1, len(GAMES), figsize=(4.1 * len(GAMES), 4.2),
                                 squeeze=False)
        for j, game in enumerate(GAMES):
            ax = axes[0, j]
            frames, scores = curves[label][game]
            line_and_band(ax, label, frames, scores)
            ax.set_title(short_game_name(game))
            style_step_axis(ax)
            ax.tick_params(axis="x", labelrotation=45)
            ax.grid(alpha=0.15)
            if j == 0:
                ax.set_ylabel("Return")
        fig.suptitle(f"{label}: mean evaluation return with 90% CI")
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name(label)}_all_games.png", dpi=SAVE_DPI)
        if not SHOW_PLOTS:
            plt.close(fig)


def main():
    experiments = available_experiments()
    print("Loading evaluation results...")
    curves = load_all_curves(experiments)
    plot_reference_grid(curves)
    plot_per_environment(curves)
    plot_per_configuration(curves)
    if SHOW_PLOTS:
        plt.show()
    print(f"Done. Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
