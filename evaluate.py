"""Plot MinAtar learning curves, violin plots, Rainbow effects, and seed-count sensitivity.

Required CSV columns:
    env_type,setup,experiment,environment,seed,frame,mean_return

Examples:
    python evaluate_rainbow_subsampling.py --csv experiments/minatar/setup1/mean_returns.csv
    python evaluate_rainbow_subsampling.py --csv combined_mean_returns.csv --output minatar_evaluation
    python evaluate_rainbow_subsampling.py --csv combined_mean_returns.csv --setup setup1

Outputs include per-environment learning curves with pointwise 90% percentile
bootstrap confidence bands, violin plots of final return across seeds, a combined
learning-curve grid, and a final-performance CSV with 90% bootstrap CIs for means.
For each setup, also writes a Rainbow-minus-ablation heatmap showing effect sizes and 90% bootstrap
CIs inside each cell, plus effect-size CSVs. DQN is shown separately
as an overall comparison rather than as an ablated Rainbow component.
Subsampling plots and CSVs compare 3, 5, 10, 15, and 20 training seeds,
including both detailed per-subset bootstrap intervals and a compact forest-style
summary that shows the pooled bootstrap mean and 90% CI for each run count.
Each final violin compares seeds at the last evaluation frame SHARED by
all selected experiment/seed combinations within that environment.

Bootstrap resampling samples complete seed-level trajectories; the same sampled
seeds are used at every plotted checkpoint within a bootstrap replicate.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")  # Safe for scripts on headless servers and HPC clusters.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
import numpy as np

REQUIRED_COLUMNS = {
    "env_type", "setup", "experiment", "environment", "seed", "frame", "mean_return"
}

# Matches the palette used by your previous evaluate.py.
COLORS = {
    "dqn": "#F77189", "rainbow": "#CE9032",
    "no_nstep": "#97A431", "no_noisy": "#32B166",
    "no_dueling": "#36ADA4", "no_double": "#39A7D0",
    "no_per": "#A48CF4", "no_distrib": "#F561DD",
}
LABELS = {
    "dqn": "DQN", "rainbow": "Rainbow",
    "no_nstep": "No Multi-step", "no_noisy": "No Noisy",
    "no_dueling": "No Dueling", "no_double": "No Double",
    "no_per": "No Prioritized", "no_distrib": "No Distributional",
}
MARKERS = ["o", "X", "s", "P", "D", "d", "^", "x", "v", "h"]
CI_LEVEL = 0.90
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42

# Explicit order makes the plots comparable between architecture setups.
EFFECT_COMPARISONS = [
    ("no_double", "No Double", "ablation"),
    ("no_per", "No PER", "ablation"),
    ("no_dueling", "No Dueling", "ablation"),
    ("no_nstep", "No Multi-step", "ablation"),
    ("no_distrib", "No Distributional", "ablation"),
    ("no_noisy", "No Noisy", "ablation"),
    ("dqn", "DQN (overall)", "overall"),
]
GAME_ORDER = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]


def keyed_rng(base_seed, key):
    """Reproducible independent RNG for each model/plot regardless of plot order."""
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    derived_seed = (int.from_bytes(digest, "little") + base_seed) % (2**64)
    return np.random.default_rng(derived_seed)


def bootstrap_mean_ci(scores, resamples=DEFAULT_BOOTSTRAP_RESAMPLES, rng=None):
    """Pointwise 90% percentile bootstrap CI for seed-level means.

    `scores` has shape (n_seeds,) or (n_seeds, n_frames). One set of
    seed-resampling weights is applied to the entire time series within
    each replicate, preserving the dependence across evaluation frames.
    """
    arr = np.asarray(scores, dtype=float)
    is_vector = arr.ndim == 1
    if is_vector:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Bootstrap CIs require at least two independent seed-level runs")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    if rng is None:
        rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)

    n = arr.shape[0]
    # Multinomial weights are exactly equivalent to resampling n seeds
    # with replacement and then averaging. They avoid huge index arrays.
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples)
    boot_means = (weights @ arr) / n
    alpha = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha], axis=0)
    if is_vector:
        return float(low[0]), float(high[0])
    return low, high



def bootstrap_difference_ci(x, y, resamples=DEFAULT_BOOTSTRAP_RESAMPLES, rng=None):
    """90% percentile bootstrap CI for mean(x) - mean(y).

    Independently resamples training runs within each configuration.
    In particular, identical numerical seed labels are NOT assumed paired.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("Bootstrap differences require at least 2 seeds per group")
    if rng is None:
        rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    wx = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=resamples)
    wy = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=resamples)
    boot_diff = (wx @ x) / len(x) - (wy @ y) / len(y)
    alpha = (1 - CI_LEVEL) / 2
    low, high = np.quantile(boot_diff, [alpha, 1 - alpha])
    return float(low), float(high)


def save_csv_rows(path, rows, columns):
    """Write a regular long-format CSV, including its header if rows are empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def draw_effect_heatmap(kind, setup, games, values, ci_lowers, ci_uppers,
                        output_path, dpi):
    """Rainbow-minus-comparison heatmap with 90% bootstrap CIs in each cell.

    The fill color represents the difference in mean final return. Each
    populated cell displays its effect estimate and the CI calculated from
    independent seed-level resampling of the *difference* (not the two means
    independently displayed as separate intervals).
    """
    if not np.any(np.isfinite(values)):
        print(f"[skip heatmap] {(kind, setup)}: no complete comparisons")
        return

    fig, ax = plt.subplots(figsize=(max(12.5, len(games) * 2.7), 7.2))
    vmax = float(np.nanmax(np.abs(values)))
    if vmax == 0:
        vmax = 1.0
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    heatmap = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                        norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(games)))
    ax.set_xticklabels([pretty_game(g) for g in games], rotation=12, ha="right")
    ax.set_yticks(np.arange(len(EFFECT_COMPARISONS)))
    ax.set_yticklabels([label for _, label, _ in EFFECT_COMPARISONS])
    ax.set_xlabel("MinAtar environment")
    ax.set_ylabel("Compared with Rainbow")
    ax.set_title(f"Rainbow - comparison | {setup} ({kind})\n"
                 "Difference in mean final return with 90% bootstrap CI")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            v = values[row, col]
            if np.isfinite(v):
                color = "white" if abs(v) >= 0.60 * vmax else "#181818"
                ax.text(col, row - 0.14, f"{v:+,.1f}",
                        ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold")
                lo, hi = ci_lowers[row, col], ci_uppers[row, col]
                ci_text = (f"({lo:+,.1f}, {hi:+,.1f})"
                           if np.isfinite(lo) and np.isfinite(hi)
                           else "CI unavailable")
                ax.text(col, row + 0.18, ci_text,
                        ha="center", va="center", fontsize=7.7, color=color)
            else:
                ax.text(col, row, "N/A", ha="center", va="center",
                        fontsize=9, color="#666666")

    # DQN is an overall baseline, not a single-component ablation.
    ax.axhline(5.5, color="#222222", lw=2)
    ax.set_xticks(np.arange(-0.5, len(games), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EFFECT_COMPARISONS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    bar = fig.colorbar(heatmap, ax=ax, fraction=0.033, pad=0.035)
    bar.set_label("Difference in mean final return (Rainbow - comparison)")
    fig.text(0.5, 0.015,
             "Top: effect estimate; bottom: 90% bootstrap CI for the difference. "
             "Raw return scales differ by game.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_plot(fig, output_path, dpi)


def export_rainbow_effects(data, frames_by_game, output, bootstrap_resamples,
                           bootstrap_seed, dpi):
    """Heatmaps per (env_type, setup), plus numeric and CI data in CSV form.

    The final frame for each environment is the latest one shared by all
    selected configurations/seeds (same convention as final violin plots).
    """
    buckets = defaultdict(dict)
    for key, seeds in data.items():
        kind, setup, exp, game = key
        buckets[(kind, setup)][(exp, game)] = seeds

    columns = [
        "env_type", "setup", "environment", "comparison", "comparison_type",
        "compared_experiment", "frame", "rainbow_n_seeds", "comparison_n_seeds",
        "rainbow_mean_return", "comparison_mean_return",
        "effect_rainbow_minus_comparison", "ci90_lower", "ci90_upper", "status",
    ]
    all_rows = []
    for (kind, setup), models in sorted(buckets.items()):
        games = sorted({game for _, game in models},
                       key=lambda g: (GAME_ORDER.index(pretty_game(g))
                                      if pretty_game(g) in GAME_ORDER else len(GAME_ORDER),
                                      pretty_game(g)))
        matrix = np.full((len(EFFECT_COMPARISONS), len(games)), np.nan)
        ci_lowers = np.full_like(matrix, np.nan)
        ci_uppers = np.full_like(matrix, np.nan)
        rows = []
        for j, game in enumerate(games):
            frame = frames_by_game.get((kind, game))
            rainbow = models.get(("rainbow", game))
            for i, (exp, label, comp_type) in enumerate(EFFECT_COMPARISONS):
                compared = models.get((exp, game))
                row = {
                    "env_type": kind, "setup": setup, "environment": game,
                    "comparison": label, "comparison_type": comp_type,
                    "compared_experiment": exp, "frame": frame if frame is not None else "",
                    "rainbow_n_seeds": len(rainbow) if rainbow else 0,
                    "comparison_n_seeds": len(compared) if compared else 0,
                    "rainbow_mean_return": "", "comparison_mean_return": "",
                    "effect_rainbow_minus_comparison": "",
                    "ci90_lower": "", "ci90_upper": "", "status": "",
                }
                if frame is None:
                    row["status"] = "no_shared_frame"
                elif rainbow is None:
                    row["status"] = "missing_rainbow"
                elif compared is None:
                    row["status"] = "missing_comparison"
                else:
                    r_scores = np.asarray([v[frame] for _, v in sorted(rainbow.items())])
                    c_scores = np.asarray([v[frame] for _, v in sorted(compared.items())])
                    delta = float(r_scores.mean() - c_scores.mean())
                    matrix[i, j] = delta
                    row.update({
                        "rainbow_mean_return": float(r_scores.mean()),
                        "comparison_mean_return": float(c_scores.mean()),
                        "effect_rainbow_minus_comparison": delta,
                        "status": "ok" if min(len(r_scores), len(c_scores)) >= 2 else "insufficient_seeds_for_ci",
                    })
                    if min(len(r_scores), len(c_scores)) >= 2:
                        low, high = bootstrap_difference_ci(
                            r_scores, c_scores, resamples=bootstrap_resamples,
                            rng=keyed_rng(bootstrap_seed,
                                          ((kind, setup, game, exp), "component_effect")),
                        )
                        row.update({"ci90_lower": low, "ci90_upper": high})
                        ci_lowers[i, j] = low
                        ci_uppers[i, j] = high
                rows.append(row)
        all_rows.extend(rows)
        stem = safe_name(f"{kind}_{setup}")
        dirpath = output / "rainbow_effect_heatmaps"
        draw_effect_heatmap(kind, setup, games, matrix, ci_lowers, ci_uppers,
                            dirpath / f"{stem}.png", dpi)
        # Matrix CSV has exactly the numerical cells displayed in the heatmap.
        matrix_columns = ["comparison", "comparison_type"] + games
        matrix_rows = [
            {"comparison": label, "comparison_type": comp_type,
             **{game: float(matrix[i, j]) if np.isfinite(matrix[i, j]) else ""
                for j, game in enumerate(games)}}
            for i, (_, label, comp_type) in enumerate(EFFECT_COMPARISONS)
        ]
        save_csv_rows(dirpath / f"{stem}_matrix.csv", matrix_rows, matrix_columns)
    save_csv_rows(output / "rainbow_effects_detailed.csv", all_rows, columns)



# Selected seed-count sensitivity analysis.
SEED_COUNTS = (3, 5, 10, 15, 20)
DEFAULT_SUBSAMPLE_REPEATS = 50
DEFAULT_SUBSAMPLE_BOOTSTRAPS = 10_000


def bootstrap_subsample_effects(x, y, n, repeats, resamples, rng):
    """Repeatedly subsample n runs per group *without* replacement.

    For each independent subset pair, use a seed-level bootstrap *with*
    replacement within each group. Returns:
        rows: array with columns [empirical_diff, boot_mean, ci90_lower, ci90_upper]
        pooled_mean: mean of all bootstrap differences across subsets/resamples
        pooled_low, pooled_high: 90% percentile CI from the pooled bootstrap differences

    When n equals both sample sizes, there is only one possible full-data
    subset and it is evaluated once (not spuriously repeated).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < n or len(y) < n:
        return np.empty((0, 4), dtype=float), np.nan, np.nan, np.nan
    if n < 2:
        raise ValueError("At least two seeds are needed for bootstrap intervals")
    if resamples < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    k = 1 if n == len(x) == len(y) else repeats

    # Each row contains a sample without replacement. Separate draws for
    # Rainbow and the comparison (no assumed pairing by numerical seed).
    ix = np.argsort(rng.random((k, len(x))), axis=1)[:, :n]
    iy = np.argsort(rng.random((k, len(y))), axis=1)[:, :n]
    xs = x[ix]
    ys = y[iy]
    empirical_diff = xs.mean(axis=1) - ys.mean(axis=1)

    means = np.empty(k, dtype=float)
    lower = np.empty(k, dtype=float)
    upper = np.empty(k, dtype=float)
    pooled_boots = []
    for start in range(0, k, 16):
        end = min(k, start + 16)
        batch = end - start
        bx = rng.integers(n, size=(batch, resamples, n))
        by = rng.integers(n, size=(batch, resamples, n))
        xr = np.take_along_axis(xs[start:end, None, :], bx, axis=2).mean(axis=2)
        yr = np.take_along_axis(ys[start:end, None, :], by, axis=2).mean(axis=2)
        bd = xr - yr
        means[start:end] = bd.mean(axis=1)
        lower[start:end] = np.quantile(bd, 0.05, axis=1)
        upper[start:end] = np.quantile(bd, 0.95, axis=1)
        pooled_boots.append(bd.reshape(-1))

    pooled_boots = np.concatenate(pooled_boots)
    pooled_mean = float(pooled_boots.mean())
    pooled_low = float(np.quantile(pooled_boots, 0.05))
    pooled_high = float(np.quantile(pooled_boots, 0.95))
    return np.column_stack([empirical_diff, means, lower, upper]), pooled_mean, pooled_low, pooled_high


def draw_subsampling_plot(kind, setup, game, panels, out_path, dpi):
    """Detailed seed-count sensitivity plot.

    Each faint point/whisker pair is one selected subset. The prominent line
    joins medians of bootstrap mean estimates across subsets.
    """
    fig, axs = plt.subplots(4, 2, figsize=(13, 15), sharex=True, squeeze=False)
    fig.suptitle(
        f"{pretty_game(game)} ({kind}, {setup}) | Seed-count sensitivity\n"
        "Rainbow - comparison: bootstrap mean difference and 90% CI",
        fontsize=15,
    )
    plot_axes = axs.flatten()
    for panel_id, (experiment, label, _) in enumerate(EFFECT_COMPARISONS):
        ax = plot_axes[panel_id]
        rows = panels.get(experiment, [])
        c = COLORS.get(experiment, "#4477AA")
        if not rows:
            ax.set_title(label + " (unavailable)")
            ax.set_xlim(1, 22)
            ax.grid(axis="y", alpha=0.18)
            continue
        ax.axhline(0, color="#666666", lw=0.9, alpha=0.6)
        full_effect = rows[0]["full_data_effect"]
        ax.axhline(full_effect, color="#333333", ls="--", lw=1.0,
                   alpha=0.7, label="Full-data effect")
        medians = []
        ns = []
        for n in SEED_COUNTS:
            rs = [r for r in rows if r["n_seeds_per_group"] == n]
            if not rs:
                continue
            jitter = keyed_rng(DEFAULT_BOOTSTRAP_SEED,
                               (kind, setup, game, experiment, n, "plot-jitter"))
            xs = n + jitter.uniform(-0.22, 0.22, size=len(rs)) if len(rs) > 1 else [n]
            for xpos, r in zip(xs, rs):
                ax.plot([xpos, xpos], [r["ci90_lower"], r["ci90_upper"]],
                        color=c, lw=0.65, alpha=0.12, zorder=1)
            ax.scatter(xs, [r["bootstrap_mean_effect"] for r in rs],
                       s=12, color=c, edgecolors="none", alpha=0.45, zorder=2)
            ns.append(n)
            medians.append(np.median([r["bootstrap_mean_effect"] for r in rs]))
        ax.plot(ns, medians, "o-", color=c, lw=2, ms=4,
                label="Median of subset estimates", zorder=4)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(SEED_COUNTS)
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=9)
        if panel_id % 2 == 0:
            ax.set_ylabel("Mean-return difference")
        if panel_id >= 5:
            ax.set_xlabel("Seeds per configuration")
    plot_axes[7].axis("off")
    handles = [
        Line2D([0], [0], color="#4477AA", lw=0.8, alpha=0.5,
               marker="o", markersize=4, label="Each subset: bootstrap mean + 90% CI"),
        Line2D([0], [0], color="#4477AA", lw=2, marker="o", markersize=4,
               label="Median of subset estimates"),
        Line2D([0], [0], color="#333333", ls="--", lw=1,
               label="Effect using all available seeds"),
    ]
    plot_axes[7].legend(handles=handles, loc="center", fontsize=10, frameon=False)
    plot_axes[7].text(0.04, 0.20,
        "At each n: subsets drawn without replacement;\n"
        "within each subset, bootstrap with replacement.\n"
        "n = full available sample: one unique subset.\n"
        "Faint vertical lines are individual 90% CIs.",
        transform=plot_axes[7].transAxes, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.3)
    save_plot(fig, out_path, dpi)


def draw_subsampling_forest_plot(kind, setup, game, summary_rows, out_path, dpi):
    """Compact forest-style plot: one point+CI per seed count and comparison."""
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    present = [item for item in EFFECT_COMPARISONS
               if any(r["compared_experiment"] == item[0] for r in summary_rows)]
    if not present:
        plt.close(fig)
        return

    present_counts = [n for n in SEED_COUNTS if any(r["n_seeds_per_group"] == n for r in summary_rows)]
    ybase = np.arange(len(present_counts), 0, -1, dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(present)) if len(present) > 1 else np.array([0.0])

    # Horizontal separators between seed-count groups, like a forest plot.
    # They are drawn behind all error bars so they cannot obscure the CIs.
    for boundary in np.arange(1.5, len(present_counts), 1.0):
        ax.axhline(boundary, color="#777777", linewidth=0.9,
                   alpha=0.55, zorder=0)
    # Subtle bottom boundary to frame the final group.
    ax.axhline(0.5, color="#777777", linewidth=0.9,
               alpha=0.55, zorder=0)
    ax.axvline(0.0, color="#333333", lw=1.0, ls="--", alpha=0.75, zorder=0)

    for offset, (experiment, label, comp_type) in zip(offsets, present):
        rows = {r["n_seeds_per_group"]: r for r in summary_rows if r["compared_experiment"] == experiment}
        color = COLORS.get(experiment, "#4477AA")
        xs = []
        ys = []
        xerr_low = []
        xerr_high = []
        for y, n in zip(ybase, present_counts):
            r = rows.get(n)
            if r is None:
                continue
            mean = r["pooled_bootstrap_mean_effect"]
            lo = r["pooled_ci90_lower"]
            hi = r["pooled_ci90_upper"]
            xs.append(mean)
            ys.append(y + offset)
            xerr_low.append(mean - lo)
            xerr_high.append(hi - mean)
        if xs:
            ax.errorbar(xs, ys, xerr=[xerr_low, xerr_high], fmt='o',
                        color=color, ecolor=color, elinewidth=1.8, capsize=3.5,
                        markersize=5.5, lw=0, label=label, zorder=3)

    ax.set_yticks(ybase)
    ax.set_yticklabels([f"{n} runs" for n in present_counts])
    ax.set_xlabel("Rainbow - comparison: bootstrapped mean difference")
    ax.set_ylabel("Independent runs per configuration")
    ax.set_title(
        f"{pretty_game(game)} ({kind}, {setup})\n"
        "Subsampling summary: pooled bootstrap mean and 90% CI"
    )
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    fig.text(0.5, 0.015,
             "For each run count n: repeatedly sample n seeds without replacement; "
             "within each subset, bootstrap with replacement. Points show the pooled "
             "bootstrap mean across subsets; whiskers show the pooled 90% bootstrap CI.",
             ha='center', fontsize=8.8, color='#555555')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_plot(fig, out_path, dpi)


def export_seed_count_subsampling(data, frames_by_game, output, repeats,
                                  resamples, bootstrap_seed, dpi):
    """Save reproducible per-subset bootstrap statistics, summaries, and plots."""
    buckets = defaultdict(dict)
    for key, seeds in data.items():
        kind, setup, exp, game = key
        buckets[(kind, setup, game)][exp] = seeds

    detailed_columns = [
        "env_type", "setup", "environment", "comparison", "comparison_type",
        "compared_experiment", "frame", "n_seeds_per_group", "subset_id",
        "rainbow_total_seeds", "comparison_total_seeds", "subset_repeats_requested",
        "bootstrap_resamples", "empirical_mean_effect", "bootstrap_mean_effect",
        "ci90_lower", "ci90_upper", "ci90_width", "ci_excludes_zero",
        "full_data_effect",
    ]
    summary_columns = [
        "env_type", "setup", "environment", "comparison", "comparison_type",
        "compared_experiment", "frame", "n_seeds_per_group", "n_subsets",
        "mean_bootstrap_effect_across_subsets", "median_bootstrap_effect_across_subsets",
        "p05_subset_bootstrap_effect", "p95_subset_bootstrap_effect",
        "mean_ci90_width", "median_ci90_width", "fraction_ci_excludes_zero",
        "full_data_effect", "bootstrap_resamples",
        "pooled_bootstrap_mean_effect", "pooled_ci90_lower", "pooled_ci90_upper",
    ]
    detail = []
    summary = []
    for (kind, setup, game), models in sorted(buckets.items()):
        frame = frames_by_game.get((kind, game))
        rainbow = models.get("rainbow")
        if frame is None or rainbow is None:
            continue
        r = np.asarray([scores[frame] for _, scores in sorted(rainbow.items())], dtype=float)
        panels = {}
        summary_for_plot = []
        for exp, label, ctype in EFFECT_COMPARISONS:
            other = models.get(exp)
            if other is None:
                continue
            y = np.asarray([scores[frame] for _, scores in sorted(other.items())], dtype=float)
            if min(len(r), len(y)) < SEED_COUNTS[0]:
                continue
            full_delta = float(r.mean() - y.mean())
            comparison_rows = []
            for n in SEED_COUNTS:
                if n > min(len(r), len(y)):
                    continue
                rng = keyed_rng(bootstrap_seed, (kind, setup, game, exp, n, "subsamples"))
                estimates, pooled_mean, pooled_low, pooled_high = bootstrap_subsample_effects(
                    r, y, n, repeats, resamples, rng,
                )
                for sid, (empirical, boot_mean, lo, hi) in enumerate(estimates, 1):
                    rec = {
                        "env_type": kind, "setup": setup, "environment": game,
                        "comparison": label, "comparison_type": ctype,
                        "compared_experiment": exp, "frame": frame,
                        "n_seeds_per_group": n, "subset_id": sid,
                        "rainbow_total_seeds": len(r), "comparison_total_seeds": len(y),
                        "subset_repeats_requested": repeats,
                        "bootstrap_resamples": resamples,
                        "empirical_mean_effect": float(empirical),
                        "bootstrap_mean_effect": float(boot_mean),
                        "ci90_lower": float(lo), "ci90_upper": float(hi),
                        "ci90_width": float(hi - lo),
                        "ci_excludes_zero": int(lo > 0 or hi < 0),
                        "full_data_effect": full_delta,
                    }
                    comparison_rows.append(rec)
                    detail.append(rec)
                rs = comparison_rows[-len(estimates):]
                bs = np.asarray([z["bootstrap_mean_effect"] for z in rs])
                widths = np.asarray([z["ci90_width"] for z in rs])
                summary_row = {
                    "env_type": kind, "setup": setup, "environment": game,
                    "comparison": label, "comparison_type": ctype,
                    "compared_experiment": exp, "frame": frame,
                    "n_seeds_per_group": n, "n_subsets": len(estimates),
                    "mean_bootstrap_effect_across_subsets": float(bs.mean()),
                    "median_bootstrap_effect_across_subsets": float(np.median(bs)),
                    "p05_subset_bootstrap_effect": float(np.quantile(bs, 0.05)),
                    "p95_subset_bootstrap_effect": float(np.quantile(bs, 0.95)),
                    "mean_ci90_width": float(widths.mean()),
                    "median_ci90_width": float(np.median(widths)),
                    "fraction_ci_excludes_zero": float(np.mean([z["ci_excludes_zero"] for z in rs])),
                    "full_data_effect": full_delta,
                    "bootstrap_resamples": resamples,
                    "pooled_bootstrap_mean_effect": float(pooled_mean),
                    "pooled_ci90_lower": float(pooled_low),
                    "pooled_ci90_upper": float(pooled_high),
                }
                summary.append(summary_row)
                summary_for_plot.append(summary_row)
            if comparison_rows:
                panels[exp] = comparison_rows
        if panels:
            fname = safe_name(f"{kind}_{setup}_{pretty_game(game)}") + ".png"
            draw_subsampling_plot(kind, setup, game, panels,
                                  output / "seed_count_subsampling" / fname, dpi)
        if summary_for_plot:
            fname = safe_name(f"{kind}_{setup}_{pretty_game(game)}") + ".png"
            draw_subsampling_forest_plot(kind, setup, game, summary_for_plot,
                                         output / "seed_count_subsampling_forest" / fname, dpi)
    save_csv_rows(output / "seed_count_subsampling_detailed.csv", detail, detailed_columns)
    save_csv_rows(output / "seed_count_subsampling_summary.csv", summary, summary_columns)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def pretty_game(name):
    return name.replace("MinAtar_", "").replace("-v0", "").replace("-v1", "")


def read_csv(path, env_type=None, setups=None):
    # data[(env_type, setup, experiment, environment)][seed][frame] = score
    data = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
        for lineno, row in enumerate(reader, 2):
            try:
                kind = row["env_type"].strip()
                setup = row["setup"].strip()
                if env_type and kind != env_type:
                    continue
                if setups and setup not in setups:
                    continue
                experiment = row["experiment"].strip()
                game = row["environment"].strip()
                seed = int(row["seed"])
                frame = int(row["frame"])
                score = float(row["mean_return"])
                if not all((kind, setup, experiment, game)) or frame < 0 or not np.isfinite(score):
                    raise ValueError("empty category, negative frame or non-finite score")
                key = (kind, setup, experiment, game)
                if frame in data[key][seed]:
                    raise ValueError(f"duplicate result for {key}, seed={seed}, frame={frame}")
                data[key][seed][frame] = score
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid CSV row {lineno}: {e}") from e
    if not data:
        raise ValueError("No evaluation rows found for the selected filters")
    return data


def experiment_label(key, duplicated_names):
    _, setup, experiment, _ = key
    name = LABELS.get(experiment, experiment.replace("_", " ").title())
    return f"{name} ({setup})" if experiment in duplicated_names else name


def group_by_environment(data):
    grouped = defaultdict(dict)
    for key, seeds in data.items():
        grouped[(key[0], key[3])][key] = seeds
    return dict(sorted(grouped.items()))


def palette_for(keys):
    extras = plt.get_cmap("tab20")
    unknown = sorted({key[2] for key in keys if key[2] not in COLORS})
    extra_colors = {name: extras(i % 20) for i, name in enumerate(unknown)}
    return {key: COLORS.get(key[2], extra_colors.get(key[2])) for key in keys}


def add_learning_curves(ax, models, color_map, bootstrap_resamples, bootstrap_seed):
    names = [key[2] for key in models]
    duplicated_names = {n for n in names if names.count(n) > 1}
    for idx, (key, seeds) in enumerate(sorted(models.items())):
        # Plot only the shared, observed frame range. No extrapolation.
        first = max(min(values) for values in seeds.values())
        last = min(max(values) for values in seeds.values())
        if first > last:
            print(f"[skip curve] {key}: seed evaluation ranges do not overlap")
            continue
        frames = sorted({frame for values in seeds.values() for frame in values if first <= frame <= last})
        if not frames:
            continue
        xframes = np.array(frames, dtype=np.float64)
        scores = np.array([
            np.interp(xframes, sorted(values), [values[f] for f in sorted(values)])
            for _, values in sorted(seeds.items())
        ])
        mean = scores.mean(axis=0)
        label = experiment_label(key, duplicated_names)
        color = color_map[key]
        if len(seeds) >= 2:
            lower, upper = bootstrap_mean_ci(
                scores,
                resamples=bootstrap_resamples,
                rng=keyed_rng(bootstrap_seed, (key, "learning_curve")),
            )
            ax.fill_between(xframes / 1e6, lower, upper,
                            color=color, alpha=0.16, linewidth=0)
        ax.plot(xframes / 1e6, mean, label=f"{label}",
                color=color, lw=1.8, marker=MARKERS[idx % len(MARKERS)],
                ms=4.5)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Mean evaluation return")
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}M"))


def final_common_frame(models):
    """Latest frame that appears for every model AND every seed."""
    common = None
    for seeds in models.values():
        for values in seeds.values():
            frames = set(values)
            common = frames if common is None else common & frames
    return max(common) if common else None


def add_final_violinplots(ax, models, color_map, frame,
                          bootstrap_resamples, bootstrap_seed):
    """Final scores across seeds; violin KDE + raw seeds + mean 90% bootstrap CI."""
    keys = sorted(models)
    names = [key[2] for key in keys]
    duplicated_names = {n for n in names if names.count(n) > 1}
    groups = []
    labels = []
    for key in keys:
        vals = np.asarray([v[frame] for _, v in sorted(models[key].items())], dtype=float)
        groups.append(vals)
        labels.append(experiment_label(key, duplicated_names))

    positions = np.arange(1, len(groups) + 1)
    for i, (key, values) in enumerate(zip(keys, groups)):
        x = positions[i]
        color = color_map[key]

        # KDE cannot be estimated from a single observation or constant scores.
        if len(values) >= 2 and np.ptp(values) > 1e-12:
            violin = ax.violinplot(
                [values], positions=[x], widths=0.75,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("#444444")
                body.set_linewidth(0.8)
                body.set_alpha(0.45)
        else:
            ax.hlines(values[0], x - 0.32, x + 0.32,
                      color=color, linewidth=5, alpha=0.45, zorder=2)

        # All individual runs remain visible; deterministic horizontal jitter.
        jitter_rng = keyed_rng(bootstrap_seed, (key, "jitter"))
        jitter = jitter_rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(x + jitter, values, s=26, color=color,
                   edgecolor="#333333", linewidth=0.45, alpha=0.85, zorder=3)

        median = float(np.median(values))
        ax.hlines(median, x - 0.25, x + 0.25,
                  color="#222222", linewidth=2, zorder=5)

        mean = float(np.mean(values))
        if len(values) >= 2:
            lower, upper = bootstrap_mean_ci(
                values, resamples=bootstrap_resamples,
                rng=keyed_rng(bootstrap_seed, (key, "final_mean")),
            )
            ax.errorbar(x, mean, yerr=[[mean - lower], [upper - mean]],
                        fmt="D", markersize=4.5, color="#ba2525",
                        ecolor="#ba2525", elinewidth=1.7, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=6)
        else:
            ax.plot(x, mean, "D", color="#ba2525", markersize=4.5, zorder=6)

    ax.legend(
        handles=[
            Line2D([0], [0], color="#ba2525", marker="D", markersize=5,
                   linewidth=1.7, label="Mean (90% bootstrap CI)"),
            Line2D([0], [0], color="#222222", linewidth=2, label="Median"),
            Line2D([0], [0], color="#666666", marker="o", markersize=5,
                   linestyle="None", label="Individual seeds"),
        ],
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True,
    )
    ax.set_xticks(positions, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Final mean evaluation return")
    ax.set_xlabel("Configuration")
    ax.set_title(f"Final evaluation at {frame:,} training steps")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return keys, groups


def save_plot(fig, destination, dpi):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("experiments/minatar/setup1/mean_returns.csv"),
                        help="Input CSV exported by to_csv.py")
    parser.add_argument("--output", type=Path, default=Path("minatar_evaluation/setup1"),
                        help="Directory for output plots and summary")
    parser.add_argument("--env-type", default=None, help="Optional filter, e.g. minatar")
    parser.add_argument("--setup", action="append", default=None,
                        help="Filter one or more setups; may be repeated")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bootstrap-resamples", type=int,
                        default=DEFAULT_BOOTSTRAP_RESAMPLES,
                        help="Number of bootstrap samples for 90%% percentile CIs (default: 10000)")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
                        help="Base RNG seed for reproducible bootstrap resampling (default: 42)")
    parser.add_argument("--subsample-repeats", type=int, default=DEFAULT_SUBSAMPLE_REPEATS,
                        help="Independent subsets per seed count (default: 50; full sample uses 1)")
    parser.add_argument("--subsample-bootstrap-resamples", type=int,
                        default=DEFAULT_SUBSAMPLE_BOOTSTRAPS,
                        help="Bootstrap replicates per subset (default: 2000)")
    parser.add_argument("--no-subsampling", action="store_true",
                        help="Skip the seed-count sensitivity plots and CSVs")
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.subsample_repeats < 1:
        parser.error("--subsample-repeats must be positive")
    if args.subsample_bootstrap_resamples < 100:
        parser.error("--subsample-bootstrap-resamples must be at least 100")

    data = read_csv(args.csv, env_type=args.env_type, setups=set(args.setup or []))
    by_game = group_by_environment(data)
    colors = palette_for(data)
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    frames_by_game = {}

    for (kind, game), models in by_game.items():
        stem = safe_name(f"{kind}_{game}")
        title = f"{pretty_game(game)} ({kind})"

        fig, ax = plt.subplots(figsize=(10, 5.5))
        add_learning_curves(ax, models, colors,
                            args.bootstrap_resamples, args.bootstrap_seed)
        ax.set_title(f"{title}: mean return across seeds (90% bootstrap CI)")
        ax.legend(fontsize=8, ncol=2)
        save_plot(fig, args.output / "learning_curves" / f"{stem}.png", args.dpi)

        frame = final_common_frame(models)
        frames_by_game[(kind, game)] = frame
        if frame is None:
            print(f"[skip final] {title}: no evaluation frame shared by all seeds/experiments")
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.45 * len(models)), 6))
        keys, groups = add_final_violinplots(
            ax, models, colors, frame,
            args.bootstrap_resamples, args.bootstrap_seed,
        )
        fig.suptitle(title, fontsize=14)
        save_plot(fig, args.output / "final_violinplots" / f"{stem}.png", args.dpi)
        for key, values in zip(keys, groups):
            if len(values) >= 2:
                lower, upper = bootstrap_mean_ci(
                    values, resamples=args.bootstrap_resamples,
                    rng=keyed_rng(args.bootstrap_seed, (key, "final_mean")),
                )
            else:
                lower, upper = "", ""
            all_summaries.append({
                "env_type": kind, "setup": key[1], "experiment": key[2],
                "environment": game, "frame": frame, "n_seeds": len(values),
                "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else "",
                "median": float(np.median(values)),
                "mean_ci90_lower": lower,
                "mean_ci90_upper": upper,
            })

    # Optional at-a-glance reference-style panel plot for all environments.
    if by_game:
        n = len(by_game)
        fig, axes = plt.subplots(1, n, figsize=(max(7.5, n * 4.2), 4.4), squeeze=False)
        for j, ((kind, game), models) in enumerate(by_game.items()):
            ax = axes[0, j]
            add_learning_curves(ax, models, colors,
                                args.bootstrap_resamples, args.bootstrap_seed)
            ax.set_title(pretty_game(game))
            ax.set_xlabel("Steps (millions)")
            if j != 0:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9)
        fig.tight_layout(rect=(0, 0.12 if handles else 0, 1, 1))
        save_plot(fig, args.output / "learning_curves_all_environments.png", args.dpi)

    export_rainbow_effects(data, frames_by_game, args.output,
                           args.bootstrap_resamples, args.bootstrap_seed, args.dpi)
    if not args.no_subsampling:
        export_seed_count_subsampling(
            data, frames_by_game, args.output,
            args.subsample_repeats, args.subsample_bootstrap_resamples,
            args.bootstrap_seed, args.dpi,
        )

    if all_summaries:
        path = args.output / "final_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0]))
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"[saved] {path}")
    print(f"Done: {len(data)} experiment/environment combinations, {len(by_game)} environments")


if __name__ == "__main__":
    main()
