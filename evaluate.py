"""Plot MinAtar experiment learning curves and final-return boxplots from a CSV.

Required CSV columns:
    env_type,setup,experiment,environment,seed,frame,mean_return

Examples:
    python evaluate.py --csv experiments/minatar/setup1/mean_returns.csv
    python evaluate.py --csv combined_mean_returns.csv --output minatar_evaluation
    python evaluate.py --csv combined_mean_returns.csv --setup setup1

Outputs include a learning curve and final boxplot for each environment,
plus a combined learning-curve grid and a final performance summary CSV.
Each final boxplot compares seeds at the last evaluation frame SHARED by
all selected experiment/seed combinations within that environment.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")  # Safe for scripts on headless servers and HPC clusters.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
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
CI_Z = NormalDist().inv_cdf(0.95)  # two-sided normal 90% CI


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


def add_learning_curves(ax, models, color_map):
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
            se = scores.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            ax.fill_between(xframes / 1e6, mean - CI_Z * se, mean + CI_Z * se,
                            color=color, alpha=0.16, linewidth=0)
        ax.plot(xframes / 1e6, mean, label=f"{label} (n={len(seeds)})",
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


def add_final_boxplots(ax, models, color_map, frame):
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
    bp = ax.boxplot(groups, positions=positions, widths=0.57, patch_artist=True, whis=1.5,
                    showfliers=False, medianprops={"color": "#232323", "linewidth": 1.7},
                    showmeans=True,      # Show mean
                    meanline=True,        # Mean as horizontal line
                    meanprops={"color": "red", "linestyle": "--","linewidth": 2,},
                    boxprops={"linewidth": 1.0},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0})

    legend_elements = [
        Line2D([0], [0], color="red", linestyle="--",
            linewidth=2, label="Mean"),
        Line2D([0], [0], color="black", linestyle="-",
            linewidth=2, label="Median"),
        Patch(facecolor="lightgray", edgecolor="black",
            alpha=0.5, label="IQR (25-75%)"),
        Line2D([0], [0], color="black", linestyle="-",
           linewidth=1.5, marker="|", markersize=10,
           label="Tukey fences (1.5 x IQR)"),
        Line2D([0], [0], color="black", marker="o",
            linestyle="None", markersize=5,
            label="Individual seeds"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
    )

    
    for i, (patch, key, values) in enumerate(zip(bp["boxes"], keys, groups)):
        patch.set_facecolor(color_map[key])
        patch.set_alpha(0.48)
        # Fixed random seed for repeatable point jitter.
        rng = np.random.default_rng(100 + i)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(positions[i] + jitter, values, s=29, c=[color_map[key]],
                   edgecolor="#333333", linewidth=0.5, alpha=0.9, zorder=3)
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
    args = parser.parse_args()

    data = read_csv(args.csv, env_type=args.env_type, setups=set(args.setup or []))
    by_game = group_by_environment(data)
    colors = palette_for(data)
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    grid_items = []

    for (kind, game), models in by_game.items():
        stem = safe_name(f"{kind}_{game}")
        title = f"{pretty_game(game)} ({kind})"

        fig, ax = plt.subplots(figsize=(10, 5.5))
        add_learning_curves(ax, models, colors)
        ax.set_title(f"{title}: mean return across seeds (90% CI)")
        ax.legend(fontsize=8, ncol=2)
        save_plot(fig, args.output / "learning_curves" / f"{stem}.png", args.dpi)

        frame = final_common_frame(models)
        if frame is None:
            print(f"[skip final] {title}: no evaluation frame shared by all seeds/experiments")
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.45 * len(models)), 6))
        keys, groups = add_final_boxplots(ax, models, colors, frame)
        fig.suptitle(title, fontsize=14)
        save_plot(fig, args.output / "final_boxplots" / f"{stem}.png", args.dpi)
        for key, values in zip(keys, groups):
            all_summaries.append({
                "env_type": kind, "setup": key[1], "experiment": key[2],
                "environment": game, "frame": frame, "n_seeds": len(values),
                "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else "",
                "median": float(np.median(values)),
            })
        grid_items.append(((kind, game), models))

    # Optional at-a-glance reference-style panel plot for all environments.
    if by_game:
        n = len(by_game)
        fig, axes = plt.subplots(1, n, figsize=(max(7.5, n * 4.2), 4.4), squeeze=False)
        for j, ((kind, game), models) in enumerate(by_game.items()):
            ax = axes[0, j]
            add_learning_curves(ax, models, colors)
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
