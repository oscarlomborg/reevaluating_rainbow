"""Run all seeds for one environment/ablation combination per LSF array task.

The LSF array index selects one combination of:
    environment x ablation

The setup is selected once with --setup.

Example:
    5 environments x 8 ablations = 40 LSF array jobs.

Within each array job, --seeds controls how many seeds are run and
--seed-procs controls how many of those seeds run concurrently.
"""

import argparse
import multiprocessing as mp
import os
from time import perf_counter

from config_utils import build_config, validate_config
from configs.ablations import ABLATIONS
from configs.setups import SETUPS
from train import run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--env-type", required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--ablations", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--seed-procs",
        type=int,
        default=2,
        help="Number of seed workers to run concurrently inside each LSF job",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="1-based environment/ablation index (defaults to LSB_JOBINDEX)",
    )

    args = parser.parse_args()

    if args.seeds < 1 or args.seed_procs < 1:
        parser.error("--seeds and --seed-procs must be positive")

    setup_name = args.setup
    ablation_names = args.ablations

    if setup_name not in SETUPS:
        parser.error(f"Unknown setup: {setup_name}")

    unknown_ablations = [
        name for name in ablation_names
        if name not in ABLATIONS
    ]
    if unknown_ablations:
        parser.error(
            f"Unknown ablation(s): {', '.join(unknown_ablations)}"
        )

    # Build one config to obtain the environment list.
    reference_cfg = build_config(
        env_type=args.env_type,
        setup_name=setup_name,
        ablation_name=ablation_names[0],
    )
    validate_config(reference_cfg)

    environments = list(reference_cfg.ENVIRONMENTS)

    # One LSF array job per environment x ablation combination.
    total_combinations = (
        len(environments)
        * len(ablation_names)
    )

    raw_index = (
        args.index
        if args.index is not None
        else os.environ.get("LSB_JOBINDEX")
    )

    if raw_index is None:
        parser.error(
            "Supply --index or submit as an LSF job array."
        )

    try:
        index = int(raw_index)
    except (ValueError, TypeError):
        parser.error(f"Invalid LSF array index: {raw_index!r}")

    if not 1 <= index <= total_combinations:
        parser.error(
            f"Index {index} out of range: 1..{total_combinations}"
        )

    zero_index = index - 1

    env_index = zero_index % len(environments)
    ablation_index = zero_index // len(environments)

    env_id = environments[env_index]
    ablation_name = ablation_names[ablation_index]

    # Validate the exact resolved config selected by this array task.
    cfg = build_config(
        env_type=args.env_type,
        setup_name=setup_name,
        ablation_name=ablation_name,
    )
    validate_config(cfg)

    jobs = [
        (
            env_id,
            seed,
            args.env_type,
            setup_name,
            ablation_name,
        )
        for seed in range(args.seeds)
    ]

    n_procs = min(args.seed_procs, args.seeds)

    print(
        f"[ARRAY] index={index}/{total_combinations} "
        f"env={env_id} "
        f"setup={setup_name} "
        f"ablation={ablation_name} "
        f"seeds={args.seeds} "
        f"concurrent_seed_workers={n_procs}",
        flush=True,
    )

    start = perf_counter()

    if n_procs == 1:
        results = [run_job(job) for job in jobs]
    else:
        # Spawn is safer with PyTorch than fork.
        with mp.get_context("spawn").Pool(processes=n_procs) as pool:
            results = pool.map(run_job, jobs, chunksize=1)

    print(
        f"[ARRAY DONE] "
        f"env={env_id} "
        f"setup={setup_name} "
        f"ablation={ablation_name} "
        f"completed_seeds={[result['seed'] for result in results]} "
        f"elapsed={perf_counter() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
