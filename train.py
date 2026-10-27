import argparse
import itertools
import multiprocessing as mp
import os
import random
from time import perf_counter

# One numerical thread per worker.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from agent import Estimator
from environments import StateProcessor, env_reset, make_env
from training import train_model
from config_utils import build_config, validate_config, config_name

###################################################
# Parallel worker
###################################################

def run_job(job):
    """Run one (environment, seed, setup, ablation) experiment."""
    env_id, seed, env_type, setup_name, ablation_name = job
    cfg = build_config(
        env_type=env_type,
        setup_name=setup_name,
        ablation_name=ablation_name,
    )
    validate_config(cfg)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    name = config_name(cfg)
    game_name = env_id.replace("/", "_")
    seed_dir = os.path.abspath(os.path.join("experiments", cfg.ENV_TYPE, cfg.SETUP, cfg.ABLATION, game_name, f"seed_{seed}"))
    os.makedirs(seed_dir, exist_ok=True)

    env = eval_env = q_estimator = target_estimator = None
    start = perf_counter()
    print(
        f"[START] env={env_id}, seed={seed}, config={name}, "
        f"pid={os.getpid()}, device={device}",
        flush=True,
    )

    try:
        # AtariPreprocessing receives NOOP_MAX directly.
        env = make_env(
            env_id, cfg.ENV_TYPE, seed, noop_max=cfg.NOOP_MAX
        )
        eval_env = make_env(
            env_id, cfg.ENV_TYPE, seed + 1, noop_max=cfg.NOOP_MAX
        )

        state_processor = StateProcessor(cfg.ENV_TYPE)

        # Reset once only to determine the network input shape.
        shape_noops = cfg.NOOP_MAX if cfg.ENV_TYPE == "minatar" else 0
        input_shape = state_processor.process(
            env_reset(env, seed=seed, noop_max=shape_noops)
        ).shape
        num_actions = env.action_space.n

        estimator_kwargs = dict(
            input_shape=input_shape,
            num_actions=num_actions,
            env_type=cfg.ENV_TYPE,
            network_type=cfg.NETWORK_TYPE,
            loss_func=cfg.LOSS_FUNC,
            noisy=cfg.NOISY,
            num_atoms=cfg.NUM_ATOMS,
            v_min=cfg.V_MIN,
            v_max=cfg.V_MAX,
            hidden_layers=cfg.HIDDEN_LAYERS,
            weight_init=cfg.WEIGHT_INIT,
            adam_lr=cfg.ADAM_LR,
            adam_eps=cfg.ADAM_EPS,
            device=device,
        )
        q_estimator = Estimator(
            **estimator_kwargs, scope="q_estimator",
            summaries_dir=seed_dir if cfg.USE_SUMMARY_WRITER else None
        )
        target_estimator = Estimator(
            **estimator_kwargs, scope="target_q"
        )

        stats = train_model(
            env=env,
            eval_env=eval_env,
            env_id=env_id,
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            state_processor=state_processor,
            cfg=cfg,
            seed_dir=seed_dir,
            seed=seed,
        )

        elapsed = perf_counter() - start
        print(
            f"[DONE] env={env_id}, seed={seed}, episodes={len(stats.episode_rewards)}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        return {
            "env_id": env_id,
            "seed": seed,
            "seed_dir": seed_dir,
            "episodes": len(stats.episode_rewards),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        print(
            f"[CRASH] env={env_id}, seed={seed}: {e}",
            flush=True
        )
        raise

    finally:
        for estimator in (q_estimator, target_estimator):
            writer = getattr(estimator, "summary_writer", None)
            if writer is not None:
                writer.flush()
                writer.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

###################################################
# Main
###################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-type", required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--ablation", required=True)
    
    parser.add_argument(
        "--max-procs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())),
    )
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    cfg = build_config(
        env_type=args.env_type,
        setup_name=args.setup,
        ablation_name=args.ablation,
    )
    validate_config(cfg)

    if args.max_procs < 1 or args.seeds < 1:
        raise ValueError("max-procs and seeds must be at least 1")

    jobs = [
        (env_id, seed, args.env_type, args.setup, args.ablation)
        for env_id, seed in itertools.product(cfg.ENVIRONMENTS, range(args.seeds))
    ]
    n_procs = min(args.max_procs, len(jobs))
    name = config_name(cfg)

    print(
        f"--- {name}: {len(cfg.ENVIRONMENTS)} environments x "
        f"{args.seeds} seeds with {n_procs} processes ---",
        flush=True,
    )

    start = perf_counter()

    if n_procs == 1:
            # Run ALL environment/seed pairs, not just the first job.
            results = [run_job(job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")
        # Reuse workers for several runs to avoid repeated Python/PyTorch startup.
        # For memory leaks set maxtasksperchild (=1 e.g.)
        with ctx.Pool(n_procs) as pool:
            results = pool.map(run_job, jobs, chunksize=1)

    print(
        f"--- Finished {len(jobs)} runs in {perf_counter() - start:.1f} seconds ---",
        flush=True,
    )
