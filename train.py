import argparse
import importlib.util
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

try:
    import minatar
except ImportError:
    minatar = None

try:
    from minatar.gym import register_envs
except ImportError:
    register_envs = None

from agent import Estimator
from environments import StateProcessor, env_reset, make_env
from training import train_model

###################################################
# Config helpers
###################################################

REQUIRED_CONFIG = (
    "ENV_TYPE", "NAME", "DOUBLE_DQN", "N_STEP", "PRIORITIZED_REPLAY",
    "NETWORK_TYPE", "NOISY", "NUM_ATOMS", "ENVIRONMENTS",
    "NUM_FRAMES", "REPLAY_PERIOD",
    "REPLAY_MEMORY_SIZE", "REPLAY_MEMORY_INIT_SIZE",
    "PER_ALPHA", "PER_BETA_START", "PER_BETA_FRAMES", "PER_EPS",
    "DISCOUNT_FACTOR", "BATCH_SIZE", "TARGET_UPDATE_EVERY",
    "EPSILON_START", "EPSILON_END", "EPSILON_DECAY_STEPS",
    "EVAL_EVERY", "EVAL_LEN", "NOOP_MAX",
    "ADAM_LR", "ADAM_EPS", "V_MIN", "V_MAX", "LOSS_FUNC", "WEIGHT_INIT"
)


def load_config(path):
    """Load a Python config file as a module."""
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config: {path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Backward-compatible defaults.
    if not hasattr(cfg, "HIDDEN_LAYERS"):
        cfg.HIDDEN_LAYERS = 1
    if not hasattr(cfg, "MINATAR_BACKEND"):
        cfg.MINATAR_BACKEND = "gym"

    return cfg


def validate_config(cfg):
    """Fail early if a required setting is missing or invalid."""
    missing = [name for name in REQUIRED_CONFIG if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f"Config is missing: {', '.join(missing)}")

    if cfg.ENV_TYPE not in ("minatar", "atari"):
        raise ValueError("ENV_TYPE must be 'minatar' or 'atari'.")
    if cfg.NETWORK_TYPE not in ("dqn", "dueling", "distributional", "distributional_dueling"):
        raise ValueError(f"Unknown NETWORK_TYPE: {cfg.NETWORK_TYPE}")
    if not isinstance(cfg.HIDDEN_LAYERS, int) or cfg.HIDDEN_LAYERS < 0:
        raise ValueError("HIDDEN_LAYERS must be a nonnegative integer.")
    if cfg.MINATAR_BACKEND not in ("gym", "native"):
        raise ValueError("MINATAR_BACKEND must be 'gym' or 'native'.")
    if cfg.N_STEP < 1:
        raise ValueError("N_STEP must be at least 1.")
    if cfg.REPLAY_PERIOD < 1:
        raise ValueError("REPLAY_PERIOD must be at least 1.")
    if cfg.TARGET_UPDATE_EVERY < 1 or cfg.EVAL_EVERY < 1:
        raise ValueError("TARGET_UPDATE_EVERY and EVAL_EVERY must be at least 1.")
    if cfg.EPSILON_DECAY_STEPS < 1:
        raise ValueError("EPSILON_DECAY_STEPS must be at least 1.")
    if cfg.NOOP_MAX < 0:
        raise ValueError("NOOP_MAX cannot be negative.")
    if cfg.WEIGHT_INIT not in ("xavier_uniform", "pytorch_default"):
        raise ValueError(f"Unknown WEIGHT_INIT: {cfg.WEIGHT_INIT}")
    if cfg.LOSS_FUNC not in ("mse", "huber"):
        raise ValueError(f"Unknown LOSS_FUNC: {cfg.LOSS_FUNC}")

    # Adam settings.
    if cfg.ADAM_LR <= 0:
        raise ValueError("ADAM_LR must be positive.")
    if cfg.ADAM_EPS <= 0:
        raise ValueError("ADAM_EPS must be positive.")

    # C51 requires an ordered support with at least two atoms.
    if "distributional" in cfg.NETWORK_TYPE:
        if cfg.NUM_ATOMS < 2:
            raise ValueError("NUM_ATOMS must be at least 2 for C51.")
        if cfg.V_MIN >= cfg.V_MAX:
            raise ValueError("V_MIN must be smaller than V_MAX.")
    if cfg.BATCH_SIZE < 1 or cfg.REPLAY_MEMORY_INIT_SIZE < cfg.BATCH_SIZE:
        raise ValueError("REPLAY_MEMORY_INIT_SIZE must be >= BATCH_SIZE >= 1.")
    if cfg.REPLAY_MEMORY_SIZE < cfg.REPLAY_MEMORY_INIT_SIZE:
        raise ValueError("REPLAY_MEMORY_SIZE must be >= REPLAY_MEMORY_INIT_SIZE.")
    if not cfg.ENVIRONMENTS:
        raise ValueError("ENVIRONMENTS cannot be empty.")


def register_environment_type(cfg):
    """Register MinAtar only when the selected config needs it."""
    if cfg.ENV_TYPE == "minatar":
        if minatar is None:
            raise ImportError("ENV_TYPE='minatar' requires the minatar package.")
        if cfg.MINATAR_BACKEND == "gym":
            if register_envs is None:
                raise ImportError("MINATAR_BACKEND='gym' requires minatar.gym.")
            register_envs()


def config_name(cfg):
    """Unique directory name for the selected ablation."""
    return f"{cfg.ENV_TYPE}_{cfg.NAME}"

###################################################
# Parallel worker
###################################################

def run_job(job):
    """Run one (environment, seed, config) experiment."""
    env_id, seed, config_path = job
    cfg = load_config(config_path)
    validate_config(cfg)
    register_environment_type(cfg)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    name = config_name(cfg)
    game_name = env_id.replace("/", "_")
    seed_dir = os.path.abspath(os.path.join("experiments", name, game_name, f"seed_{seed}"))
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
            env_id, cfg.ENV_TYPE, seed, noop_max=cfg.NOOP_MAX,
            minatar_backend=cfg.MINATAR_BACKEND,
        )
        eval_env = make_env(
            env_id, cfg.ENV_TYPE, seed + 1, noop_max=cfg.NOOP_MAX,
            minatar_backend=cfg.MINATAR_BACKEND,
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
            **estimator_kwargs, scope="q_estimator", summaries_dir=seed_dir
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
    parser.add_argument("--config", required=True, help="Path to Python config file")
    parser.add_argument(
        "--max-procs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())),
    )
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    validate_config(cfg)

    if args.max_procs < 1 or args.seeds < 1:
        raise ValueError("max-procs and seeds must be at least 1")

    jobs = [
        (env_id, seed, config_path)
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
