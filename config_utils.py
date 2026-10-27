# config_utils.py

from types import SimpleNamespace

from configs import (minatar_base) # TODO add more if needed
from configs.setups import SETUPS
from configs.ablations import ABLATIONS

BASE_CONFIGS = {
    "minatar": minatar_base,
    # "atari": atari_base, # TODO update if needed
}

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
    "ADAM_LR", "ADAM_EPS", "V_MIN", "V_MAX", "LOSS_FUNC", "WEIGHT_INIT",
    "HIDDEN_LAYERS", "CLIP_REWARDS", "USE_SUMMARY_WRITER"
)

def build_config(env_type, setup_name, ablation_name):
    if env_type not in BASE_CONFIGS:
        raise ValueError(f"Unknown environment type: {env_type}")

    if setup_name not in SETUPS:
        raise ValueError(f"Unknown setup: {setup_name}")

    if ablation_name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    base = BASE_CONFIGS[env_type]

    config = {
        name: value
        for name, value in vars(base).items()
        if name.isupper()
    }

    config.update(SETUPS[setup_name])
    config.update(ABLATIONS[ablation_name])

    config["ENV_TYPE"] = env_type
    config["SETUP"] = setup_name
    config["ABLATION"] = ablation_name
    config["NAME"] = f"{setup_name}_{ablation_name}"

    return SimpleNamespace(**config)

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
    if cfg.WEIGHT_INIT not in ("xavier_uniform", "kaiming_uniform"):
        raise ValueError(f"Unknown WEIGHT_INIT: {cfg.WEIGHT_INIT}")
    if cfg.LOSS_FUNC not in ("mse", "huber"):
        raise ValueError(f"Unknown LOSS_FUNC: {cfg.LOSS_FUNC}")

    # Adam settings.
    if cfg.ADAM_LR <= 0:
        raise ValueError("ADAM_LR must be positive.")
    if cfg.ADAM_EPS <= 0:
        raise ValueError("ADAM_EPS must be positive.")
    
    if not isinstance(cfg.CLIP_REWARDS, bool):
        raise ValueError("CLIP_REWARDS must be a boolean.")
    if not isinstance(cfg.USE_SUMMARY_WRITER, bool):
        raise ValueError("USE_SUMMARY_WRITER must be a boolean.")

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

def config_name(cfg):
    """Unique directory name for the selected ablation."""
    return f"{cfg.ENV_TYPE}/{cfg.SETUP}/{cfg.ABLATION}"