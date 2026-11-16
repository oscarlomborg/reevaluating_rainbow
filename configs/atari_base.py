from .minatar_base import *

ENV_TYPE = "atari"

NOOP_MAX = 0

ZERO_PADDING = False

ENVIRONMENTS = [
    "ALE/Breakout-v5",
    "ALE/Seaquest-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Asterix-v5",
    "ALE/Freeway-v5",
]

NUM_FRAMES = 10_000 # Agent steps

EVAL_EVERY = 5_000
EVAL_LEN = 125_000