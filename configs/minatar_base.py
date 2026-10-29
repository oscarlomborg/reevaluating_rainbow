# configs/minatar_rainbow.py

ENV_TYPE = "minatar"

NOOP_MAX = 0 # TODO

DOUBLE_DQN = True
N_STEP = 3
PRIORITIZED_REPLAY = True

# ("dqn", "dueling", "distributional", "distributional_dueling")
NETWORK_TYPE = "distributional_dueling"
NOISY = True

NUM_ATOMS = 51
V_MIN = -100.0
V_MAX = 100.0

# Number of fully connected hidden layers after the convolution.
HIDDEN_LAYERS = 0

#-v1 for minimal action space, -v0 for full action space
ENVIRONMENTS = [
    "MinAtar/Asterix-v0",
    "MinAtar/Breakout-v0",
    "MinAtar/Freeway-v0",
    "MinAtar/Seaquest-v0",
    "MinAtar/SpaceInvaders-v0",
]

NUM_FRAMES = 10_000_000 # Agent steps

REPLAY_PERIOD = 4 # Agent steps before update
REPLAY_MEMORY_SIZE = 100_000
REPLAY_MEMORY_INIT_SIZE = 1_000

PER_ALPHA = 0.5
PER_BETA_START = 0.5 # TODO activate decaying beta
PER_BETA_FRAMES = NUM_FRAMES
PER_EPS = 1e-6

DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 1_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 250_000

EVAL_EVERY = 1_000_000
EVAL_LEN = 125_000

LOSS_FUNC = "huber" # "mse" or "huber"

WEIGHT_INIT = "xavier_uniform" # "kaiming_uniform" or "xavier_uniform"

ADAM_LR = 2.5e-4
ADAM_EPS = 3.125e-4

CLIP_REWARDS = True

USE_SUMMARY_WRITER = False