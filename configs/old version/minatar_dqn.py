# configs/minatar_dqn.py

from configs.minatar_base import *

NAME = "dqn"

DOUBLE_DQN = False
N_STEP = 1
PRIORITIZED_REPLAY = False

# ("dqn", "dueling", "distributional", "distributional_dueling")
NETWORK_TYPE = "dqn"
NOISY = False