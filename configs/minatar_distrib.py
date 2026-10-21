# configs/minatar_distrib.py

from configs.minatar_base import *

NAME = "no-distrib"

# ("dqn", "dueling", "distributional", "distributional_dueling")
NETWORK_TYPE = "dueling"