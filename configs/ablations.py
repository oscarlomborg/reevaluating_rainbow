ABLATIONS = {
    "rainbow": {},

    "dqn": {
        "DOUBLE_DQN": False,
        "N_STEP": 1,
        "PRIORITIZED_REPLAY": False,
        "NETWORK_TYPE": "dqn",
        "NOISY": False,
    },

    "no_double": {
        "DOUBLE_DQN": False,
    },

    "no_nstep": {
        "N_STEP": 1,
    },

    "no_per": {
        "PRIORITIZED_REPLAY": False,
    },

    "no_dueling": {
        "NETWORK_TYPE": "distributional",
    },

    "no_distrib": {
        "NETWORK_TYPE": "dueling",
    },

    "no_noisy": {
        "NOISY": False,
    },
}