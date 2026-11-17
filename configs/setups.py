SETUPS = {
    "base": {
        "HIDDEN_LAYERS": 0,
        "ZERO_PADDING": True,
    },
    
    "setup1": {
        "HIDDEN_LAYERS": 0,
        "ZERO_PADDING": False,
    },

    "setup2": {
        "HIDDEN_LAYERS": 1,
        "ZERO_PADDING": True,
    },

    # conventional Atari DQN/Rainbow CNN architecture
    "conventional": {
        "HIDDEN_LAYERS": 1,
        "ZERO_PADDING": False,
    }
}
