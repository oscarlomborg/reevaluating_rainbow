SETUPS = {
    "setup1": {
        "LOSS_FUNC": "huber",
        "HIDDEN_LAYERS": 0,
        "WEIGHT_INIT": "xavier_uniform",
    },

    "setup2": {
        "LOSS_FUNC": "huber",
        "HIDDEN_LAYERS": 0,
        "WEIGHT_INIT": "xavier_uniform",
        "ADAM_LR" : 1.0e-4,
        "ENVIRONMENTS" : ["MinAtar/Seaquest-v0"],
    },

    "setup3": {
        "LOSS_FUNC": "huber",
        "HIDDEN_LAYERS": 0,
        "WEIGHT_INIT": "xavier_uniform",
        "ENVIRONMENTS" : ["MinAtar/Seaquest-v0"],
        "CLIP_REWARDS" : True,
    },

    "setup4": {
            "LOSS_FUNC": "huber",
            "HIDDEN_LAYERS": 0,
            "WEIGHT_INIT": "xavier_uniform",
            "ADAM_LR" : 1.0e-4,
            "ENVIRONMENTS" : ["MinAtar/Seaquest-v0"],
            "CLIP_REWARDS" : True,
        }
    }
