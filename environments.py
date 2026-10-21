
import numpy as np
import gymnasium as gym

try:
    import minatar
except ImportError:
    minatar = None

try:
    from minatar.gym import register_envs
except ImportError:
    register_envs = None

###################################################
# Environment helpers
###################################################

def env_reset(env, seed=None, noop_max=0):
    """Reset environment and optionally add a random NOOP start."""
    obs, _ = env.reset(seed=seed)

    if noop_max > 0:
        # Use the environment RNG so seeding stays reproducible.
        # For Atari, prefer AtariPreprocessing(noop_max=...) instead.
        num_noops = int(env.np_random.integers(1, noop_max + 1))

        for _ in range(num_noops):
            obs, _, terminated, truncated, _ = env.step(0)

            if terminated or truncated:
                obs, _ = env.reset()

    return obs


def env_step(env, action):
    """Gymnasium step -> obs, reward, done, info."""
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated or truncated, info

###################################################
# State processing
###################################################

class StateProcessor:
    """
    Convert observations to CHW uint8 format.

    MinAtar:
        HWC (10,10,C) -> CHW (C,10,10)

    Atari:
        assumes Atari preprocessing/frame stacking has already
        produced 84x84 observations.
    """

    def __init__(self, env_type):
        if env_type not in ("minatar", "atari"):
            raise ValueError(f"Unknown env_type: {env_type}")

        self.env_type = env_type


    def process(self, state):
        state = np.asarray(state)

        if state.ndim != 3:
            raise ValueError(
                f"Expected 3-D observation, got {state.shape}"
            )

        if self.env_type == "minatar":
            # MinAtar: HWC -> CHW
            state = np.transpose(state, (2, 0, 1))

        elif self.env_type == "atari":
            # Atari network expects 84x84 input
            if state.shape[-2:] == (84, 84):
                pass                        # already CHW
            elif state.shape[:2] == (84, 84):
                state = np.transpose(state, (2, 0, 1))  # HWC -> CHW
            else:
                raise ValueError(
                    "Atari observation must already be preprocessed "
                    f"to 84x84, got {state.shape}"
                )

        return state.astype(np.uint8, copy=False)

###################################################
# Small helpers
###################################################

def _minatar_game_name(env_id):
    """Convert e.g. MinAtar/SpaceInvaders-v1 -> space_invaders."""
    name = env_id.split("/")[-1].rsplit("-v", 1)[0]
    return {
        "Asterix": "asterix",
        "Breakout": "breakout",
        "Freeway": "freeway",
        "Seaquest": "seaquest",
        "SpaceInvaders": "space_invaders",
    }.get(name, name)

def register_environment_type(cfg):
    """Register packages needed by the selected environment type."""
    if cfg.ENV_TYPE == "minatar":
        if minatar is None:
            raise ImportError(
                "ENV_TYPE='minatar' requires the minatar package."
            )

        if cfg.MINATAR_BACKEND == "gym":
            if register_envs is None:
                raise ImportError(
                    "MINATAR_BACKEND='gym' requires minatar.gym."
                )

            register_envs()

class NativeMinAtarEnv(gym.Env):
    """Gymnasium API around minatar.Environment (full 6-action set)."""

    metadata = {}

    def __init__(self, env_id):
        super().__init__()
        self.game = minatar.Environment(env_name=_minatar_game_name(env_id))
        self.action_space = gym.spaces.Discrete(self.game.num_actions())
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=tuple(self.game.state_shape()), dtype=np.bool_
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.seed(seed)
        self.game.reset()
        return self.game.state(), {}

    def step(self, action):
        reward, terminated = self.game.act(int(action))
        return self.game.state(), float(reward), bool(terminated), False, {}


def make_env(env_id, env_type, seed, noop_max=0, minatar_backend="gym"):
    """Create a MinAtar or preprocessed Atari environment."""
    if env_type == "minatar":
        if minatar_backend == "native":
            env = NativeMinAtarEnv(env_id)
        else:
            env = gym.make(env_id)
    else:
        # Atari preprocessing also handles Atari no-op starts.
        env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=False,
        )
        # Gymnasium renamed FrameStack -> FrameStackObservation in newer versions.
        if hasattr(gym.wrappers, "FrameStackObservation"):
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        else:
            env = gym.wrappers.FrameStack(env, num_stack=4)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

###################################################
# Policy evaluation
###################################################

def evaluate_policy(env, q_estimator, state_processor,
                    eval_len, epsilon=0.0, seed=None, noop_max=0):

    state = state_processor.process(env_reset(env, seed=seed, noop_max=noop_max))

    episode_reward = episode_length = 0
    rewards, lengths = [], []

    for _ in range(eval_len):

        # Random action: skip neural-network inference
        if epsilon > 0.0 and np.random.random() < epsilon:
            action = int(np.random.randint(env.action_space.n))

        else:
            # Evaluation -> no NoisyNet noise
            q = q_estimator.predict(
                np.expand_dims(state, 0),
                use_noise=False
            )[0]

            action = int(np.argmax(q))

        next_state, reward, done, _ = env_step(env, action)
        next_state = state_processor.process(next_state)

        episode_reward += reward
        episode_length += 1

        if done:
            rewards.append(episode_reward)
            lengths.append(episode_length)

            # Do not reseed here; continue the environment RNG sequence
            state = state_processor.process(env_reset(env, noop_max=noop_max))
            episode_reward = episode_length = 0
        else:
            state = next_state

    return {
        "rewards": np.asarray(rewards, dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }