
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

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*environment MinAtar/.*-v0 is out of date.*",
    category=DeprecationWarning,
)

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

def make_env(env_id, env_type, seed, noop_max=0):
    """Create a MinAtar or preprocessed Atari environment."""
    if env_type == "minatar":
        if env_id not in gym.registry:
            register_envs()
        env = gym.make(env_id, sticky_action_prob=0.1, difficulty_ramping=True)
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
                    eval_len, epsilon=0.001, seed=None, noop_max=0, use_noise=False):

    state = state_processor.process(env_reset(env, seed=seed, noop_max=noop_max))

    episode_reward = episode_length = 0
    rewards, lengths = [], []

    for _ in range(eval_len):

        # Random action: skip neural-network inference
        if epsilon > 0.0 and np.random.random() < epsilon:
            action = int(np.random.randint(env.action_space.n))

        else:
            q = q_estimator.predict(
                np.expand_dims(state, 0),
                use_noise=use_noise
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