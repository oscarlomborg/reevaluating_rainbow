import os
import gymnasium as gym
import numpy as np
import torch

from gymnasium.wrappers import RecordVideo
from minatar.gym import register_envs

from SETUP import Estimator, StateProcessor

register_envs()


class ScaleRender(gym.Wrapper):
    """Upscale MinAtar RGB frames for video recording."""

    def __init__(self, env, scale=30):
        super().__init__(env)
        self.scale = scale

    def render(self):
        frame = self.env.render()

        if frame is None:
            return None

        frame = np.asarray(frame)

        # MinAtar often renders floats in [0, 1].
        if np.issubdtype(frame.dtype, np.floating):
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.repeat(
            np.repeat(frame, self.scale, axis=0),
            self.scale,
            axis=1,
        )


# --------------------------------------------------
# Settings
# --------------------------------------------------

MODEL_PATH = (
    "experiments/minatar_rainbow/"
    "MinAtar_Breakout-v1/"
    "seed_0/checkpoints/model.pt"
)

ENV_ID = "MinAtar/Breakout-v1"

VIDEO_DIR = "videos"
SEED = 1

device = torch.device("cpu")


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

ckpt = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

cfg = ckpt["config"]

print("Checkpoint config:", cfg)
print("Training step:", ckpt["total_t"])


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = gym.make(
    ENV_ID,
    render_mode="rgb_array",
)

env = ScaleRender(env, scale=30)

env = RecordVideo(
    env,
    video_folder=VIDEO_DIR,
    episode_trigger=lambda episode_id: True,
    name_prefix="model_episode",
)

processor = StateProcessor("minatar")

obs, _ = env.reset(seed=SEED)
state = processor.process(obs)

input_shape = state.shape
num_actions = env.action_space.n


# --------------------------------------------------
# Recreate model
# --------------------------------------------------

estimator = Estimator(
    input_shape=input_shape,
    num_actions=num_actions,
    env_type="minatar",
    network_type=cfg["network_type"],
    noisy=cfg["noisy"],
    num_atoms=cfg["num_atoms"],
    v_min=cfg.get("v_min", -100.0),
    v_max=cfg.get("v_max", 100.0),
    device=device,
)

estimator.model.load_state_dict(
    ckpt["q_estimator"]
)

estimator.model.eval()


# --------------------------------------------------
# Run one greedy episode
# --------------------------------------------------

episode_return = 0.0
episode_length = 0

terminated = False
truncated = False

while not (terminated or truncated):

    q_values = estimator.predict(
        state[None, ...],
        use_noise=False,
    )[0]

    action = int(np.argmax(q_values))

    obs, reward, terminated, truncated, info = env.step(action)

    state = processor.process(obs)

    episode_return += reward
    episode_length += 1
    if (terminated or truncated):
        print(f'Terminated: {terminated}, Truncated: {truncated}')

env.close()

print(f"Return: {episode_return}")
print(f"Length: {episode_length}")
print(f"Video: {os.path.abspath(VIDEO_DIR)}")