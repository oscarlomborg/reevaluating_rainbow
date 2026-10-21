import ctypes
import itertools
import os
import sys
from collections import deque, namedtuple
from time import perf_counter

import numpy as np
import psutil
import torch

from agent import (
    ModelParametersCopier,
    build_c51_targets,
    build_dqn_targets,
    make_epsilon_greedy_policy,
)

from environments import (
    env_reset,
    env_step,
    evaluate_policy,
)

from replay import (
    ReplayBuffer,
    Transition,
    add_n_step,
    replay_priorities,
)


EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])

###################################################
# Training
###################################################

def train_model(
    env, eval_env, env_id, q_estimator, target_estimator,
    state_processor, cfg, seed_dir, seed=0,
):
    """Train one configurable DQN variant on one environment and seed."""

    replay = ReplayBuffer(
        cfg.REPLAY_MEMORY_SIZE,
        prioritized=cfg.PRIORITIZED_REPLAY,
        alpha=cfg.PER_ALPHA,
        eps=cfg.PER_EPS,
    )
    nstep_buffer = deque()
    copier = ModelParametersCopier(q_estimator, target_estimator)
    current_process = psutil.Process()
    current_process.cpu_percent(None)  # Prime CPU measurement.

    # Work around large amounts of free native heap memory retained by glibc.
    # Keep this independent of Python GC and tracemalloc. Disable by setting
    # MALLOC_TRIM_EVERY=0, or tune the interval in environment steps.
    trim_every = int(os.environ.get("MALLOC_TRIM_EVERY", "1000"))
    if trim_every < 0:
        raise ValueError("MALLOC_TRIM_EVERY must be nonnegative")
    malloc_trim = None
    if sys.platform == "linux" and trim_every:
        try:
            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
        except (OSError, AttributeError):
            print("[WARN] glibc malloc_trim unavailable; trimming disabled", flush=True)

    checkpoint_dir = os.path.join(seed_dir, "model")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_t = 0

    # Print five progress milestones per environment/seed (including warm-up).
    progress_start = perf_counter()
    progress_milestones = [
        (percent, (cfg.NUM_FRAMES * percent + 99) // 100)
        for percent in (20, 40, 60, 80, 100)
    ]
    next_progress = 0

    def report_progress():
        nonlocal next_progress
        while (next_progress < len(progress_milestones)
               and total_t >= progress_milestones[next_progress][1]):
            percent, _ = progress_milestones[next_progress]
            elapsed = perf_counter() - progress_start
            print(
                f"[PROGRESS] env={env_id} seed={seed} "
                f"{percent}% ({total_t:,}/{cfg.NUM_FRAMES:,} steps) "
                f"elapsed={int(elapsed // 3600):02d}:{int(elapsed % 3600 // 60):02d}",
                flush=True,
            )
            next_progress += 1

    episode_lengths, episode_rewards = [], []
    evaluation_frames, evaluation_returns, evaluation_mean_returns = [], [], []

    # Start every run from a fresh model.
    copier.make()

    next_eval_t = ((total_t // cfg.EVAL_EVERY) + 1) * cfg.EVAL_EVERY
    epsilons = np.linspace(
        cfg.EPSILON_START, cfg.EPSILON_END, cfg.EPSILON_DECAY_STEPS, dtype=np.float32
    )
    policy = make_epsilon_greedy_policy(
        q_estimator, env.action_space.n, use_noisy_exploration=cfg.NOISY
    )

    # MinAtar uses our reset helper for no-op starts.
    # AtariPreprocessing already performs Atari no-op starts.
    reset_noop_max = cfg.NOOP_MAX if cfg.ENV_TYPE == "minatar" else 0

    # --------------------------------------------------
    # Populate replay memory.
    # --------------------------------------------------
    state = state_processor.process(
        env_reset(env, seed=seed, noop_max=reset_noop_max)
    )
    while len(replay) < cfg.REPLAY_MEMORY_INIT_SIZE and total_t < cfg.NUM_FRAMES:
        # NoisyNet replaces epsilon-greedy exploration.
        epsilon = 0.0 if cfg.NOISY else 1.0
        action = policy(state, epsilon)
        next_state, reward, done, _ = env_step(env, action)
        next_state = state_processor.process(next_state)

        add_n_step(
            nstep_buffer, replay,
            Transition(state, action, reward, next_state, done),
            cfg.N_STEP, cfg.DISCOUNT_FACTOR,
        )
        total_t += 1
        if malloc_trim is not None and total_t % trim_every == 0:
            malloc_trim(0)
        report_progress()
        # Seed only the first reset; later resets continue the RNG sequence.
        state = (
            state_processor.process(env_reset(env, noop_max=reset_noop_max))
            if done else next_state
        )

    # Discard an unfinished warm-up prefix before measured episodes.
    nstep_buffer.clear()

    # Replay memory is now populated.
    # Epsilon decay starts from this point.
    learning_start_t = total_t

    # --------------------------------------------------
    # Main loop.
    # --------------------------------------------------
    i_episode = 0
    beta = cfg.PER_BETA_START  # Safe initial value for logging.

    while total_t < cfg.NUM_FRAMES:
        # Start a new episode without reseeding the environment.
        state = state_processor.process(env_reset(env, noop_max=reset_noop_max))
        nstep_buffer.clear()
        episode_reward = episode_length = 0

        for _ in itertools.count():

            # Number of environment steps since learning started.
            training_t = total_t - learning_start_t

            # NoisyNet exploration uses epsilon = 0.
            epsilon = 0.0 if cfg.NOISY else epsilons[min(training_t, cfg.EPSILON_DECAY_STEPS - 1)]

            if total_t % cfg.TARGET_UPDATE_EVERY == 0:
                copier.make()

            action = policy(state, epsilon)
            next_state, reward, done, _ = env_step(env, action)
            next_state = state_processor.process(next_state)

            add_n_step(
                nstep_buffer, replay,
                Transition(state, action, reward, next_state, done),
                cfg.N_STEP, cfg.DISCOUNT_FACTOR,
            )

            episode_reward += reward
            episode_length += 1

            if total_t % cfg.REPLAY_PERIOD == 0:

                # Sample uniformly or with PER.
                # beta = min(
                #     1.0,
                #     cfg.PER_BETA_START
                #     + total_t * (1.0 - cfg.PER_BETA_START) / max(1, cfg.PER_BETA_FRAMES),
                # ) #TODO activate decaying beta
                samples, indices, weights = replay.sample(cfg.BATCH_SIZE, beta)
                states_b, actions_b, rewards_b, next_states_b, dones_b = map(np.array, zip(*samples))

                # n-step target uses gamma^n; terminal shortened prefixes do not bootstrap.
                gamma_n = cfg.DISCOUNT_FACTOR ** cfg.N_STEP
                target_fn = build_c51_targets if q_estimator.distributional else build_dqn_targets
                targets = target_fn(
                    q_estimator, target_estimator, next_states_b,
                    rewards_b, dones_b, gamma_n, cfg.DOUBLE_DQN,
                )

                # PER: priority controls sampling; IS weights correct the update bias.
                priorities = None
                if cfg.PRIORITIZED_REPLAY:
                    priorities = replay_priorities(
                        q_estimator, states_b, actions_b, targets, cfg.PER_EPS
                    )

                q_estimator.update(
                    states_b, actions_b, targets,
                    weights=weights if cfg.PRIORITIZED_REPLAY else None,
                )
                if cfg.PRIORITIZED_REPLAY:
                    replay.update_priorities(indices, priorities)

            total_t += 1
            if malloc_trim is not None and total_t % trim_every == 0:
                malloc_trim(0)
            report_progress()

            # Validation.
            if total_t >= next_eval_t:
                result = evaluate_policy(
                    eval_env, q_estimator, state_processor,
                    eval_len=cfg.EVAL_LEN, seed=seed + 1,
                    noop_max=reset_noop_max,
                )
                eval_rewards, eval_lengths = result["rewards"], result["lengths"]
                mean_return = float(np.mean(eval_rewards)) if len(eval_rewards) else np.nan

                evaluation_frames.append(total_t)
                evaluation_returns.append(eval_rewards.copy())
                evaluation_mean_returns.append(mean_return)
                np.savez(
                    os.path.join(seed_dir, "evaluation.npz"),
                    frames=np.asarray(evaluation_frames, dtype=np.int64),
                    mean_returns=np.asarray(evaluation_mean_returns, dtype=np.float32),
                    raw_returns=np.asarray(evaluation_returns, dtype=object),
                    seed=seed,
                    game=env_id,
                )

                writer = q_estimator.summary_writer
                if writer is not None and len(eval_rewards):
                    writer.add_scalar("eval/return_mean", mean_return, total_t)
                    writer.add_scalar("eval/return_median", float(np.median(eval_rewards)), total_t)
                    writer.add_scalar("eval/episode_length_mean", float(np.mean(eval_lengths)), total_t)
                    writer.add_scalar("eval/num_episodes", len(eval_rewards), total_t)
                    writer.flush()

                next_eval_t = ((total_t // cfg.EVAL_EVERY) + 1) * cfg.EVAL_EVERY

            if done or total_t >= cfg.NUM_FRAMES:
                break
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        i_episode += 1

        # TensorBoard episode/system statistics.
        writer = q_estimator.summary_writer
        if writer is not None:
            writer.add_scalar("episode/epsilon", float(epsilon), i_episode)
            writer.add_scalar("episode/reward", float(episode_reward), i_episode)
            writer.add_scalar("episode/length", int(episode_length), i_episode)
            mem = current_process.memory_info()
            writer.add_scalar("system/memory_rss_mb", mem.rss / 1024**2, total_t)
            writer.add_scalar("system/memory_vms_mb", mem.vms / 1024**2, total_t)
            if cfg.PRIORITIZED_REPLAY:
                writer.add_scalar("replay/beta", beta, total_t)
            writer.flush()

    # Save model
    torch.save(
        {
            "q_estimator": q_estimator.model.state_dict(),
            "target_estimator": target_estimator.model.state_dict(),
            "optimizer": q_estimator.optimizer.state_dict(),
            "global_step": q_estimator.global_step,
            "total_t": total_t,
            "episode": i_episode,
            "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
            "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
            "config": {
                "env_type": cfg.ENV_TYPE,
                "network_type": cfg.NETWORK_TYPE,
                "loss_func": cfg.LOSS_FUNC,
                "double_dqn": cfg.DOUBLE_DQN,
                "n_step": cfg.N_STEP,
                "prioritized_replay": cfg.PRIORITIZED_REPLAY,
                "noisy": cfg.NOISY,
                "num_atoms": cfg.NUM_ATOMS,
                "v_min": cfg.V_MIN,
                "v_max": cfg.V_MAX,
                "adam_lr": cfg.ADAM_LR,
                "adam_eps": cfg.ADAM_EPS,
                "hidden_layers": cfg.HIDDEN_LAYERS,
                "weight_init": cfg.WEIGHT_INIT,
                "minatar_backend": cfg.MINATAR_BACKEND,
            },
        },
        checkpoint_path,
    )

    stats = EpisodeStats(
        np.asarray(episode_lengths, dtype=np.int64),
        np.asarray(episode_rewards, dtype=np.float32),
    )
    np.savez(
        os.path.join(seed_dir, "training.npz"),
        episode_lengths=stats.episode_lengths,
        episode_rewards=stats.episode_rewards,
        seed=seed,
        game=env_id,
    )
    return stats