import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from networks import build_network

###################################################
# Estimator
###################################################

class Estimator:
    """Wrap the network, optimizer, prediction, and training update."""

    def __init__(self, input_shape, num_actions, env_type,
                 network_type="dqn", loss_func="mse", noisy=False, num_atoms=51,
                 v_min=-10.0, v_max=10.0, hidden_layers=1, weight_init="kaiming_uniform",
                 scope="estimator", summaries_dir=None, device=None,
                 adam_lr=0.00025, adam_eps=1e-6):

        self.scope = scope
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noisy = noisy
        self.distributional = "distributional" in network_type
        self.loss_func = loss_func.lower()

        self.model = build_network(
            input_shape, num_actions, env_type,
            network_type, noisy, num_atoms, v_min, v_max, hidden_layers, weight_init
        ).to(self.device)

        # Adam hyperparameters come from the experiment config.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=adam_lr,
            eps=adam_eps
        )

        self.global_step = 0
        self.summary_writer = None

        if summaries_dir:
            path = os.path.join(summaries_dir, f"summaries_{scope}")
            os.makedirs(path, exist_ok=True)
            self.summary_writer = SummaryWriter(path)

    def _to_tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.uint8, device=self.device)


    def _set_mode(self, use_noise):
        # Noise is only active while model.training == True
        if use_noise and self.noisy:
            self.model.train()
            self.model.reset_noise()
        else:
            self.model.eval()

    @torch.no_grad()
    def predict(self, states, use_noise=False):
        # Return expected Q-values
        old_mode = self.model.training
        self._set_mode(use_noise)

        q = self.model(self._to_tensor(states))

        self.model.train(old_mode)
        return q.cpu().numpy()


    @torch.no_grad()
    def predict_dist(self, states, use_noise=False):
        # Return p(z_i | s,a)
        if not self.distributional:
            raise RuntimeError("Network is not distributional.")

        old_mode = self.model.training
        self._set_mode(use_noise)

        probs = self.model.dist(self._to_tensor(states))

        self.model.train(old_mode)
        return probs.cpu().numpy()


    def update(self, states, actions, targets, weights=None):

        self.model.train()

        # Fresh noise for this gradient update
        if self.noisy:
            self.model.reset_noise()

        states = self._to_tensor(states)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        if self.distributional:
            # Cross entropy between projected target and p(z|s,a)
            probs = self.model.dist(states)
            idx = torch.arange(len(actions), device=self.device)
            action_probs = probs[idx, actions]

            losses = -(targets * torch.log(action_probs.clamp(min=1e-8))).sum(dim=1)

            # Only needed for logging
            q_values = (probs * self.model.support.view(1, 1, -1)).sum(dim=2)

        else:
            # Loss of TD error
            q_values = self.model(states)
            action_q = q_values.gather(1, actions[:, None]).squeeze(1)

            if self.loss_func == "mse":
                losses = (targets - action_q).pow(2)

            elif self.loss_func == "huber":
                losses = F.smooth_l1_loss(
                    action_q,
                    targets,
                    reduction="none",
                )
            else:
                raise ValueError(f"Unknown loss function: {self.loss_func}")

        # Prioritized replay importance weights
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            loss = (weights * losses).mean()
        else:
            loss = losses.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        # TensorBoard
        # if self.summary_writer and self.global_step % 1000 == 0:
        #     s = self.global_step
        #     self.summary_writer.add_scalar("train/loss", loss.item(), s)

        return loss.item(), losses.detach().cpu().numpy()


###################################################
# Standard / Double DQN targets
###################################################

@torch.no_grad()
def build_dqn_targets(online, target, next_states, rewards, dones,
                      gamma, double_dqn=True):

    x = online._to_tensor(next_states)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=online.device)
    d = torch.as_tensor(dones, dtype=torch.float32, device=online.device)

    # Fresh noise for target calculation
    online._set_mode(online.noisy)
    target._set_mode(target.noisy)

    if double_dqn:
        # Online chooses, target evaluates
        actions = online.model(x).argmax(dim=1)
        next_q = target.model(x).gather(1, actions[:, None]).squeeze(1)
    else:
        next_q = target.model(x).max(dim=1).values

    return (r + gamma * (1 - d) * next_q).cpu().numpy()

###################################################
# C51 projection
###################################################

def categorical_projection(next_dist, rewards, dones, gamma, support):

    device = next_dist.device
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device)[:, None]
    d = torch.as_tensor(dones, dtype=torch.float32, device=device)[:, None]

    v_min, v_max = support[0], support[-1]
    dz = (v_max - v_min) / (len(support) - 1)

    # Bellman transformed atoms
    tz = (r + gamma * (1 - d) * support[None, :]).clamp(v_min, v_max)

    # Position between support atoms
    b = (tz - v_min) / dz
    l, u = b.floor().long(), b.ceil().long()

    # Probability weights for lower/upper neighbours
    wl, wu = u.float() - b, b - l.float()

    # Exact atom -> all mass remains there
    same = l == u
    wl = torch.where(same, torch.ones_like(wl), wl)
    wu = torch.where(same, torch.zeros_like(wu), wu)

    projected = torch.zeros_like(next_dist)
    projected.scatter_add_(1, l, next_dist * wl)
    projected.scatter_add_(1, u, next_dist * wu)

    return projected


###################################################
# Distributional / C51 targets
###################################################

@torch.no_grad()
def build_c51_targets(online, target, next_states, rewards, dones,
                      gamma, double_dqn=True):

    if not online.distributional or not target.distributional:
        raise ValueError("C51 requires distributional networks.")

    x = online._to_tensor(next_states)

    # Fresh NoisyNet samples
    online._set_mode(online.noisy)
    target._set_mode(target.noisy)

    # Target-network distribution p(z | s',a)
    target_probs = target.model.dist(x)

    if double_dqn:
        # Online chooses next action
        actions = online.model(x).argmax(dim=1)
    else:
        # Target chooses from expected Q-values
        actions = target.model(x).argmax(dim=1)

    # Distribution corresponding to selected action
    idx = torch.arange(len(x), device=online.device)
    next_dist = target_probs[idx, actions]

    projected = categorical_projection(
        next_dist, rewards, dones, gamma,
        target.model.support
    )

    return projected.cpu().numpy()

###################################################
# Target network copy
###################################################

class ModelParametersCopier:
    """Copy online-network parameters to the target network."""

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def make(self):
        self.target.model.load_state_dict(self.source.model.state_dict())


###################################################
# Epsilon-greedy policy
###################################################

def make_epsilon_greedy_policy(estimator, num_actions,
                               use_noisy_exploration=False):

    def policy_fn(observation, epsilon):

        # Random exploration: skip neural-network inference
        if np.random.random() < epsilon:
            return int(np.random.randint(num_actions))

        # Greedy action selection (with NoisyNet if enabled)
        q = estimator.predict(
            np.expand_dims(observation, 0),
            use_noise=use_noisy_exploration
        )[0]

        return int(np.argmax(q))

    return policy_fn