
from collections import namedtuple
import numpy as np

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

###################################################
# Replay buffer: uniform or prioritized
###################################################

class ReplayBuffer:
    def __init__(self, capacity, prioritized=False, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.eps = eps
        self.data, self.pos = [], 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def add(self, transition):
        # New PER samples receive the current maximum priority.
        priority = self.priorities[:len(self.data)].max() if self.data else 1.0
        priority = max(float(priority), self.eps)

        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=1.0):
        n = len(self.data)
        if self.prioritized:
            probs = self.priorities[:n] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(n, batch_size, p=probs)
            weights = (n * probs[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(n, batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)

        return [self.data[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        if self.prioritized:
            self.priorities[indices] = np.maximum(np.asarray(priorities), self.eps)


###################################################
# N-step transition construction
###################################################

def _make_n_step(buffer, n_step, gamma):
    """Collapse the oldest prefix into one n-step transition."""
    first = buffer[0]
    reward, next_state, done = 0.0, first.next_state, False

    for i, tr in enumerate(list(buffer)[:n_step]):
        reward += (gamma ** i) * tr.reward
        next_state, done = tr.next_state, tr.done
        if done:
            break

    return Transition(first.state, first.action, reward, next_state, done)


def add_n_step(buffer, replay, transition, n_step, gamma):
    """Add a raw transition and emit complete n-step transitions."""
    buffer.append(transition)

    if len(buffer) >= n_step:
        replay.add(_make_n_step(buffer, n_step, gamma))
        buffer.popleft()

    # Flush shortened terminal prefixes; done=True means no bootstrap term.
    if transition.done:
        while buffer:
            replay.add(_make_n_step(buffer, n_step, gamma))
            buffer.popleft()

###################################################
# PER priority signal
###################################################

def replay_priorities(estimator, states, actions, targets, eps):
    """TD error for scalar DQN; per-sample cross entropy for C51."""
    idx = np.arange(len(actions))

    if estimator.distributional:
        pred = estimator.predict_dist(states)[idx, actions]
        return -(targets * np.log(np.clip(pred, 1e-8, 1.0))).sum(axis=1) + eps

    pred = estimator.predict(states)[idx, actions]
    return np.abs(targets - pred) + eps
