import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################
# Network architectures
###################################################

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.1): #TODO changed from sigma0=0.5
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable means
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise scales
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma0 = sigma0

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        p = self.in_features

        bound = 1 / np.sqrt(p)

        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        self.weight_sigma.data.fill_(
            self.sigma0 / np.sqrt(p)
        )
        self.bias_sigma.data.fill_(
            self.sigma0 / np.sqrt(p)
        )

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)

        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = (self.weight_mu + self.weight_sigma * self.weight_epsilon)

            bias = (self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

def get_linear(noisy):
    return NoisyLinear if noisy else nn.Linear

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_shape, env_type):
        super().__init__()

        self.env_type = env_type
        channels, height, width = input_shape

        if env_type == "minatar":
            self.conv1 = nn.Conv2d(channels, 16, 3, stride=1)

            conv_h = height - 2
            conv_w = width - 2

            self.feature_dim = 16 * conv_h * conv_w

        elif env_type == "atari":
            self.conv1 = nn.Conv2d(channels, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

            self.feature_dim = 64 * 7 * 7

        else:
            raise ValueError(f"Unknown env_type: {env_type}")

    def extract_features(self, x):
        x = x.float()

        if self.env_type == "minatar":
            x = F.relu(self.conv1(x))

        elif self.env_type == "atari":
            x = x / 255.0
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

        return torch.flatten(x, start_dim=1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DQN(ConvFeatureExtractor):
    """Standard DQN: convolutional features followed by one Q-value head."""

    def __init__(self, input_shape, num_actions, env_type, noisy=False, hidden_layers=1):
        super().__init__(input_shape, env_type)

        hidden = 128 if env_type == "minatar" else 512
        Linear = get_linear(noisy)
        self.hidden_layers = hidden_layers

        if hidden_layers > 0:
            self.fc1 = Linear(self.feature_dim, hidden)
            self.extra_fc = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            output_in = hidden
        else:
            self.fc1 = nn.Identity()
            self.extra_fc = nn.ModuleList()
            output_in = self.feature_dim

        self.fc2 = Linear(output_in, num_actions)

    def forward(self, x):
        x = self.extract_features(x)

        if self.hidden_layers > 0:
            x = F.relu(self.fc1(x))
            for layer in self.extra_fc:
                x = F.relu(layer(x))

        return self.fc2(x)

class DuelDQN(ConvFeatureExtractor):
    """Dueling DQN with separate state-value and advantage streams."""

    def __init__(self, input_shape, num_actions, env_type, noisy=False, hidden_layers=1):
        super().__init__(input_shape, env_type)

        hidden = 128 if env_type == "minatar" else 512
        Linear = get_linear(noisy)
        self.hidden_layers = hidden_layers

        if hidden_layers > 0:
            self.value_fc = Linear(self.feature_dim, hidden)
            self.adv_fc = Linear(self.feature_dim, hidden)
            self.value_hidden = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            self.adv_hidden = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            output_in = hidden
        else:
            self.value_fc = nn.Identity()
            self.adv_fc = nn.Identity()
            self.value_hidden = nn.ModuleList()
            self.adv_hidden = nn.ModuleList()
            output_in = self.feature_dim

        self.value_out = Linear(output_in, 1)
        self.adv_out = Linear(output_in, num_actions)

    def forward(self, x):
        x = self.extract_features(x)

        if self.hidden_layers > 0:
            value = F.relu(self.value_fc(x))
            advantage = F.relu(self.adv_fc(x))
            for layer in self.value_hidden:
                value = F.relu(layer(value))
            for layer in self.adv_hidden:
                advantage = F.relu(layer(advantage))
        else:
            value = advantage = x

        value = self.value_out(value)
        advantage = self.adv_out(advantage)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DistributionalDQN(ConvFeatureExtractor):
    """C51 network: predicts a probability distribution over return atoms."""

    def __init__(self, input_shape, num_actions, env_type, num_atoms=51,
                 noisy=False, v_min=-10.0, v_max=10.0, hidden_layers=1):
        super().__init__(input_shape, env_type)

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.hidden_layers = hidden_layers

        if self.v_min >= self.v_max:
            raise ValueError("v_min must be smaller than v_max.")

        hidden = 128 if env_type == "minatar" else 512
        Linear = get_linear(noisy)

        # Fixed atom support z_i
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, num_atoms))

        if hidden_layers > 0:
            self.fc1 = Linear(self.feature_dim, hidden)
            self.extra_fc = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            output_in = hidden
        else:
            self.fc1 = nn.Identity()
            self.extra_fc = nn.ModuleList()
            output_in = self.feature_dim

        self.fc2 = Linear(output_in, self.num_actions * self.num_atoms)

    def dist(self, x):
        x = self.extract_features(x)

        if self.hidden_layers > 0:
            x = F.relu(self.fc1(x))
            for layer in self.extra_fc:
                x = F.relu(layer(x))

        logits = self.fc2(x)
        logits = logits.view(-1, self.num_actions, self.num_atoms)
        probs = F.softmax(logits, dim=2)
        return probs

    def forward(self, x):
        probs = self.dist(x)
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=2)
        return q_values

class DistributionalDuelDQN(ConvFeatureExtractor):
    """C51 + dueling architecture."""

    def __init__(self, input_shape, num_actions, env_type, num_atoms=51,
                 noisy=False, v_min=-10.0, v_max=10.0, hidden_layers=1):
        super().__init__(input_shape, env_type)

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.hidden_layers = hidden_layers

        if self.v_min >= self.v_max:
            raise ValueError("v_min must be smaller than v_max.")

        hidden = 128 if env_type == "minatar" else 512
        Linear = get_linear(noisy)

        # Fixed atom support z_i
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, num_atoms))

        if hidden_layers > 0:
            self.value_fc = Linear(self.feature_dim, hidden)
            self.adv_fc = Linear(self.feature_dim, hidden)
            self.value_hidden = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            self.adv_hidden = nn.ModuleList(
                Linear(hidden, hidden) for _ in range(hidden_layers - 1)
            )
            output_in = hidden
        else:
            self.value_fc = nn.Identity()
            self.adv_fc = nn.Identity()
            self.value_hidden = nn.ModuleList()
            self.adv_hidden = nn.ModuleList()
            output_in = self.feature_dim

        self.value_out = Linear(output_in, num_atoms)
        self.adv_out = Linear(output_in, num_actions * num_atoms)

    def dist(self, x):
        x = self.extract_features(x)

        if self.hidden_layers > 0:
            value = F.relu(self.value_fc(x))
            advantage = F.relu(self.adv_fc(x))
            for layer in self.value_hidden:
                value = F.relu(layer(value))
            for layer in self.adv_hidden:
                advantage = F.relu(layer(advantage))
        else:
            value = advantage = x

        value = self.value_out(value).view(-1, 1, self.num_atoms)
        advantage = self.adv_out(advantage).view(-1, self.num_actions, self.num_atoms)

        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        probs = F.softmax(logits, dim=2)
        return probs

    def forward(self, x):
        probs = self.dist(x)
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=2)
        return q_values

###################################################
# Network construction and initialization
###################################################

def init_weights(module, init_type="kaiming_uniform"):
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return

    if init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    
    if init_type == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight)

    else:
        raise ValueError(f"Unknown initialization: {init_type}")

    if module.bias is not None:
        nn.init.zeros_(module.bias)

def build_network(input_shape, num_actions, env_type, network_type="dqn",
                  noisy=False, num_atoms=51, v_min=-10.0, v_max=10.0,
                  hidden_layers=1, weight_init="kaiming_uniform"):

    nets = {
        "dqn": DQN,
        "dueling": DuelDQN,
        "distributional": DistributionalDQN,
        "distributional_dueling": DistributionalDuelDQN,
    }

    if network_type not in nets:
        raise ValueError(f"Unknown network_type: {network_type}")

    kwargs = dict(input_shape=input_shape, num_actions=num_actions,
                  env_type=env_type, noisy=noisy, hidden_layers=hidden_layers)

    if "distributional" in network_type:
        # C51 support is configured outside the network.
        kwargs["num_atoms"] = num_atoms
        kwargs["v_min"] = v_min
        kwargs["v_max"] = v_max

    model = nets[network_type](**kwargs)

    model.apply(lambda m: init_weights(m, init_type=weight_init))

    return model
