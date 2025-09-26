import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Config Inicial
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Creamos el entorno
# Usamos el entorno CartPole-v1 de Gymnasium
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=seed)

n_obs = env.observation_space.shape[0]
n_act = env.action_space.n
print(f"Dimensión de estado: {n_obs} | Acciones: {n_act}")

# Definimos las redes
class DQN(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], n_act),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

policy_net = DQN(n_obs, n_act).to(device)
target_net = DQN(n_obs, n_act).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Creamos los hiperparametros (ajustables)
@dataclass
class DQNConfig:
    # Entrenamiento
    episodes: int = 500            # Si no se usa Cuda, hay que ver cuantos episodios se pueden correr sin que muera
    max_steps_per_ep: int = 1000   # Tope de pasos por episodio
    gamma: float = 0.99            # factor de descuento

    # Replay buffer
    replay_capacity: int = 50_000
    batch_size: int = 64

    # Exploración ε-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 25_000 

    # Optimización
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    # Target network
    target_update_freq: int = 1000  # cada N pasos de entrenamiento
    tau: float = 1.0  # 1.0 = hard update; <1.0 = soft update

    # Render/Log
    render_train: bool = False
    print_every: int = 10

cfg = DQNConfig()

optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = nn.SmoothL1Loss()

print("Hiperparametros usados:")
for k, v in vars(cfg).items():
    print(f"  {k}: {v}")

