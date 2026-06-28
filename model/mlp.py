from typing import Any

import torch
import torch.nn as nn

class CostWeightMLP(nn.Module): 
    """
    weights from driving context.
    Input:  [v_norm, d_norm, a_recent] — shape (3,) or (B, 3)
    Output: [w_jerk, w_speed, w_aggression] — shape (3,) or (B, 3), values in [1e-3, 100]

    Batch of B timestamps
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # (3 → 32): project context into hidden space
            nn.Tanh(),                          # bounds activations to (-1, 1); smooth gradients
            nn.Linear(hidden_dim, hidden_dim),  # (32 → 32): learn non-linear weight relationships -> 32 x 32 = 1024
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),  # (32 → 3): produce one raw score per cost weight
            nn.ReLU(),                          # enforce non-negativity, cost weights must be ≥ 0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(1e-3, 100.0)
