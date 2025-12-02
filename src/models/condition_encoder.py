import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """Encode multi-label + scalar properties into a condition embedding."""

    # in_dim 改成 11：7 个原始标签 + 4 个物化属性
    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, in_dim)
           e.g. [Label, ABP, AFP, AGnP, AGgP, AVP, pLDDT_norm,
                 length_norm, net_charge_norm, frac_pos, frac_hydro]
        Returns: (B, out_dim)
        """
        x = F.relu(self.fc1(y))
        x = self.fc2(x)
        return x