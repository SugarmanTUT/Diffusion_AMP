import torch
import torch.nn as nn
import torch.nn.functional as F


class SE3StructEncoder(nn.Module):
    """
    简化版的 SE(3)-invariant 结构编码器（EGNN 风格）：
    - 使用坐标差的范数（距离）和接触图作为边特征；
    - 更新的是 per-node scalar features（结构 embedding），
      坐标仅用于计算距离，不在本实现中更新，保持 SE(3)-invariance。
    """

    def __init__(
        self,
        in_dim: int = 0,      # 如果你有结构端初始 scalar 特征，可以设为 >0
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 如果没有初始特征，就从 0 constant 开始
        if in_dim > 0:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = None

        # message 网络：基于 (h_i, h_j, ||x_i - x_j||, contact_ij) -> m_ij
        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim + 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
            self.node_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

    def forward(self, coords: torch.Tensor, contacts: torch.Tensor, node_feats: torch.Tensor = None):
        """
        coords:   (B, L, 3)
        contacts: (B, L, L)  — 0/1 或 [0,1] 权重
        node_feats: (B, L, in_dim) or None

        返回:
            h_struct: (B, L, hidden_dim)
        """
        B, L, _ = coords.shape
        device = coords.device

        if node_feats is None:
            h = torch.zeros((B, L, self.hidden_dim), device=device)
        else:
            h0 = self.input_proj(node_feats) if self.input_proj is not None else node_feats
            h = h0

        # 构造 pairwise 距离 (B, L, L, 1)
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, L, L, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)         # (B, L, L, 1)

        # edge mask：根据 contacts 决定哪些边有效
        # 这里直接使用 contacts>0 作为 mask
        edge_mask = (contacts > 0).unsqueeze(-1)  # (B, L, L, 1)

        for layer in range(self.num_layers):
            edge_mlp = self.edge_mlps[layer]
            node_mlp = self.node_mlps[layer]

            # 构造边特征：concat(h_i, h_j, dist_ij, contact_ij)
            hi = h[:, :, None, :].expand(B, L, L, self.hidden_dim)
            hj = h[:, None, :, :].expand(B, L, L, self.hidden_dim)
            contact_ij = contacts.unsqueeze(-1)  # (B, L, L, 1)
            edge_input = torch.cat([hi, hj, dist, contact_ij], dim=-1)  # (B, L, L, 2H+2)

            m_ij = edge_mlp(edge_input)  # (B, L, L, H)
            # 只保留有效边的消息
            m_ij = m_ij * edge_mask

            # 聚合邻居消息：sum_j m_ij
            m_i = m_ij.sum(dim=2)  # (B, L, H)

            # 更新节点特征
            node_input = torch.cat([h, m_i], dim=-1)  # (B, L, 2H)
            delta_h = node_mlp(node_input)
            h = h + delta_h  # 残差连接

        return h  # (B, L, hidden_dim)