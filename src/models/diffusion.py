import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal embedding followed by MLP.
        t: (B,) or scalar tensor, time step in [0, T-1]
        Returns: (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        emb = self.lin1(emb)
        emb = F.relu(emb)
        emb = self.lin2(emb)
        return emb


class DiffusionMLP(nn.Module):
    """Simple MLP-based noise predictor for latent diffusion."""

    def __init__(self, latent_dim: int, cond_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.time_emb = TimeEmbedding(latent_dim)
        self.fc1 = nn.Linear(latent_dim + latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t: (B, latent_dim)
        t:   (B,)
        cond:(B, cond_dim)
        """
        t_emb = self.time_emb(t)  # (B, latent_dim)
        x = torch.cat([z_t, t_emb, cond], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LatentDiffusion(nn.Module):
    """DDPM-style latent diffusion."""

    def __init__(
        self,
        latent_dim: int = 64,
        cond_dim: int = 128,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.T = timesteps

        # Linear noise schedule
        beta_start = 1e-4
        beta_end = 2e-2
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.eps_model = DiffusionMLP(latent_dim, cond_dim)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        z_t = sqrt(a_bar_t) * z0 + sqrt(1 - a_bar_t) * eps
        """
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)        # (B,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (B,1)
        return sqrt_ab * z0 + sqrt_om * noise

    def p_losses(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training loss for one batch:
        - Sample t
        - Add noise
        - Predict noise
        """
        B = z0.size(0)
        device = z0.device
        t = torch.randint(0, self.T, (B,), device=device).long()
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        noise_pred = self.eps_model(z_t, t, cond)
        loss = F.mse_loss(noise_pred, noise)
        return {
            "loss": loss,
        }

    @torch.no_grad()
    def p_sample(self, z_t: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step.
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))

        # Predict noise
        B = z_t.size(0)
        t_tensor = torch.full((B,), t, device=z_t.device, dtype=torch.long)
        eps_theta = self.eps_model(z_t, t_tensor, cond)

        # From epsilon to z0
        z0_hat = (z_t - eps_theta * sqrt_one_minus_alphas_cumprod_t) / torch.sqrt(
            self.alphas_cumprod[t]
        )

        if t > 0:
            noise = torch.randn_like(z_t)
        else:
            noise = torch.zeros_like(z_t)

        coef1 = torch.sqrt(self.alphas[t]) * (1.0 - self.alphas_cumprod[t - 1]) / (
            1.0 - self.alphas_cumprod[t]
        ) if t > 0 else torch.tensor(1.0, device=z_t.device)
        coef2 = torch.sqrt(self.alphas_cumprod[t - 1]) * (1.0 - self.alphas[t]) / (
            1.0 - self.alphas_cumprod[t]
        ) if t > 0 else torch.tensor(0.0, device=z_t.device)

        # This is a slightly simplified form; for research it's acceptable.
        mean = sqrt_recip_alphas_t * (z_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)
        var = betas_t
        std = torch.sqrt(var) if t > 0 else torch.tensor(0.0, device=z_t.device)
        z_prev = mean + std * noise
        return z_prev

    @torch.no_grad()
    def sample(self, batch_size: int, cond: torch.Tensor, device=None) -> torch.Tensor:
        """
        Run full reverse diffusion from pure noise.
        cond: (B, cond_dim)
        Returns: z0 (B, latent_dim)
        """
        if device is None:
            device = cond.device
        z_t = torch.randn(batch_size, self.latent_dim, device=device)
        for t in reversed(range(self.T)):
            z_t = self.p_sample(z_t, t, cond)
        return z_t