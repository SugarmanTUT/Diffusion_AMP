import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import get_dataloaders, AA_TO_IDX, IDX_TO_AA
from src.models.vae import SeqStructVAE

# ====== 项目根目录 ======
BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"
# =======================


class SimpleLabelPredictor(nn.Module):
    """Simple MLP on VAE latent to predict labels (for in-silico property check)."""

    def __init__(self, latent_dim: int = 64, out_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x


@torch.no_grad()
def evaluate_reconstruction():
    """Evaluate VAE reconstruction on validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 绝对路径 ===
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "dataset.csv")
    pdb_dir = os.path.join(data_dir, "pdb")
    # =================

    max_len = 64
    batch_size = 32

    train_loader, val_loader = get_dataloaders(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        max_len=max_len,
        batch_size=batch_size,
        val_ratio=0.1,
        pl_ddt_filter=0.0,
        num_workers=0,
    )

    vocab_size = len(AA_TO_IDX) + 1
    vae = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=128,
        latent_dim=64,
    ).to(device)

    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "vae_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    total_edit_distance = 0.0
    total_tokens = 0

    for batch in val_loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        out = vae(batch)
        z = out["z"]
        tokens_hat = vae.generate(z)  # (B, L)
        tokens_true = batch["seq_tokens"]  # (B, L)

        B, L = tokens_true.shape
        for i in range(B):
            # Convert tokens to strings and compute Levenshtein distance (approx via simple mismatches)
            true_seq = "".join(
                IDX_TO_AA.get(idx.item(), "X")
                for idx in tokens_true[i]
                if idx.item() != 0
            )
            pred_seq = "".join(
                IDX_TO_AA.get(idx.item(), "X")
                for idx in tokens_hat[i]
                if idx.item() != 0
            )

            # Very rough: count mismatched positions after aligning to min len
            m = min(len(true_seq), len(pred_seq))
            mismatches = sum(1 for a, b in zip(true_seq[:m], pred_seq[:m]) if a != b)
            mismatches += abs(len(true_seq) - len(pred_seq))
            total_edit_distance += mismatches
            total_tokens += max(len(true_seq), len(pred_seq), 1)

    avg_edit = total_edit_distance / total_tokens
    print(f"Approximate per-position edit rate: {avg_edit:.4f}")


if __name__ == "__main__":
    evaluate_reconstruction()