import os

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.dataset import get_dataloaders, AA_TO_IDX
from src.models.vae import SeqStructVAE
from src.models.diffusion import LatentDiffusion
from src.models.condition_encoder import ConditionEncoder

# ====== 绝对路径项目根目录 ======
BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"
# ===============================


def train_diffusion_epoch(
    vae: SeqStructVAE,
    diffusion: LatentDiffusion,
    cond_encoder: ConditionEncoder,
    dataloader,
    optimizer,
    device,
):
    vae.eval()  # VAE encoder frozen during diffusion training
    diffusion.train()
    cond_encoder.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train Diffusion", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            out = vae(batch)
            z0 = out["z"]  # (B, latent_dim)

        labels = batch["labels"]  # (B, 11)
        cond = cond_encoder(labels)  # (B, cond_dim)

        loss_dict = diffusion.p_losses(z0, cond)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def eval_diffusion_epoch(
    vae: SeqStructVAE,
    diffusion: LatentDiffusion,
    cond_encoder: ConditionEncoder,
    dataloader,
    device,
):
    vae.eval()
    diffusion.eval()
    cond_encoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val Diffusion", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            out = vae(batch)
            z0 = out["z"]

        labels = batch["labels"]  # (B, 11)
        cond = cond_encoder(labels)
        loss_dict = diffusion.p_losses(z0, cond)
        loss = loss_dict["loss"]

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 改成绝对路径 ===
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "dataset.csv")
    pdb_dir = os.path.join(data_dir, "pdb")
    # ====================

    max_len = 64
    batch_size = 32
    num_epochs = 50
    lr = 1e-4
    timesteps = 1000

    train_loader, val_loader = get_dataloaders(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        max_len=max_len,
        batch_size=batch_size,
        val_ratio=0.1,
        pl_ddt_filter=0.0,
        num_workers=0,
    )

    vocab_size = len(AA_TO_IDX) + 2
    vae = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=128,
        latent_dim=64,
    ).to(device)

    # === VAE checkpoint 绝对路径 ===
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "vae_best.pt")
    assert os.path.exists(ckpt_path), "Train VAE first and save vae_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    # ===============================
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    diffusion = LatentDiffusion(latent_dim=64, cond_dim=128, timesteps=timesteps).to(device)
    # in_dim 改为 11
    cond_encoder = ConditionEncoder(in_dim=11, hidden_dim=64, out_dim=128).to(device)

    params = list(diffusion.parameters()) + list(cond_encoder.parameters())
    optimizer = Adam(params, lr=lr)

    # === checkpoints 绝对路径 ===
    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    # ============================

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs} (Diffusion)")

        train_stats = train_diffusion_epoch(
            vae, diffusion, cond_encoder, train_loader, optimizer, device
        )
        val_stats = eval_diffusion_epoch(
            vae, diffusion, cond_encoder, val_loader, device
        )

        print(f"Train diff loss: {train_stats['loss']:.6f}")
        print(f"Val   diff loss: {val_stats['loss']:.6f}")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            ckpt_path = os.path.join(save_dir, "diffusion_best_with_props.pt")
            torch.save(
                {
                    "diffusion_state": diffusion.state_dict(),
                    "cond_encoder_state": cond_encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"Saved best diffusion checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()