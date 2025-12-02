import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from src.dataset import get_dataloaders, AA_TO_IDX, EOS_ID
from src.models.vae import SeqStructVAE
from src.models.diffusion import LatentDiffusion
from src.models.condition_encoder import ConditionEncoder  # 假设已存在

BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"


def train_epoch(diffusion_model, cond_encoder, vae, dataloader, optimizer, device):
    diffusion_model.train()
    cond_encoder.train()
    vae.eval()  # VAE 只做特征提取，不更新

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train Diffusion", leave=False):
        # 把 batch 张量搬到 device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        labels = batch["labels"]  # (B,11)

        # 使用已经训练好的 VAE 得到 latent z0
        with torch.no_grad():
            mu, logvar, _ = vae.encode(batch)
            z0 = vae.reparameterize(mu, logvar)  # (B, latent_dim)

        # 条件编码
        cond_emb = cond_encoder(labels)  # (B,cond_dim)

        optimizer.zero_grad()
        loss_dict = diffusion_model.p_losses(z0, cond_emb)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def eval_epoch(diffusion_model, cond_encoder, vae, dataloader, device):
    diffusion_model.eval()
    cond_encoder.eval()
    vae.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val Diffusion", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        labels = batch["labels"]

        with torch.no_grad():
            mu, logvar, _ = vae.encode(batch)
            z0 = vae.reparameterize(mu, logvar)

        cond_emb = cond_encoder(labels)

        loss_dict = diffusion_model.p_losses(z0, cond_emb)
        loss = loss_dict["loss"]

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "dataset.csv")
    pdb_dir = os.path.join(data_dir, "pdb")

    max_len = 64
    batch_size = 16
    num_epochs = 100
    lr = 1e-4

    train_loader, val_loader = get_dataloaders(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        max_len=max_len,
        batch_size=batch_size,
        val_ratio=0.1,
        pl_ddt_filter=0.0,
        num_workers=0,
        add_eos=True,
        eos_id=EOS_ID,
    )

    vocab_size = len(AA_TO_IDX) + 2  # 0 PAD + 20 AA + 1 EOS

    # === 加载训练好的 VAE ===
    vae = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        latent_dim=64,
        n_dec_layers=4,
        n_dec_heads=8,
        esm_model_name="esm2_t33_650M_UR50D",
        finetune_esm=False,
        pad_id=0,
        eos_id=EOS_ID,
        struct_hidden_dim=128,
    ).to(device)

    vae_ckpt_path = os.path.join(BASE_DIR, "checkpoints", "vae_best_esm_se3_eos.pt")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"Loaded VAE checkpoint from {vae_ckpt_path}")

    # === 初始化条件编码器 ===
    # 假设 ConditionEncoder(in_dim=11, hidden_dim=cond_dim)，输出 cond_dim 维
    cond_dim = 128
    cond_encoder = ConditionEncoder(in_dim=11, hidden_dim=cond_dim).to(device)

    # === 初始化 Diffusion 模型（使用你 diffusion.py 中的 LatentDiffusion） ===
    latent_dim = vae.latent_dim  # 64
    diffusion_model = LatentDiffusion(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        timesteps=1000,
    ).to(device)

    optimizer = Adam(
        list(diffusion_model.parameters()) + list(cond_encoder.parameters()),
        lr=lr,
    )

    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    diff_ckpt_path = os.path.join(save_dir, "diffusion_best_with_props_esm_se3_eos.pt")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_stats = train_epoch(diffusion_model, cond_encoder, vae, train_loader, optimizer, device)
        val_stats = eval_epoch(diffusion_model, cond_encoder, vae, val_loader, device)

        print(f"Train diffusion loss: {train_stats['loss']:.4f}")
        print(f"Val   diffusion loss: {val_stats['loss']:.4f}")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "model_state": diffusion_model.state_dict(),
                    "cond_encoder_state": cond_encoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                diff_ckpt_path,
            )
            print(f"Saved best diffusion checkpoint to {diff_ckpt_path}")


if __name__ == "__main__":
    main()