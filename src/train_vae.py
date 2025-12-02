import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.dataset import AA_TO_IDX, EOS_ID
from src.models.vae import SeqStructVAE

# ====== 项目根目录（根据你的实际路径确认）======
BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"
# =================================================


def train_epoch(model, dataloader, optimizer, device, kl_weight: float = 1e-3):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train VAE", leave=False):
        # 把 batch 里的张量搬到 device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        # forward: 返回 {"z","mu","logvar","logits",...}
        outputs = model(batch)

        # 使用 SeqStructVAE.loss_function 计算 loss
        loss_dict = SeqStructVAE.loss_function(
            outputs,      # 第一个参数：forward 的输出 dict
            batch,        # 第二个参数：原始 batch
            kl_weight=kl_weight,
        )
        loss = loss_dict["loss"]
        recon = loss_dict["recon_loss"]
        kl = loss_dict["kl_loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def eval_epoch(model, dataloader, device, kl_weight: float = 1e-3):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val VAE", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        outputs = model(batch)
        loss_dict = SeqStructVAE.loss_function(
            outputs,
            batch,
            kl_weight=kl_weight,
        )
        loss = loss_dict["loss"]
        recon = loss_dict["recon_loss"]
        kl = loss_dict["kl_loss"]

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === 数据路径 ===
    data_dir = os.path.join(BASE_DIR, "data")
    csv_path = os.path.join(data_dir, "dataset.csv")
    pdb_dir = os.path.join(data_dir, "pdb")

    max_len = 64
    batch_size = 16        # 用 ESM2 650M 时建议先用小 batch
    num_epochs = 20
    lr = 1e-4
    kl_weight = 1e-3

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

    # === 初始化带 ESM2 + SE3 encoder 的 VAE ===
    model = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,                         # 稍大一点
        latent_dim=64,
        esm_model_name="esm2_t33_650M_UR50D",
        finetune_esm=False,
        pad_id=0,
        eos_id=EOS_ID,
        # 先冻结 ESM2
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "vae_best_esmsg.pt")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_stats = train_epoch(model, train_loader, optimizer, device, kl_weight)
        val_stats = eval_epoch(model, val_loader, device, kl_weight)

        print(
            f"Train loss: {train_stats['loss']:.4f}, "
            f"recon: {train_stats['recon_loss']:.4f}, "
            f"kl: {train_stats['kl_loss']:.4f}"
        )
        print(
            f"Val   loss: {val_stats['loss']:.4f}, "
            f"recon: {val_stats['recon_loss']:.4f}, "
            f"kl: {val_stats['kl_loss']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"Saved best VAE checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()