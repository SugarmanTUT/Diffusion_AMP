import os
import torch

from src.dataset import IDX_TO_AA, PAD_ID, EOS_ID, AA_TO_IDX
from src.models.vae import SeqStructVAE
from src.models.diffusion import LatentDiffusion
from src.models.condition_encoder import ConditionEncoder  # 你的条件编码器

BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"


def tokens_to_seq(tokens: torch.Tensor) -> str:
    """
    将 (L,) 的 token 序列转换为氨基酸字符串：
    - 跳过 PAD
    - 遇到 EOS 截断
    """
    seq = []
    for idx in tokens.tolist():
        if idx == PAD_ID:
            continue
        if idx == EOS_ID:
            break
        aa = IDX_TO_AA.get(idx, "X")  # 只对 1..20 有效，EOS 不在 AA_TO_IDX 中
        seq.append(aa)
    return "".join(seq)


def build_condition_batch(batch_size: int, device: torch.device):
    """
    构造一个简单的条件批次:
    labels 向量: (B,11)
    这里只给出一个示例配置，你可以根据需要修改：
    - Label=1, ABP=1, 其余功能标签 0
    - pLDDT_norm=0.8
    - length_norm=0.5, net_charge_norm=0.5
    - frac_pos=0.2, frac_hydro=0.3
    """
    B = batch_size
    labels = torch.zeros(B, 11, device=device)

    # 多标签功能: Label, ABP, AFP, AGnP, AGgP, AVP
    labels[:, 0] = 1.0  # Label
    labels[:, 1] = 1.0  # ABP

    # 结构置信度
    labels[:, 6] = 0.8  # pLDDT_norm

    # 长度/电荷/比例
    labels[:, 7] = 0.5      # length_norm
    labels[:, 8] = 0.5      # net_charge_norm
    labels[:, 9] = 0.2      # frac_pos
    labels[:, 10] = 0.3     # frac_hydro

    return labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    max_len = 64
    vocab_size = len(AA_TO_IDX) + 2  # 0 PAD + 20 AA + 1 EOS

    # === 加载 VAE ===
    vae = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        latent_dim=64,
        n_dec_layers=4,
        n_dec_heads=8,
        esm_model_name="esm2_t33_650M_UR50D",
        finetune_esm=False,
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        struct_hidden_dim=128,
    ).to(device)

    vae_ckpt_path = os.path.join(BASE_DIR, "checkpoints", "vae_best_esm_se3_eos.pt")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"Loaded VAE from {vae_ckpt_path}")

    # === 加载 Diffusion + 条件编码器 ===
    cond_dim = 128
    cond_encoder = ConditionEncoder(in_dim=11, hidden_dim=cond_dim).to(device)

    diffusion_model = LatentDiffusion(
        latent_dim=vae.latent_dim,
        cond_dim=cond_dim,
        timesteps=1000,
    ).to(device)

    diff_ckpt_path = os.path.join(BASE_DIR, "checkpoints", "diffusion_best_with_props_esm_se3_eos.pt")
    diff_ckpt = torch.load(diff_ckpt_path, map_location=device)
    diffusion_model.load_state_dict(diff_ckpt["model_state"])
    cond_encoder.load_state_dict(diff_ckpt["cond_encoder_state"])
    diffusion_model.eval()
    cond_encoder.eval()
    print(f"Loaded diffusion model from {diff_ckpt_path}")

    # === 构造条件，并从扩散模型采样 latent z ===
    B = 16
    labels = build_condition_batch(B, device=device)  # (B,11)
    with torch.no_grad():
        cond_emb = cond_encoder(labels)               # (B,cond_dim)
        z_samples = diffusion_model.sample(
            batch_size=B,
            cond=cond_emb,
            device=device,
        )  # (B,latent_dim)

        tokens = vae.generate(z_samples)  # (B,L)

    # 转成字符串并打印
    seqs = [tokens_to_seq(tokens[i]) for i in range(B)]

    for i, s in enumerate(seqs):
        print(f"[{i}] {s}")


if __name__ == "__main__":
    main()