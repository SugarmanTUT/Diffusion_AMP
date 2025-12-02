import os
from typing import List

import numpy as np
import torch

from src.dataset import AA_TO_IDX, IDX_TO_AA
from src.models.vae import SeqStructVAE
from src.models.diffusion import LatentDiffusion
from src.models.condition_encoder import ConditionEncoder

# ====== 项目根目录 ======
BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"
# =======================

POS_AA = set(["K", "R", "H"])
NEG_AA = set(["D", "E"])
HYDRO_AA = set(["A", "V", "I", "L", "M", "F", "W", "Y"])


def tokens_to_seq(tokens: torch.Tensor) -> str:
    """Convert a 1D token tensor to amino acid string, ignoring PAD(0)."""
    seq = []
    for idx in tokens.tolist():
        if idx == 0:
            continue
        aa = IDX_TO_AA.get(idx, "X")
        seq.append(aa)
    return "".join(seq)


def compute_basic_props(seq: str):
    """与 eval_generated.py 一致的长度/电荷/比例计算。"""
    seq = seq.strip().upper()
    L = len(seq)
    if L == 0:
        return L, 0.0, 0.0, 0.0

    net_charge = 0
    n_pos = 0
    n_neg = 0
    n_hydro = 0
    for aa in seq:
        if aa in POS_AA:
            net_charge += 1
            n_pos += 1
        elif aa in NEG_AA:
            net_charge -= 1
            n_neg += 1
        if aa in HYDRO_AA:
            n_hydro += 1

    frac_pos = n_pos / L
    frac_hydro = n_hydro / L
    return L, float(net_charge), float(frac_pos), float(frac_hydro)


def rough_amp_style_score(L, net_charge, frac_pos, frac_hydro) -> float:
    """粗略 AMP 风格分数，用于打印参考。"""
    def gauss_score(x, mu, sigma):
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    s_len = gauss_score(L, mu=30.0, sigma=10.0)
    s_charge = gauss_score(net_charge, mu=8.0, sigma=4.0)
    s_pos = gauss_score(frac_pos, mu=0.25, sigma=0.10)
    s_hydro = gauss_score(frac_hydro, mu=0.35, sigma=0.15)
    return (s_len + s_charge + s_pos + s_hydro) / 4.0


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 64
    latent_dim = 64
    cond_dim = 128
    timesteps = 1000
    batch_size = 16  # 一次生成的样本数

    vocab_size = len(AA_TO_IDX) + 1

    # === 加载 VAE ===
    vae = SeqStructVAE(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=128,
        latent_dim=latent_dim,
    ).to(device)
    vae_ckpt_path = os.path.join(BASE_DIR, "checkpoints", "vae_best.pt")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # === 加载 diffusion + condition encoder ===
    diffusion = LatentDiffusion(latent_dim=latent_dim, cond_dim=cond_dim, timesteps=timesteps).to(
        device
    )
    cond_encoder = ConditionEncoder(in_dim=11, hidden_dim=64, out_dim=cond_dim).to(device)

    diff_ckpt_path = os.path.join(BASE_DIR, "checkpoints", "diffusion_best_with_props.pt")
    assert os.path.exists(
        diff_ckpt_path
    ), "Train diffusion with properties first and save diffusion_best_with_props.pt"
    diff_ckpt = torch.load(diff_ckpt_path, map_location=device)
    diffusion.load_state_dict(diff_ckpt["diffusion_state"])
    cond_encoder.load_state_dict(diff_ckpt["cond_encoder_state"])
    diffusion.eval()
    cond_encoder.eval()

    # === 条件向量：11 维 ===
    # [Label, ABP, AFP, AGnP, AGgP, AVP, pLDDT_norm,
    #  length_norm, net_charge_norm, frac_pos, frac_hydro]
    target_length = 30.0        # 目标物理长度（大概）
    target_net_charge = 8.0     # 目标净电荷
    target_frac_pos = 0.25      # 正电比例
    target_frac_hydro = 0.35    # 疏水比例

    base_cond = torch.tensor(
        [
            1.0,                         # Label
            1.0,                         # ABP
            0.0,                         # AFP
            1.0,                         # AGnP
            0.0,                         # AGgP
            0.0,                         # AVP
            0.8,                         # pLDDT_norm
            target_length / max_len,     # length_norm
            max(min(target_net_charge / 20.0, 1.0), -1.0),  # net_charge_norm
            target_frac_pos,
            target_frac_hydro,
        ],
        dtype=torch.float32,
        device=device,
    )

    cond_vec = base_cond.unsqueeze(0).repeat(batch_size, 1)  # (B, 11)
    cond_emb = cond_encoder(cond_vec)  # (B, cond_dim)

    # === 采样 latent ===
    z0 = diffusion.sample(batch_size=batch_size, cond=cond_emb, device=device)

    # === 解码为序列 ===
    tokens = vae.generate(z0)  # (B, L)

    # Convert to AA strings & 打印属性
    seqs: List[str] = []
    out_lines: List[str] = []
    for i in range(batch_size):
        seq = tokens_to_seq(tokens[i])
        L, q, fp, fh = compute_basic_props(seq)
        score = rough_amp_style_score(L, q, fp, fh)
        seqs.append(seq)
        info = (
            f"gen_{i}: len={L}, net_charge={q:.1f}, "
            f"frac_pos={fp:.2f}, frac_hydro={fh:.2f}, score={score:.3f}"
        )
        print(info)
        out_lines.append(f">{info}\n{seq}\n")

    out_path = os.path.join(BASE_DIR, "generated_sequences_with_props.txt")
    with open(out_path, "w") as f:
        for line in out_lines:
            f.write(line)

    print(f"Saved generated sequences to {out_path}")


if __name__ == "__main__":
    main()