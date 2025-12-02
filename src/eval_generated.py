import os
from typing import List, Dict

import numpy as np

# 根据你项目的根目录修改
BASE_DIR = "/home/zhangmingjian/Diffusion_AMP"

GENERATED_PATH = os.path.join(BASE_DIR, "generated_sequences.txt")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# 简单的氨基酸电荷映射（pH ~7.4 下大致情况）
POS_AA = set(["K", "R", "H"])  # 正电
NEG_AA = set(["D", "E"])       # 负电


def read_fasta_like(path: str) -> Dict[str, str]:
    """读取类似 FASTA 的文件，返回 {id: seq} 字典。"""
    seqs = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r") as f:
        current_id = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush
                if current_id is not None and buf:
                    seqs[current_id] = "".join(buf)
                    buf = []
                current_id = line[1:]
            else:
                buf.append(line)
        # last
        if current_id is not None and buf:
            seqs[current_id] = "".join(buf)
    return seqs


def compute_basic_props(seq: str) -> Dict[str, float]:
    """计算长度、净电荷、疏水性等简单性质。"""
    seq = seq.strip().upper()
    length = len(seq)
    if length == 0:
        return {
            "length": 0,
            "net_charge": 0.0,
            "frac_pos": 0.0,
            "frac_hydro": 0.0,
        }

    # 净电荷（非常简化）：+1 对 K,R,H； -1 对 D,E
    net_charge = 0
    n_pos = 0
    n_neg = 0
    n_hydro = 0

    # 一个简单的疏水氨基酸集合
    HYDRO_AA = set(["A", "V", "I", "L", "M", "F", "W", "Y"])

    for aa in seq:
        if aa in POS_AA:
            net_charge += 1
            n_pos += 1
        elif aa in NEG_AA:
            net_charge -= 1
            n_neg += 1
        if aa in HYDRO_AA:
            n_hydro += 1

    frac_pos = n_pos / length
    frac_hydro = n_hydro / length

    return {
        "length": length,
        "net_charge": float(net_charge),
        "frac_pos": float(frac_pos),
        "frac_hydro": float(frac_hydro),
    }


def print_overall_stats(props_list: List[Dict[str, float]]):
    lengths = np.array([p["length"] for p in props_list], dtype=np.float32)
    charges = np.array([p["net_charge"] for p in props_list], dtype=np.float32)
    frac_pos = np.array([p["frac_pos"] for p in props_list], dtype=np.float32)
    frac_hydro = np.array([p["frac_hydro"] for p in props_list], dtype=np.float32)

    def describe(x):
        return {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }

    print("=== Overall statistics of generated sequences ===")
    print("Length:", describe(lengths))
    print("Net charge:", describe(charges))
    print("Fraction of positive AAs (K/R/H):", describe(frac_pos))
    print("Fraction of hydrophobic AAs (A/V/I/L/M/F/W/Y):", describe(frac_hydro))
    print("===============================================")


def rough_amp_style_score(p: Dict[str, float]) -> float:
    """
    给每条序列一个非常粗糙的“AMP 风格分数”:
    - 偏好中短链 (10-50)
    - 偏好净正电荷在 [2, 15] 内
    - 偏好正电氨基酸比例在 [0.1, 0.4]
    - 偏好疏水比例在 [0.2, 0.6]
    这里只做归一化高斯型评分，方便比较，不是严格物理意义。
    """
    length = p["length"]
    net_charge = p["net_charge"]
    frac_pos = p["frac_pos"]
    frac_hydro = p["frac_hydro"]

    # 简单的高斯形得分函数
    def gauss_score(x, mu, sigma):
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    # 长度目标 ~30 aa, 容忍度 10
    s_len = gauss_score(length, mu=30.0, sigma=10.0)

    # 电荷目标 ~+8, 容忍度 4
    s_charge = gauss_score(net_charge, mu=8.0, sigma=4.0)

    # 正电比例目标 0.25, 容忍度 0.1
    s_pos = gauss_score(frac_pos, mu=0.25, sigma=0.10)

    # 疏水比例目标 0.35, 容忍度 0.15
    s_hydro = gauss_score(frac_hydro, mu=0.35, sigma=0.15)

    # 综合（可以改权重）
    score = (s_len + s_charge + s_pos + s_hydro) / 4.0
    return float(score)


def main():
    seqs = read_fasta_like(GENERATED_PATH)
    print(f"Loaded {len(seqs)} generated sequences from {GENERATED_PATH}")

    props_list = []
    for sid, s in seqs.items():
        props = compute_basic_props(s)
        props_list.append(props)

    if not props_list:
        print("No sequences found, please check generated_sequences.txt")
        return

    # 打印总体统计
    print_overall_stats(props_list)

    # 逐条打印简单的属性和粗略 AMP 风格分数
    print("\n=== Per-sequence quick summary ===")
    for (sid, s), props in zip(seqs.items(), props_list):
        score = rough_amp_style_score(props)
        print(
            f"{sid}: len={props['length']}, "
            f"net_charge={props['net_charge']:.1f}, "
            f"frac_pos={props['frac_pos']:.2f}, "
            f"frac_hydro={props['frac_hydro']:.2f}, "
            f"amp_style_score={score:.3f}"
        )


if __name__ == "__main__":
    main()