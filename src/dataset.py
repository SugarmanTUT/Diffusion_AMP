import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1..20, 0 reserved for PAD
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}

PAD_ID = 0
EOS_ID = 21  # EOS token id（不在 AA_TO_IDX 中）

# 氨基酸性质（用于简单属性计算）
POS_AA = set(["K", "R", "H"])
NEG_AA = set(["D", "E"])
HYDRO_AA = set(["A", "V", "I", "L", "M", "F", "W", "Y"])

# ESM 支持的氨基酸/模糊符号
VALID_AA_FOR_ESM = set("ACDEFGHIKLMNPQRSTVWYXBZOU")


def clean_sequence_for_esm(seq: str) -> str:
    """
    把序列清洗成仅包含 ESM 支持的氨基酸字符:
    - 去掉空白字符（空格、tab、换行）
    - 非法字符统一映射为 'X'
    - 全部转换为大写
    """
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq.strip()
    # 去掉所有空白字符
    seq = "".join(ch for ch in seq if not ch.isspace())

    cleaned = []
    for ch in seq:
        ch_u = ch.upper()
        if ch_u in VALID_AA_FOR_ESM:
            cleaned.append(ch_u)
        else:
            cleaned.append("X")
    return "".join(cleaned)


def compute_simple_props_from_seq(seq: str):
    """
    从序列计算:
    - length
    - net_charge
    - frac_pos: 正电氨基酸比例 (K/R/H)
    - frac_hydro: 疏水氨基酸比例 (A/V/I/L/M/F/W/Y)
    """
    seq = seq.strip().upper()
    L = len(seq)
    if L == 0:
        return 0, 0.0, 0.0, 0.0

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


def tokenize_sequence(
    seq: str,
    max_len: int,
    add_eos: bool = True,
    eos_id: int = EOS_ID,
) -> Tuple[torch.Tensor, int]:
    """
    把氨基酸序列转为索引，并在末尾添加 EOS（可选），再 pad 到 max_len。
    使用 AA_TO_IDX，0 作为 PAD。

    返回:
        token_ids: (max_len,) LongTensor
        length:    实际 token 序列长度（含 EOS 时包含 EOS，<= max_len）
    """
    seq = seq.strip().upper()
    token_ids = [AA_TO_IDX.get(a, PAD_ID) for a in seq]  # 未知 aa 映射到 PAD

    if add_eos:
        token_ids.append(eos_id)

    length = min(len(token_ids), max_len)
    token_ids = token_ids[:max_len]
    pad_len = max_len - len(token_ids)
    if pad_len > 0:
        token_ids += [PAD_ID] * pad_len

    return torch.tensor(token_ids, dtype=torch.long), length


def load_pdb_ca_coordinates(pdb_path: str, max_len: int) -> np.ndarray:
    """
    读取 PDB 的 C-alpha 坐标，每个残基一个 CA，pad/truncate 到 max_len。
    返回:
        coords: (max_len, 3) float32，pad 部分为 0。
    """
    coords = []
    if not os.path.exists(pdb_path):
        return np.zeros((max_len, 3), dtype=np.float32)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pep", pdb_path)
    except Exception:
        return np.zeros((max_len, 3), dtype=np.float32)

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords.append(atom.coord)
    if len(coords) == 0:
        return np.zeros((max_len, 3), dtype=np.float32)

    coords = np.array(coords, dtype=np.float32)
    if coords.shape[0] > max_len:
        coords = coords[:max_len]
    pad_len = max_len - coords.shape[0]
    if pad_len > 0:
        pad = np.zeros((pad_len, 3), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=0)
    return coords


def build_contact_matrix(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    基于 CA-CA 距离构造接触矩阵：
    contacts[i,j] = 1 if dist(i,j) < threshold and i!=j, else 0
    """
    L = coords.shape[0]
    if np.allclose(coords, 0):
        return np.zeros((L, L), dtype=np.float32)

    diff = coords[:, None, :] - coords[None, :, :]  # (L, L, 3)
    dist = np.linalg.norm(diff, axis=-1)            # (L, L)
    contacts = (dist < threshold).astype(np.float32)
    np.fill_diagonal(contacts, 0.0)
    return contacts


class PeptideStructDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        pdb_dir: str,
        max_len: int = 64,
        pl_ddt_filter: float = 0.0,
        add_eos: bool = True,
        eos_id: int = EOS_ID,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # 可选：按 pLDDT 过滤
        if pl_ddt_filter > 0.0 and "pLDDT" in self.df.columns:
            self.df = self.df[self.df["pLDDT"] >= pl_ddt_filter].reset_index(drop=True)

        self.pdb_dir = pdb_dir
        self.max_len = max_len
        self.add_eos = add_eos
        self.eos_id = eos_id

        # 归一化 pLDDT 到 [0,1]
        if "pLDDT" in self.df.columns:
            pl = self.df["pLDDT"].values.astype(np.float32)
            pl_min, pl_max = pl.min(), pl.max()
            if pl_max > pl_min:
                self.df["pLDDT_norm"] = (pl - pl_min) / (pl_max - pl_min)
            else:
                self.df["pLDDT_norm"] = 0.5
        else:
            self.df["pLDDT_norm"] = 0.5

        # 确保标签列存在
        for col in ["Label", "ABP", "AFP", "AGnP", "AGgP", "AVP"]:
            if col not in self.df.columns:
                self.df[col] = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        raw_seq = row["Sequence"]

        # 清洗后的序列（供 token 化和 ESM 使用）
        seq = clean_sequence_for_esm(raw_seq)
        # 这里会在末尾自动 append EOS（如果 add_eos=True）
        seq_tokens, seq_len = tokenize_sequence(
            seq,
            self.max_len,
            add_eos=self.add_eos,
            eos_id=self.eos_id,
        )

        # PDB 文件名假定为 sequence_{idx+1}.pdb
        pdb_name = f"sequence_{idx+1}.pdb"
        pdb_path = os.path.join(self.pdb_dir, pdb_name)
        coords = load_pdb_ca_coordinates(pdb_path, self.max_len)  # (L,3)
        contacts = build_contact_matrix(coords)                   # (L,L)

        # 计算简单属性：长度/净电荷/比例（注意这里用的是原始氨基酸序列长度，不含 EOS）
        length, net_charge, frac_pos, frac_hydro = compute_simple_props_from_seq(seq)
        length_norm = min(length / float(self.max_len), 1.0)  # [0,1]
        # 假设净电荷在 [-20,20] 左右，缩放到 [-1,1]
        net_charge_norm = max(min(net_charge / 20.0, 1.0), -1.0)

        # labels 向量：7 个原始标签 + 4 个属性 = 11 维
        labels = np.array(
            [
                row["Label"],
                row["ABP"],
                row["AFP"],
                row["AGnP"],
                row["AGgP"],
                row["AVP"],
                row["pLDDT_norm"],
                length_norm,
                net_charge_norm,
                frac_pos,
                frac_hydro,
            ],
            dtype=np.float32,
        )

        sample = {
            "seq_tokens": seq_tokens,                                # (L,)
            "seq_len": seq_len,  # 含 EOS 时包含 EOS
            "coords": torch.tensor(coords, dtype=torch.float32),     # (L,3)
            "contacts": torch.tensor(contacts, dtype=torch.float32), # (L,L)
            "labels": torch.tensor(labels, dtype=torch.float32),     # (11,)
            "raw_seq": seq,      # 清洗后的序列字符串，给 ESM 用（不含 EOS）
            "index": idx,
        }
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 假设所有样本有相同的 max_len
    seq_tokens = torch.stack([b["seq_tokens"] for b in batch], dim=0)   # (B,L)
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    coords = torch.stack([b["coords"] for b in batch], dim=0)           # (B,L,3)
    contacts = torch.stack([b["contacts"] for b in batch], dim=0)       # (B,L,L)
    labels = torch.stack([b["labels"] for b in batch], dim=0)           # (B,11)
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    raw_seqs = [b["raw_seq"] for b in batch]

    return {
        "seq_tokens": seq_tokens,
        "seq_lens": seq_lens,
        "coords": coords,
        "contacts": contacts,
        "labels": labels,
        "indices": indices,
        "raw_seqs": raw_seqs,
    }


def get_dataloaders(
    csv_path: str,
    pdb_dir: str,
    max_len: int = 64,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    pl_ddt_filter: float = 0.0,
    num_workers: int = 0,
    add_eos: bool = True,
    eos_id: int = EOS_ID,
) -> Tuple[DataLoader, DataLoader]:
    dataset = PeptideStructDataset(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        max_len=max_len,
        pl_ddt_filter=pl_ddt_filter,
        add_eos=add_eos,
        eos_id=eos_id,
    )
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader