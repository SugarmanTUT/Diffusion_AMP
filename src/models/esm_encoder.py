import torch
import torch.nn as nn

try:
    import esm
except ImportError:
    esm = None
    print("WARNING: fair-esm is not installed. Please run `pip install fair-esm`.")


class ESM2SeqEncoder(nn.Module):
    """
    使用 facebook/esm2_t33_650M_UR50D 作为序列编码器，输出 per-residue embedding。

    设计要点：
    - 使用 fair-esm 官方提供的预训练模型和 alphabet；
    - 默认冻结 ESM 参数（finetune=False），只做特征提取；
    - 输入是 batch 中的 raw 序列字符串列表；
    - 输出是对齐到 VAE 使用的 max_len 的 (B, L, d_esm) 张量。
    """

    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        finetune: bool = False,
    ):
        super().__init__()
        assert esm is not None, (
            "fair-esm is not installed. Please install via `pip install fair-esm` "
            "or follow https://github.com/facebookresearch/esm"
        )

        self.esm_model_name = esm_model_name
        self.finetune = finetune

        # 加载预训练模型
        print(f"Loading ESM2 model: {esm_model_name}")
        self.esm_model, self.alphabet = esm.pretrained.__dict__[esm_model_name]()
        self.batch_converter = self.alphabet.get_batch_converter()

        # 冻结或解冻
        if not finetune:
            for p in self.esm_model.parameters():
                p.requires_grad = False

        # ESM2 的隐藏维度，例如 t33_650M 是 1280
        self.hidden_dim = self.esm_model.embed_dim

    def forward(self, seq_tokens: torch.Tensor, raw_seqs, device=None):
        """
        seq_tokens: (B, L)  — 这里只用它的长度 L 作为截断/填充目标，不参与 ESM 输入。
        raw_seqs: List[str]  — 批次中的原始氨基酸序列（我们假定 dataset 已经清洗成合法 AA 字符）。

        返回:
            h_seq: (B, L_max, hidden_dim)
              - L_max 来自 seq_tokens.shape[1]（与你的 VAE max_len 一致）
              - 超过部分被截断，不足部分用 0 padding。
        """
        if device is None:
            device = next(self.parameters()).device

        max_len = seq_tokens.shape[1]  # VAE 使用的最大长度
        B = len(raw_seqs)

        # 清洗 + 构造 ESM batch_converter 所需的格式
        # data: List[(name, seq_str)]
        data = []
        for i, s in enumerate(raw_seqs):
            if not isinstance(s, str):
                s = str(s)
            # 去掉前后空白
            s = s.strip()
            # 去掉所有空白字符（空格、tab、换行）
            s = "".join(ch for ch in s if not ch.isspace())
            # 转为大写
            s = s.upper()
            data.append((f"seq_{i}", s))

        # 使用 ESM 的 batch_converter 得到 tokens
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(device)  # (B, L_esm)

        # ESM 前向
        # 注意：repr_layers={33} 指取最后一层的表示（对 t33 模型而言）
        with torch.set_grad_enabled(self.finetune):
            out = self.esm_model(
                batch_tokens,
                repr_layers=[33],
                need_head_weights=False,
            )
            token_reps = out["representations"][33]  # (B, L_esm, hidden_dim)

        # 去掉 BOS 和 EOS：
        #   ESM 的 token_reps[ :, 0, :] 是 BOS，
        #   token_reps[ :, 1:-1, :] 对应实际氨基酸，
        #   token_reps[ :, -1, :] 是 EOS。
        token_reps = token_reps[:, 1:-1, :]  # (B, L_seq, D)

        # 对齐到 max_len：截断或 padding
        D = token_reps.size(-1)
        h_seq = torch.zeros((B, max_len, D), device=device, dtype=token_reps.dtype)

        for i, s in enumerate(batch_strs):
            # batch_strs[i] 是 ESM 看到的序列字符串（与 data 中的一致）
            L_seq = len(s)
            L_use = min(L_seq, max_len)
            h_seq[i, :L_use, :] = token_reps[i, :L_use, :]

        return h_seq  # (B, max_len, hidden_dim)