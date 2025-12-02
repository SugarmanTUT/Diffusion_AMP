import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.esm_encoder import ESM2SeqEncoder
from src.models.se3_struct_encoder import SE3StructEncoder


class SeqStructVAE(nn.Module):
    """
    VAE with:
    - Encoder: ESM2 + SE(3)-invariant structural encoder -> fused per-residue features
    - Latent:  pooled fused features -> (mu, logvar) -> z
    - Decoder: autoregressive Transformer decoder with EOS, variable-length support
    - Structural head: contact map reconstruction from fused features to enforce structure awareness
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 256,
        latent_dim: int = 64,
        n_dec_layers: int = 4,
        n_dec_heads: int = 8,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        finetune_esm: bool = False,
        pad_id: int = 0,
        eos_id: int = None,
        struct_hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.pad_id = pad_id
        # 若未指定 eos_id，则默认取 vocab_size-1 作为 EOS
        self.eos_id = eos_id if eos_id is not None else (vocab_size - 1)

        # ===== 序列侧：ESM2 encoder =====
        self.seq_encoder = ESM2SeqEncoder(
            esm_model_name=esm_model_name,
            finetune=finetune_esm,
        )
        seq_emb_dim = self.seq_encoder.hidden_dim  # e.g. 1280

        # ===== 结构侧：SE(3)-invariant 结构 encoder =====
        self.struct_encoder = SE3StructEncoder(
            in_dim=0,
            hidden_dim=d_model,
            num_layers=3,
        )

        # ===== 融合：序列 + 结构 =====
        self.seq_proj = nn.Linear(seq_emb_dim, d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        # ===== VAE 潜变量层 =====
        self.pool = nn.AdaptiveAvgPool1d(1)  # over L
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ===== 自回归 Transformer decoder =====
        # 把 z 升维到 (B, L, d_model) 作为全局条件 (memory)
        self.z_to_memory = nn.Linear(latent_dim, d_model)

        # token embedding（decoder side）
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_dec_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_dec_layers,
        )
        self.dec_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # ===== 结构重构头（contact map）=====
        self.struct_head = nn.Sequential(
            nn.Linear(d_model, struct_hidden_dim),
            nn.ReLU(),
            nn.Linear(struct_hidden_dim, struct_hidden_dim),
            nn.ReLU(),
        )
        # pairwise bilinear / dot-product head: (B,L,H) -> (B,L,L)
        self.struct_out = nn.Linear(struct_hidden_dim, struct_hidden_dim, bias=False)

    # ---------------- Encoder ----------------
    def encode(self, batch):
        """
        batch keys:
        - seq_tokens: (B, L)
        - coords:     (B, L, 3)
        - contacts:   (B, L, L)
        - raw_seqs:   List[str]
        """
        seq_tokens = batch["seq_tokens"]
        coords = batch["coords"]
        contacts = batch["contacts"]
        raw_seqs = batch["raw_seqs"]

        device = seq_tokens.device

        # ESM2 序列 embedding
        h_seq_esm = self.seq_encoder(seq_tokens, raw_seqs, device=device)  # (B, L, d_esm)
        h_seq = self.seq_proj(h_seq_esm)  # (B, L, d_model)

        # 结构 embedding
        h_struct = self.struct_encoder(coords, contacts)  # (B, L, d_model)

        # 融合
        h_cat = torch.cat([h_seq, h_struct], dim=-1)  # (B,L,2*d_model)
        h_fuse = self.fuse_proj(h_cat)                # (B,L,d_model)

        # pooling -> (B,d_model)
        h_fuse_t = h_fuse.transpose(1, 2)             # (B,d_model,L)
        pooled = self.pool(h_fuse_t).squeeze(-1)      # (B,d_model)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar, h_fuse

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------------- Structural reconstruction head ----------------
    def reconstruct_contacts(self, h_fuse):
        """
        从 encoder 输出的 h_fuse (B,L,d_model) 预测 contact map (B,L,L) 概率。
        """
        B, L, D = h_fuse.shape
        h = self.struct_head(h_fuse)  # (B,L,H)
        # 先通过一个线性映射
        h_proj = self.struct_out(h)   # (B,L,H)
        # pairwise score: (B,L,L)
        logits = torch.matmul(h_proj, h.transpose(1, 2))
        # 为了数值稳定，可以缩放一下
        logits = logits / (h_proj.size(-1) ** 0.5)
        return logits  # (B,L,L)

    # ---------------- Autoregressive decoder ----------------
    def decode_autoregressive(self, z, target_tokens=None, teacher_forcing: bool = True):
        """
        自回归解码：
        - 训练时：teacher_forcing=True，输入 target_tokens 的右移版本；
        - 推理时：teacher_forcing=False，逐步生成。
        输入：
            z: (B, latent_dim)
            target_tokens: (B, L_max) LongTensor or None
        返回：
            logits: (B, L_max, vocab_size)  [teacher_forcing=True]
            或
            tokens: (B, L_max)              [teacher_forcing=False]
        """
        B = z.size(0)
        device = z.device

        # memory: (B,L_max,d_model)，每个位置都是 z 的投影（全局上下文）
        mem = self.z_to_memory(z).unsqueeze(1).expand(B, self.max_len, self.d_model)

        if teacher_forcing:
            assert target_tokens is not None, "target_tokens is required in teacher forcing mode"
            # decoder input = 左移一位的目标序列（第一位用 PAD 作为 BOS）
            inp = target_tokens.clone()
            inp[:, 1:] = target_tokens[:, :-1]
            inp[:, 0] = self.pad_id

            tok_emb = self.token_embed(inp)  # (B,L,d_model)
            pos_ids = torch.arange(self.max_len, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embed(pos_ids)
            dec_inp = tok_emb + pos_emb  # (B,L,d_model)

            # causal mask: (L,L)，True 代表屏蔽（不看未来）
            tgt_mask = torch.triu(
                torch.ones(self.max_len, self.max_len, device=device),
                diagonal=1,
            ).bool()

            dec_out = self.decoder(
                tgt=dec_inp,
                memory=mem,
                tgt_mask=tgt_mask,
            )  # (B,L,d_model)
            dec_out = self.dec_ln(dec_out)
            logits = self.out_proj(dec_out)  # (B,L,V)
            return logits

        else:
            # 推理：一步一步生成
            tokens = torch.full(
                (B, self.max_len),
                self.pad_id,
                dtype=torch.long,
                device=device,
            )
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for t in range(self.max_len):
                pos_ids = torch.arange(self.max_len, device=device).unsqueeze(0).expand(B, -1)
                tok_emb = self.token_embed(tokens)
                pos_emb = self.pos_embed(pos_ids)
                dec_inp = tok_emb + pos_emb

                tgt_mask = torch.triu(
                    torch.ones(self.max_len, self.max_len, device=device),
                    diagonal=1,
                ).bool()
                dec_out = self.decoder(
                    tgt=dec_inp,
                    memory=mem,
                    tgt_mask=tgt_mask,
                )
                dec_out = self.dec_ln(dec_out)
                logits = self.out_proj(dec_out)  # (B,L,V)

                # 当前时间步 t 的输出
                step_logits = logits[:, t, :]  # (B,V)
                step_tokens = step_logits.argmax(dim=-1)  # (B,)

                # 更新尚未 finished 的序列
                update_mask = ~finished
                tokens[update_mask, t] = step_tokens[update_mask]

                # 若本步生成 EOS，则标记为 finished
                finished = finished | (step_tokens == self.eos_id)

                # 若所有样本都 finished，提前结束
                if finished.all():
                    break

            return tokens  # (B,L)

    # ---------------- Forward & Loss ----------------
    def forward(self, batch):
        """
        Forward 仅做 encode + reparameterize + decode（teacher forcing），
        loss 计算放在静态方法 loss_function 里。
        """
        seq_tokens = batch["seq_tokens"]  # (B,L)
        contacts = batch["contacts"]      # (B,L,L)

        mu, logvar, h_fuse = self.encode(batch)
        z = self.reparameterize(mu, logvar)

        # 自回归 decoder logits (teacher forcing)
        logits = self.decode_autoregressive(z, target_tokens=seq_tokens, teacher_forcing=True)

        # 结构重构 logits
        contact_logits = self.reconstruct_contacts(h_fuse)  # (B,L,L)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "logits": logits,               # for token reconstruction
            "contact_logits": contact_logits,  # for structure reconstruction
            "true_contacts": contacts,      # (B,L,L)
        }

    @staticmethod
    def loss_function(outputs, batch, kl_weight: float = 1e-3, struct_weight: float = 1.0):
        """
        - outputs: forward 的返回 dict
        - batch:   包含 seq_tokens, contacts
        计算:
        - 自回归重构损失（CE，带 EOS，忽略 PAD）
        - KL 损失
        - 结构重构损失（contact map BCE）
        """
        logits = outputs["logits"]            # (B,L,V)
        mu = outputs["mu"]                    # (B,latent_dim)
        logvar = outputs["logvar"]            # (B,latent_dim)
        true_tokens = batch["seq_tokens"]     # (B,L)

        contact_logits = outputs["contact_logits"]  # (B,L,L)
        true_contacts = outputs["true_contacts"]    # (B,L,L)

        B, L, V = logits.shape

        # ---- 自回归 token 重构 loss ----
        logits_flat = logits.view(B * L, V)
        targets_flat = true_tokens.view(B * L)

        recon_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=0,  # 忽略 PAD
        )

        # ---- KL loss ----
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (B,)
        kl_loss = kl.mean()

        # ---- 结构重构 loss (BCE with logits) ----
        contact_logits_flat = contact_logits.view(B, L * L)
        true_contacts_flat = true_contacts.view(B, L * L)

        struct_loss = F.binary_cross_entropy_with_logits(
            contact_logits_flat,
            true_contacts_flat,
        )

        loss = recon_loss + kl_weight * kl_loss + struct_weight * struct_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "struct_loss": struct_loss,
        }

    @torch.no_grad()
    def generate(self, z):
        """
        采样接口：给定 latent z，使用自回归解码生成 token 序列。
        z: (B,latent_dim)
        返回:
            tokens: (B,L)
        """
        tokens = self.decode_autoregressive(z, target_tokens=None, teacher_forcing=False)
        return tokens