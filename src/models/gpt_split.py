import torch
import torch.nn.functional as F
import torch.nn as nn
from src.utils.helpers import apply_rotary_emb
from src.utils.optimizers import Muon, DualOptimizer
import math
import inspect

"""
GPT variant with split QKV and optionally split gate/up projections.
Designed to test whether fused projections hurt Muon's orthogonalization.
"""

# Global flag set from config before model construction
SPLIT_MLP = False


class Attention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        assert (
            n_embd % n_heads == 0
        ), f"Embedding dim ({n_embd}) must be divisible by number of heads ({n_heads})."
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.H = n_embd // n_heads

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.H)
        self.k_norm = nn.RMSNorm(self.H)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.H).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.H).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.H).transpose(1, 2)
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLPFused(nn.Module):
    """Original fused gate/up SwiGLU MLP (used when split_mlp=False)."""

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(8 * n_embd // 3)
        self.hidden_dim = (hidden_dim + 255) // 256 * 256
        self.gate_proj = nn.Linear(n_embd, self.hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x):
        y, gate = torch.chunk(self.gate_proj(x), 2, dim=-1)
        gate = F.silu(gate)
        y = gate * y
        return self.down_proj(y)


class MLPSplit(nn.Module):
    """Split gate/up SwiGLU MLP -- each projection is a square-ish matrix for Muon."""

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(8 * n_embd // 3)
        self.hidden_dim = (hidden_dim + 255) // 256 * 256
        self.gate_proj = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        y = gate * self.up_proj(x)
        return self.down_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, split_mlp=False):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.sa = Attention(n_embd, n_heads)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = MLPSplit(n_embd) if split_mlp else MLPFused(n_embd)

    def forward(self, x, sin, cos):
        x = x + self.sa(self.ln1(x), sin, cos)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_embd,
        vocab_size,
        block_size,
        n_heads,
        head_size,
        rope_head_size,
        kv_latent_size,
        q_latent_size,
        n_layers,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, n_embd)

        sin, cos = self._precompute_rotary_embeddings(
            block_size, (n_embd // n_heads)
        )  # pyrefly: ignore
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        self.transformer = nn.ModuleList(
            [
                Block(n_embd, n_heads, split_mlp=SPLIT_MLP)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.01
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        x = self.wte(idx)
        sin = self.sin[:, :, :T, :]  # pyrefly: ignore
        cos = self.cos[:, :, :T, :]  # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos)
        x = self.ln(x)

        if targets is not None:
            logits = self.lm_head(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return None, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def generate(
        self,
        idx,
        num_sequences=5,
        max_tokens=200,
        topk=50,
        chat_mode=False,
        eos_token=50256,
    ):
        idx = torch.repeat_interleave(idx.unsqueeze(0), num_sequences, dim=0)

        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]  # pyrefly: ignore
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=topk)
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            idx_next = torch.gather(topk_indices, -1, idx_next)

            if chat_mode and (idx_next == eos_token).all():
                break

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, device,
                             use_muon=True, muon_wd=None):
        if muon_wd is None:
            muon_wd = weight_decay

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        adamw_decay_params = []
        adamw_nodecay_params = []
        muon_params = []
        seen = set()

        for pn, p in param_dict.items():
            if p in seen:
                continue
            seen.add(p)

            if use_muon and p.dim() >= 2 and "wte" not in pn and "lm_head" not in pn:
                muon_params.append(p)
            elif p.dim() >= 2:
                adamw_decay_params.append(p)
            else:
                adamw_nodecay_params.append(p)

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in str(device)

        if use_muon:
            print(f"Muon params (2D hidden): {len(muon_params)} tensors")
            print(f"AdamW decay params (Embed/Head): {len(adamw_decay_params)} tensors")
            print(f"AdamW no-decay params (1D norms): {len(adamw_nodecay_params)} tensors")
            print(f"Muon weight decay: {muon_wd}")
            print(f"Using fused AdamW: {use_fused}")

            adam_opt = torch.optim.AdamW(
                [
                    {"params": adamw_decay_params, "weight_decay": weight_decay},
                    {"params": adamw_nodecay_params, "weight_decay": 0.0}
                ],
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )

            muon_opt = Muon(
                [{"params": muon_params}],
                lr=0.01,
                momentum=0.95,
                weight_decay=muon_wd,
            )

            return DualOptimizer(adam_opt, muon_opt)
        else:
            print(f"Decayed params (2D): {len(adamw_decay_params)} tensors")
            print(f"Non-decayed params (1D): {len(adamw_nodecay_params)} tensors")
            print(f"Using fused AdamW: {use_fused}")

            return torch.optim.AdamW(
                [
                    {"params": adamw_decay_params, "weight_decay": weight_decay},
                    {"params": adamw_nodecay_params, "weight_decay": 0.0}
                ],
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
