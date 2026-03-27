import torch
import torch.nn.functional as F
import torch.nn as nn
from src.utils.helpers import apply_rotary_emb
from src.utils.optimizers import Muon, DualOptimizer
import math
import inspect

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print(
        "Warning: transformer_engine not installed — expert loop fallback active. "
        "Install TE for parallel grouped expert execution."
    )

"""
GPT with DeepSeek-style Fine-Grained Mixture of Experts (MoE).

Each transformer block replaces the dense MLP with a MoE layer consisting of:
  - Shared experts: always-active SwiGLU MLPs (aggregate hidden ~ n_shared * expert_hidden)
  - Routed experts: top-k selected per token from a large pool (n_routed_experts total)

Total active computation per token ≈ (n_shared + topk) * expert_hidden,
which is tuned to match the standard MLP's hidden dim (~8/3 * n_embd).

Expert execution (when transformer_engine is available):
  All routed expert GEMMs run in parallel in a single te.GroupedLinear kernel call
  instead of looping over experts sequentially. Tokens are first sorted by their
  assigned expert (so each expert's tokens are contiguous), then passed to
  GroupedLinear with m_splits specifying how many tokens each expert receives.

Load balancing uses DeepSeek's auxiliary-loss-free method: a learnable bias
in the gate is updated at each step via sign(target_load - actual_load) to
keep expert utilization uniform without adding an auxiliary loss term.

Expert load is tracked in MoE.last_global_counts and exposed via GPT.get_expert_loads()
for external logging.
"""


class Attention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        assert (
            n_embd % n_heads == 0
        ), f"Embedding dim ({n_embd}) must be divisible by number of heads ({n_heads})."
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.H = n_embd // n_heads
        self.attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.H)
        self.k_norm = nn.RMSNorm(self.H)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_heads, self.H).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.H).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.H).transpose(1, 2)
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SharedExpert(nn.Module):
    """Always-active SwiGLU expert with combined hidden size for n_shared experts."""

    def __init__(self, n_embd, n_shared_experts, expert_hidden_size):
        super().__init__()
        raw_hidden = n_shared_experts * expert_hidden_size
        self.hidden_dim = (raw_hidden + 255) // 256 * 256
        self.gate_proj = nn.Linear(n_embd, self.hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x):
        y, gate = torch.chunk(self.gate_proj(x), 2, dim=-1)
        gate = F.silu(gate)
        return self.down_proj(gate * y)


class Expert(nn.Module):
    """Single small routed SwiGLU expert (used as fallback when TE is not available)."""

    def __init__(self, n_embd, expert_hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, expert_hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x):
        y, gate = torch.chunk(self.gate_proj(x), 2, dim=-1)
        gate = F.silu(gate)
        return self.down_proj(gate * y)


class Gate(nn.Module):
    """
    Sigmoid-based top-k router with a learnable load-balancing bias.

    The bias is a registered buffer (not a gradient parameter) — it is updated
    manually by MoE.update_bias() after each forward pass.
    """

    def __init__(self, n_embd, n_routed_experts, topk, route_scale=1.0):
        super().__init__()
        self.topk = topk
        self.route_scale = route_scale
        self.linear = nn.Linear(n_embd, n_routed_experts, bias=False)
        # Bias updated by load balancing — not a gradient parameter
        self.register_buffer("bias", torch.zeros(n_routed_experts))

    def forward(self, x):
        """Returns (topk_idx, weights) for each token in x (N, C)."""
        logits = self.linear(x)
        scores = logits.sigmoid()

        # Detached bias so load balancing doesn't affect gradient flow
        topk_idx = torch.topk(scores + self.bias.detach(), self.topk, dim=-1)[1]

        # Gather scores for selected experts and normalize per token
        weights = torch.gather(scores, -1, topk_idx)
        weights = (weights / weights.sum(-1, keepdim=True)) * self.route_scale
        return topk_idx, weights


class MoE(nn.Module):
    """
    Fine-grained Mixture of Experts (single-GPU).

    Combines:
      - SharedExpert: always-active, fused hidden = n_shared * expert_hidden (rounded to 256)
      - n_routed_experts small routed experts, topk selected per token via Gate

    Expert execution (two paths):
      TE path (HAS_TE=True):
        Tokens are sorted by assigned expert then passed to te.GroupedLinear, which
        runs all n_routed_experts GEMMs in a single fused kernel call. This replaces
        the sequential Python loop entirely.
      Fallback path (HAS_TE=False):
        Sorted-slice loop over nn.ModuleList of Expert modules (one kernel pair per expert).

    Load balancing (DeepSeek auxiliary-loss-free):
      After each forward, update gate bias: bias += 0.001 * sign(target_load - actual_load)
      where target_load = 1/n_routed_experts (uniform distribution).

    Expert utilization is stored in self.last_global_counts for external logging.
    """

    def __init__(
        self, n_embd, n_shared_experts, n_routed_experts, topk, expert_hidden_size
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.expert_hidden_size = expert_hidden_size

        self.gate = Gate(n_embd, n_routed_experts, topk)
        self.shared_expert = SharedExpert(n_embd, n_shared_experts, expert_hidden_size)

        if HAS_TE:
            # Two GroupedLinear layers cover all n_routed_experts simultaneously.
            # GroupedLinear(num_gemms=E) stores E separate weight matrices and fuses
            # all E GEMMs into a single kernel call when given m_splits per expert.
            self.up_proj = te.GroupedLinear(
                in_features=n_embd,
                out_features=expert_hidden_size * 2,  # fused gate+up for SwiGLU
                num_gemms=n_routed_experts,
                bias=False,
            )
            self.down_proj = te.GroupedLinear(
                in_features=expert_hidden_size,
                out_features=n_embd,
                num_gemms=n_routed_experts,
                bias=False,
            )
        else:
            self.experts = nn.ModuleList(
                [Expert(n_embd, expert_hidden_size) for _ in range(n_routed_experts)]
            )

        # Populated during training for external expert-load logging
        self.last_global_counts: list[int] | None = None

    def update_bias(self, token_counts, update_rate=0.001):
        """DeepSeek auxiliary-loss-free load balancing via bias correction."""
        with torch.no_grad():
            total = token_counts.sum()
            if total == 0:
                return
            actual_load = token_counts.float() / total
            target_load = 1.0 / self.n_routed_experts
            correction = torch.sign(target_load - actual_load)
            self.gate.bias.add_(update_rate * correction)  # pyrefly: ignore

    def _route_tokens(self, x_flat, topk_idx, weights, N):
        """
        Shared permutation logic for both TE and fallback paths.

        Returns:
          x_permuted   : (N*topk, C) — input tokens sorted by assigned expert
          token_ids_s  : (N*topk,)   — original token index for each permuted entry
          flat_weights_s: (N*topk,)  — routing weight for each permuted entry
          token_counts : (n_routed_experts,) — tokens per expert (on GPU)
          m_splits     : list[int]   — same, on CPU (for GroupedLinear / loop)
        """
        expert_ids = topk_idx.reshape(-1)   # (N*topk,)
        token_ids = (
            torch.arange(N, device=x_flat.device)
            .unsqueeze(1)
            .expand(N, self.topk)
            .reshape(-1)
        )
        flat_weights = weights.reshape(-1)  # (N*topk,)

        # Sort all (token, expert) pairs by expert index so each expert's tokens
        # are contiguous — required by GroupedLinear's m_splits convention.
        order = torch.argsort(expert_ids, stable=True)
        token_ids_s = token_ids[order]
        flat_weights_s = flat_weights[order]

        # Token count per expert: one CPU sync for m_splits (unavoidable for GroupedLinear)
        token_counts = expert_ids.bincount(minlength=self.n_routed_experts)
        m_splits = token_counts.tolist()

        x_permuted = x_flat[token_ids_s]   # (N*topk, C)
        return x_permuted, token_ids_s, flat_weights_s, token_counts, m_splits

    def _forward_grouped(self, x_flat, topk_idx, weights, N):
        """
        TE path: all expert GEMMs in two fused te.GroupedLinear kernel calls.

        GroupedLinear requires inputs sorted by expert (m_splits[i] tokens for expert i
        come first). The sort is done in _route_tokens. After computing, outputs are
        scatter-added back to original token positions weighted by routing scores.
        """
        x_permuted, token_ids_s, flat_weights_s, token_counts, m_splits = (
            self._route_tokens(x_flat, topk_idx, weights, N)
        )

        # Single fused kernel: all n_routed_experts up-projections in parallel
        up = self.up_proj(x_permuted, m_splits=m_splits)  # (N*topk, expert_hidden*2)
        gate_val, up_val = up.chunk(2, dim=-1)
        h = F.silu(gate_val) * up_val                     # (N*topk, expert_hidden)

        # Single fused kernel: all n_routed_experts down-projections in parallel
        out_permuted = self.down_proj(h, m_splits=m_splits)  # (N*topk, C)

        # Apply routing weights and scatter-add back to original token positions
        out_weighted = (out_permuted * flat_weights_s.unsqueeze(-1)).to(x_flat.dtype)
        output = torch.zeros_like(x_flat)
        output.scatter_add_(
            0, token_ids_s.unsqueeze(1).expand_as(out_weighted), out_weighted
        )
        return output, token_counts

    def _forward_sequential(self, x_flat, topk_idx, weights, N):
        """
        Fallback path: sorted-slice loop over individual Expert modules.
        Each expert still gets its own matmul kernels, but only non-empty experts
        are launched and token gather/scatter uses contiguous slices.
        """
        _, token_ids_s, flat_weights_s, token_counts, m_splits = (
            self._route_tokens(x_flat, topk_idx, weights, N)
        )

        output = torch.zeros_like(x_flat)
        offset = 0
        for expert, count in zip(self.experts, m_splits):
            if count > 0:
                tok = token_ids_s[offset : offset + count]
                w = flat_weights_s[offset : offset + count].unsqueeze(1)
                out = (expert(x_flat[tok]) * w).to(output.dtype)
                output.scatter_add_(0, tok.unsqueeze(1).expand_as(out), out)
            offset += count

        return output, token_counts

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (N, C),  N = B*T
        N = x_flat.shape[0]

        shared = self.shared_expert(x)  # (B, T, C)

        topk_idx, weights = self.gate(x_flat)  # (N, topk) each

        if HAS_TE:
            output, token_counts = self._forward_grouped(x_flat, topk_idx, weights, N)
        else:
            output, token_counts = self._forward_sequential(x_flat, topk_idx, weights, N)

        if self.training:
            self.last_global_counts = token_counts.tolist()
            self.update_bias(token_counts)

        return shared + output.view(B, T, C)


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        n_shared_experts,
        n_routed_experts,
        topk,
        expert_hidden_size,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.sa = Attention(n_embd, n_heads)
        self.ln2 = nn.RMSNorm(n_embd)
        self.moe = MoE(
            n_embd, n_shared_experts, n_routed_experts, topk, expert_hidden_size
        )

    def forward(self, x, sin, cos):
        x = x + self.sa(self.ln1(x), sin, cos)
        x = x + self.moe(self.ln2(x))
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
        n_shared_experts=2,
        n_routed_experts=64,
        topk_experts=6,
        expert_hidden_size=256,
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
                Block(
                    n_embd,
                    n_heads,
                    n_shared_experts,
                    n_routed_experts,
                    topk_experts,
                    expert_hidden_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # te.GroupedLinear weights are not nn.Linear, so _init_weights misses them.
        # Initialize them here with the same scheme: std=0.01 for up, residual-scaled for down.
        if HAS_TE:
            residual_std = 0.01 / math.sqrt(2 * n_layers)
            for block in self.transformer:
                for p in block.moe.up_proj.parameters():
                    nn.init.normal_(p, mean=0.0, std=0.01)
                for p in block.moe.down_proj.parameters():
                    nn.init.normal_(p, mean=0.0, std=residual_std)

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

    def get_expert_loads(self) -> dict[str, list[int]] | None:
        """
        Returns per-layer expert token counts from the last forward pass.
        Returns None if no counts are available (e.g., before training starts).
        Keys are layer indices as strings; values are lists of int counts per expert.
        """
        loads = {}
        for i, block in enumerate(self.transformer):
            counts = block.moe.last_global_counts
            if counts is not None:
                loads[str(i)] = counts
        return loads if loads else None

    def configure_optimizers(
        self, weight_decay, learning_rate, device, use_muon=True, muon_wd=None, muon_backend="custom"
    ):
        if muon_wd is None:
            muon_wd = weight_decay

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # te.GroupedLinear weights must go to AdamW regardless of dim —
        # Muon's orthogonalization is not designed for concatenated multi-expert weights.
        te_grouped_param_ids: set[int] = set()
        if HAS_TE:
            for module in self.modules():
                if isinstance(module, te.GroupedLinear):
                    for p in module.parameters():
                        te_grouped_param_ids.add(id(p))

        adamw_decay_params = []
        adamw_nodecay_params = []
        muon_params = []
        seen = set()

        for pn, p in param_dict.items():
            if p in seen:
                continue
            seen.add(p)

            if id(p) in te_grouped_param_ids:
                # GroupedLinear expert weights → AdamW decay
                adamw_decay_params.append(p)
            elif use_muon and p.dim() >= 2 and "wte" not in pn and "lm_head" not in pn:
                muon_params.append(p)
            elif p.dim() >= 2:
                adamw_decay_params.append(p)
            else:
                adamw_nodecay_params.append(p)

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in str(device)

        if use_muon:
            print(f"Muon params (2D hidden): {len(muon_params)} tensors")
            print(f"AdamW decay params (Embed/Head + GroupedLinear): {len(adamw_decay_params)} tensors")
            print(
                f"AdamW no-decay params (1D norms): {len(adamw_nodecay_params)} tensors"
            )
            print(f"Muon weight decay: {muon_wd}")
            print(f"Using fused AdamW: {use_fused}")
            print(f"TE GroupedLinear active: {HAS_TE}")

            adam_opt = torch.optim.AdamW(
                [
                    {"params": adamw_decay_params, "weight_decay": weight_decay},
                    {"params": adamw_nodecay_params, "weight_decay": 0.0},
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
            print(f"Decayed params (2D + GroupedLinear): {len(adamw_decay_params)} tensors")
            print(f"Non-decayed params (1D): {len(adamw_nodecay_params)} tensors")
            print(f"Using fused AdamW: {use_fused}")
            print(f"TE GroupedLinear active: {HAS_TE}")

            return torch.optim.AdamW(
                [
                    {"params": adamw_decay_params, "weight_decay": weight_decay},
                    {"params": adamw_nodecay_params, "weight_decay": 0.0},
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
