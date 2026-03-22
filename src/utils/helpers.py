import math
import humanize
import torch
import torch.distributed as dist
import os


def print_trainable_parameters(cfg, model):
    """
    Estimates parameters based on configuration.
    """
    # 1. Architecture Constants
    n_layers = cfg.model.n_layers
    n_embd = cfg.model.n_embd
    vocab_size = cfg.model.vocab_size
    n_heads = cfg.model.n_heads

    # MLA specifics
    k_size = cfg.model.kv_latent_size
    q_size = cfg.model.q_latent_size
    head_size = cfg.model.head_size
    rope_size = cfg.model.rope_head_size
    lat_head_size = head_size - rope_size

    # 2. Embedding + Head
    params_emb = vocab_size * n_embd
    params_ln_f = n_embd  # Final Layer Norm

    # 3. Parameters per Block
    # --- MLA Attention ---
    mla_down_q = n_embd * q_size
    mla_down_kv = n_embd * (k_size + rope_size)
    mla_norms = q_size + k_size
    mla_up_q = q_size * n_heads * (lat_head_size + rope_size)
    mla_up_kv = k_size * n_heads * (lat_head_size + head_size)
    mla_proj = n_heads * head_size * n_embd

    params_mla = mla_down_q + mla_down_kv + mla_norms + mla_up_q + mla_up_kv + mla_proj

    # --- MoE (Shared + Routed) ---
    s_hidden_req = cfg.model.get("n_shared_experts", 0) * cfg.model.get(
        "expert_hidden_size", 0
    )
    s_hidden = (s_hidden_req + 255) // 256 * 256
    # Gate + Up (swiglu usually 2 matrices) + Down. No bias.
    params_shared = (n_embd * s_hidden) + (n_embd * s_hidden) + (s_hidden * n_embd)

    # Routed Experts
    n_routed = cfg.model.get("n_routed_experts", 0)
    topk = cfg.model.get("topk_experts", 0)
    expert_hidden = cfg.model.get("expert_hidden_size", 0)

    # SwiGLU: (n_embd -> 2*h) + (h -> n_embd) -> 3 * n_embd * h
    params_per_expert = 3 * n_embd * expert_hidden

    # MoE Gate
    params_gate = n_embd * n_routed

    params_moe_total = params_shared + params_gate + (n_routed * params_per_expert)
    params_moe_active = params_shared + params_gate + (topk * params_per_expert)

    # Block Layer Norms
    params_block_ln = 2 * n_embd

    # --- Layer Totals ---
    params_layer_total = params_mla + params_moe_total + params_block_ln
    params_layer_active = params_mla + params_moe_active + params_block_ln

    # 4. Final Sums
    total_params = params_emb + (n_layers * params_layer_total) + params_ln_f
    active_params = params_emb + (n_layers * params_layer_active) + params_ln_f

    params_per_shard = 0
    for i, param in enumerate(model.parameters()):
        params_per_shard += param.numel()

    print("| --------------------------------------------------------------------")
    print(f"| Config: {cfg.experiment.run_name}")
    print(f"| Architecture: {n_layers} layers, {n_heads} heads, {n_embd} dim")
    print(
        f"| Experts: {n_routed} routed, {cfg.model.get('n_shared_experts', 0)} shared, TopK: {topk}"
    )
    print("| --------------------------------------------------------------------")
    print(
        f"| Total Params (Storage):      {humanize.intword(total_params)} ({total_params:,})"
    )
    print(
        f"| Active Params (Forward):     {humanize.intword(active_params)} ({active_params:,})"
    )
    print(
        f"| True Params per GPU:         {humanize.intword(params_per_shard)} ({params_per_shard:,})"
    )
    print(f"| Utilization:                 {active_params/total_params:.1%}")


def estimate_flops(cfg):
    """Prints the estimated number of FLOPs per token for the model and for the run."""
    n_layers = cfg.model.n_layers
    n_embd = cfg.model.n_embd
    n_heads = cfg.model.n_heads

    k_size = cfg.model.kv_latent_size
    q_size = cfg.model.q_latent_size
    head_size = cfg.model.head_size
    rope_size = cfg.model.rope_head_size
    lat_head_size = head_size - rope_size

    # MLA Params per layer
    mla_down_q = n_embd * q_size
    mla_down_kv = n_embd * (k_size + rope_size)
    mla_norms = q_size + k_size
    mla_up_q = q_size * n_heads * (lat_head_size + rope_size)
    mla_up_kv = k_size * n_heads * (lat_head_size + head_size)
    mla_proj = n_heads * head_size * n_embd
    params_mla = mla_down_q + mla_down_kv + mla_norms + mla_up_q + mla_up_kv + mla_proj

    # MoE Active Params per layer
    s_hidden_req = cfg.model.get("n_shared_experts", 0) * cfg.model.get(
        "expert_hidden_size", 0
    )
    s_hidden = (s_hidden_req + 255) // 256 * 256
    params_shared = (n_embd * s_hidden) * 3

    topk = cfg.model.get("topk_experts", 0)
    expert_hidden = cfg.model.get("expert_hidden_size", 0)
    params_per_expert = 3 * n_embd * expert_hidden

    params_gate = n_embd * cfg.model.get("n_routed_experts", 0)

    params_moe_active = params_shared + params_gate + (topk * params_per_expert)
    params_block_ln = 2 * n_embd

    active_params_per_layer = params_mla + params_moe_active + params_block_ln
    active_body_params = n_layers * active_params_per_layer

    l, t = n_layers, cfg.model.block_size
    head_dim = n_embd // n_heads

    num_flops_per_token = 6 * active_body_params + 12 * l * n_heads * head_dim * t

    print("| --------------------------------------------------------------------")

    # Account for world_size (number of GPUs)
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()

    # Assuming cfg.training.batch_size is PER-DEVICE batch size
    total_tokens = (
        cfg.training.max_steps
        * cfg.training.batch_size
        * cfg.model.block_size
        * cfg.training.grad_accum_steps
        * world_size
    )

    print(
        f"| Total tokens to be used for training: {humanize.intword(total_tokens)} ({total_tokens:,})"
    )
    print(
        f"| FLOPs per token: {humanize.intword(num_flops_per_token)} ({num_flops_per_token:,})."
    )
    total_flops = num_flops_per_token * total_tokens
    print(
        f"| Total FLOPs for the training run: {humanize.intword(total_flops)} ({total_flops:,})."
    )
    print("| --------------------------------------------------------------------")


def apply_rotary_emb(x, sin, cos):
    """
    Standard RoPE application.
    Expects x to be (B, H, T, D) or broadcastable.
    sin, cos are precomputed and passed in.
    """
    # x is (B, H, T, head_dim)
    # chunk into two halves for the rotation
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    # Standard RoPE rotation formula
    # [-x2, x1] * sin + [x1, x2] * cos
    return torch.cat((-x2, x1), dim=-1) * sin + x * cos


