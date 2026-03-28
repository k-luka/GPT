import math
import humanize
import torch
import torch.distributed as dist
import os


def print_trainable_parameters(cfg, model):
    """
    Estimates parameters based on configuration.
    """
    n_layers = cfg.model.n_layers
    n_embd = cfg.model.n_embd
    vocab_size = cfg.model.vocab_size
    n_heads = cfg.model.n_heads
    model_type = cfg.model.get("model_type", "gpt")

    # Embedding (tied with lm_head, counted once) + final LN
    params_emb = vocab_size * n_embd
    params_ln_f = n_embd

    # Standard QKV attention used by all model types:
    #   fused QKV (n_embd -> 3*n_embd) + output proj + q_norm + k_norm
    H = n_embd // n_heads
    params_attn = 4 * n_embd * n_embd + 2 * H

    # Block layer norms (ln1 + ln2)
    params_block_ln = 2 * n_embd

    if model_type == "gpt_moe":
        n_routed = cfg.model.get("n_routed_experts", 0)
        topk = cfg.model.get("topk_experts", 0)
        expert_hidden = cfg.model.get("expert_hidden_size", 0)

        s_hidden_req = cfg.model.get("n_shared_experts", 0) * expert_hidden
        s_hidden = (s_hidden_req + 255) // 256 * 256
        params_shared = 3 * n_embd * s_hidden
        params_gate = n_embd * n_routed
        params_per_expert = 3 * n_embd * expert_hidden

        params_mlp_total = params_shared + params_gate + n_routed * params_per_expert
        params_mlp_active = params_shared + params_gate + topk * params_per_expert
    else:
        n_routed = 0
        topk = 0
        hidden_dim = int(8 * n_embd // 3)
        hidden_dim = (hidden_dim + 255) // 256 * 256
        params_mlp_total = 3 * n_embd * hidden_dim
        params_mlp_active = params_mlp_total

    params_layer_total = params_attn + params_mlp_total + params_block_ln
    params_layer_active = params_attn + params_mlp_active + params_block_ln

    total_params = params_emb + n_layers * params_layer_total + params_ln_f
    active_params = params_emb + n_layers * params_layer_active + params_ln_f

    params_per_shard = sum(p.numel() for p in model.parameters())

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
    model_type = cfg.model.get("model_type", "gpt")

    # Standard QKV attention (all model types)
    params_attn = 4 * n_embd * n_embd

    if model_type == "gpt_moe":
        s_hidden_req = cfg.model.get("n_shared_experts", 0) * cfg.model.get(
            "expert_hidden_size", 0
        )
        s_hidden = (s_hidden_req + 255) // 256 * 256
        params_shared = 3 * n_embd * s_hidden
        params_gate = n_embd * cfg.model.get("n_routed_experts", 0)
        params_per_expert = 3 * n_embd * cfg.model.get("expert_hidden_size", 0)
        topk = cfg.model.get("topk_experts", 0)
        params_mlp_active = params_shared + params_gate + topk * params_per_expert
    else:
        hidden_dim = int(8 * n_embd // 3)
        hidden_dim = (hidden_dim + 255) // 256 * 256
        params_mlp_active = 3 * n_embd * hidden_dim

    params_block_ln = 2 * n_embd
    active_params_per_layer = params_attn + params_mlp_active + params_block_ln
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
