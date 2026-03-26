# CLAUDE.md — GPT Training Codebase Guide

## Project Overview

This is a research-oriented GPT (Generative Pre-trained Transformer) training framework focused on comparing optimizer strategies and architecture variations. The primary goal is to evaluate the **Muon optimizer** (Newton-Schulz orthogonalization) against AdamW across 8 controlled experiments.

- **Author:** Kirill Luka
- **License:** MIT
- **Language:** Python 3 + PyTorch
- **Target hardware:** Single GPU (NVIDIA B200) on HPC cluster via SLURM

---

## Repository Structure

```
GPT/
├── pretrain.py                    # Main entry point — Hydra-based training launcher
├── config/
│   ├── config_basemodel.yaml      # Base model/training configuration
│   └── experiments/               # Per-experiment config overrides (exp_0 through exp_7)
├── scripts/
│   ├── baseline_model.sh          # SLURM job script for a single run
│   ├── run_experiments.sh         # Sequentially submits all 8 experiments
│   └── data_prep/
│       └── hellaswag.py           # Downloads and parses HellaSwag benchmark
└── src/
    ├── datasets/
    │   └── dataloader.py          # Memory-mapped token shard loader
    ├── eval/
    │   └── metrics.py             # Validation loss + HellaSwag evaluation
    ├── models/
    │   ├── gpt.py                 # Standard GPT (fused QKV + SwiGLU + RoPE)
    │   └── gpt_split.py           # GPT variant with separate Q/K/V and optional split MLP
    ├── training/
    │   └── trainer_single_gpu.py  # Full training loop, LR schedule, checkpointing, wandb
    └── utils/
        ├── helpers.py             # RoPE implementation, FLOPs estimation, param counting
        └── optimizers.py          # Muon optimizer + DualOptimizer wrapper
```

---

## Core Components

### Entry Point: `pretrain.py`

- Decorated with `@hydra.main` — config is loaded automatically from `config/config_basemodel.yaml`
- Experiment overrides are applied by passing `+experiments=exp_X_name` on the CLI
- Initializes wandb, sets global seed, loads tiktoken GPT-2 encoder
- Selects model class (`GPT` or `GPT_split`) based on `cfg.model.use_split`
- Compiles model with `torch.compile()` before training

### Models

**`src/models/gpt.py`** — Standard GPT
- `Attention`: Fused QKV projection, multi-head attention, RoPE embeddings, causal mask
- `MLP`: SwiGLU with hidden dim `= 4 * n_embd * 8/3` (rounded to multiple of 64)
- `Block`: Pre-RMSNorm residual block (attention + MLP)
- `GPT`: Full model with weight tying (embedding ↔ output projection), `configure_optimizers()` method that splits parameters for AdamW vs Muon

**`src/models/gpt_split.py`** — Split-projection variant
- Separates Q, K, V into individual linear layers instead of one fused projection
- Adds `MLPFused` (gate+up fused) and `MLPSplit` (gate+up separate) variants
- Controlled by module-level `SPLIT_MLP` flag (set externally before instantiation)

### Training: `src/training/trainer_single_gpu.py`

Key behavior:
- **Gradient accumulation**: `global_batch_tokens / (T * B)` micro-steps per update
- **LR schedule**: linear warmup → cosine decay to `min_lr`
- **Mixed precision**: bfloat16 autocast + TF32 enabled
- **Dual optimizer**: `DualOptimizer` routes parameters to AdamW or Muon based on shape/type
- **Checkpointing**: Saves `model`, `optimizer`, `dataloader`, `step` — resumes automatically if checkpoint exists
- **Logging**: Every `log_interval` steps logs train loss, LR, gradient norm, token throughput to wandb
- **Evaluation**: Every `eval_interval` steps runs `estimate_loss()` and `evaluate_hella_swag()`

### Data: `src/datasets/dataloader.py`

- Loads pre-tokenized `.npy` shard files from `data/edu_fineweb350B/`
- Memory-mapped for efficiency; streams `(input, target)` pairs of length `T`
- DDP-aware: each rank loads from offset `rank * B`, strides by `world_size * B`
- Resumes mid-shard from saved `(shard_idx, position)` state

### Evaluation: `src/eval/metrics.py`

- `estimate_loss(model, loader, eval_iters)`: Averages loss over `eval_iters` batches
- `evaluate_hella_swag(model, device)`: Runs all 10,042 validation examples, picks completion with lowest per-token cross-entropy loss, returns accuracy

### Optimizers: `src/utils/optimizers.py`

- `Muon`: Implements Shampoo-style update using Newton-Schulz5 polynomial to approximate matrix square root. Applied to 2D weight matrices (attention projections, MLP weights).
- `DualOptimizer`: Combines Muon (for 2D params) + AdamW (for 1D params: biases, norms, embeddings). Exposes a unified `step()` / `zero_grad()` interface.
- `zeropower_via_newtonschulz5()`: Core numerical routine — 5 iterations of `a, b, c = 3.4445, -4.7750, 2.0315` coefficients

---

## Configuration System

Uses [Hydra](https://hydra.cc/) with YAML configs. Base config is `config/config_basemodel.yaml`; experiments override it.

### Base Config Key Parameters

| Section | Key | Description |
|---|---|---|
| `model` | `n_embd`, `n_layer`, `n_head`, `vocab_size` | Model architecture |
| `model` | `use_split` | Use `GPT_split` instead of `GPT` |
| `train` | `max_steps`, `batch_size`, `seq_len` | Training loop |
| `train` | `warmup_steps`, `min_lr` | LR schedule |
| `train` | `use_muon`, `muon_lr`, `muon_lr_scale` | Muon optimizer controls |
| `train` | `eval_interval`, `log_interval` | Logging/eval frequency |
| `data` | `train_data_dir`, `val_data_dir` | Shard file locations |

### Experiment Configs (in `config/experiments/`)

| File | Variation |
|---|---|
| `exp_0_adamw.yaml` | Baseline: AdamW only (`use_muon: false`) |
| `exp_1_muon_current.yaml` | Muon with default settings |
| `exp_2_muon_no_wd.yaml` | Muon, weight decay disabled |
| `exp_3_muon_lr100.yaml` | Muon with `lr_scale: 100` |
| `exp_4_muon_lr150.yaml` | Muon with `lr_scale: 150` |
| `exp_5_split_qkv.yaml` | Split Q/K/V projections |
| `exp_6_split_all.yaml` | Split QKV + split MLP |
| `exp_7_best.yaml` | Best combo: split model + `split_mlp: true` + `muon_lr_scale: 100` |

---

## Running Training

### Single experiment (local/interactive)

```bash
conda activate LLM
python pretrain.py +experiments=exp_0_adamw
```

### Via SLURM (single job)

```bash
sbatch scripts/baseline_model.sh
```

### All 8 experiments sequentially

```bash
bash scripts/run_experiments.sh
```

### Resuming from checkpoint

Training resumes automatically if `output/<run_name>/checkpoint.pt` exists. No flag needed.

---

## Development Conventions

### Code Style

- **Classes**: PascalCase (`TrainerSingleGPU`, `DualOptimizer`)
- **Functions/methods**: snake_case (`apply_rotary_emb`, `estimate_loss`)
- **Constants**: UPPERCASE (`SPLIT_MLP`)
- **Private methods**: Leading underscore (`_train_global_batch`, `_init_weights`)

### Model Design Patterns

- All model components subclass `nn.Module`
- Pre-LayerNorm architecture (RMSNorm before each sublayer, not after)
- Residual paths scaled by `1/sqrt(2 * n_layers)` at init
- RoPE embeddings precomputed and stored as non-persistent buffers
- Weight tying between `wte` (token embedding) and `lm_head` (output projection)

### Configuration Pattern

- Never hardcode hyperparameters in source — everything goes through Hydra config
- `TrainerConfig` is a dataclass populated from `cfg.train.*`
- Experiment files only contain fields that differ from the base config

### Checkpointing Pattern

Checkpoints are saved as `output/<run_name>/checkpoint.pt` containing:
```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "dataloader": dataloader.state_dict(),
    "step": current_step,
}
```

---

## Key Dependencies

No `requirements.txt` exists. The expected conda environment (`LLM`) includes:

| Package | Purpose |
|---|---|
| `torch` | Core ML framework (bfloat16, compile, DDP) |
| `hydra-core` | Config management |
| `wandb` | Experiment tracking |
| `tiktoken` | GPT-2 tokenizer |
| `numpy` | Memory-mapped data shards |
| `transformers` | Reference GPT-2 for validation |
| `humanize` | Human-readable numbers in logs |

---

## Data

- Training data: `data/edu_fineweb350B/` — pre-tokenized `.npy` shards (symlinked, not committed)
- Eval data: HellaSwag downloaded by `scripts/data_prep/hellaswag.py` to `data/hellaswag/`
- The `data/` directory is in `.gitignore`

---

## What NOT to Do

- Do not add `requirements.txt` or package boilerplate unless asked — this is a research script, not a library
- Do not introduce test frameworks (pytest, unittest) — evaluation is done through training metrics
- Do not refactor working experiment configs — each `exp_N_*.yaml` is an intentional controlled variation
- Do not add Docker/containerization unless asked — the project targets a specific HPC SLURM environment
- Do not commit `data/`, `output/`, `wandb/`, or `__pycache__/` — all are in `.gitignore`
- Do not change the Muon optimizer coefficients (`3.4445, -4.7750, 2.0315`) — these are numerically calibrated
