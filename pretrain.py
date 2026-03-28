import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast
import os
import wandb
import random
import numpy as np

from src.training.trainer_single_gpu import TrainerSingleGPU, TrainerConfig
from src.utils.helpers import print_trainable_parameters, estimate_flops


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_name="config_basemodel", config_path="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility before any other operations
    seed = cfg.experiment.get("seed", 42)
    set_seed(seed)

    # Enforce strict single GPU execution
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(OmegaConf.to_yaml(cfg))

    # init wandb
    wandb_run = wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.run_name,
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        dir=os.getcwd(),
    )

    # define tokenizer
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt2")

    # speed up
    torch.set_float32_matmul_precision("high")
    if device_obj.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_type = cfg.model.get("model_type", "gpt")
    if model_type == "gpt_split":
        import src.models.gpt_split as gpt_split_module

        gpt_split_module.SPLIT_QKV = cfg.model.get("split_qkv", False)
        gpt_split_module.SPLIT_MLP = cfg.model.get("split_mlp", False)
        GPT = gpt_split_module.GPT
    elif model_type == "gpt_moe":
        from src.models.gpt_moe import GPT
    else:
        from src.models.gpt import GPT

    moe_kwargs = {}
    if model_type == "gpt_moe":
        moe_kwargs = {
            "n_shared_experts": cfg.model.get("n_shared_experts", 2),
            "n_routed_experts": cfg.model.get("n_routed_experts", 64),
            "topk_experts": cfg.model.get("topk_experts", 6),
            "expert_hidden_size": cfg.model.get("expert_hidden_size", 256),
        }

    model = GPT(
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_heads=cfg.model.n_heads,
        head_size=cfg.model.head_size,
        rope_head_size=cfg.model.rope_head_size,
        n_layers=cfg.model.n_layers,
        **moe_kwargs,
    )

    # Move to GPU and optionally compile
    model.to(device=device_obj, dtype=torch.bfloat16)
    model = torch.compile(model)

    print_trainable_parameters(cfg, model)
    estimate_flops(cfg)

    trainer_config = TrainerConfig(
        run_name=cfg.experiment.run_name,
        batch_size=cfg.training.batch_size,
        block_size=cfg.model.block_size,
        grad_accum_steps=cfg.training.grad_accum_steps,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        min_lr=cfg.training.min_lr,
        max_lr=cfg.training.max_lr,
        use_muon=cfg.training.get("use_muon", True),
        muon_backend=cfg.training.get("muon_backend", "custom"),
        muon_lr_scale=cfg.training.get("muon_lr_scale", 30.0),
        muon_wd=cfg.training.get("muon_wd", cfg.training.weight_decay),
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        checkpoint_interval=cfg.training.checkpoint_interval,
        generation_interval=cfg.training.generation_interval,
        eval_interval=cfg.training.eval_interval,
        eval_steps=cfg.training.eval_steps,
        eval_batch_size=cfg.training.eval_batch_size,
        eval_block_size=cfg.training.eval_block_size,
        device=str(device_obj),
    )

    trainer = TrainerSingleGPU(
        model=model,
        wandb_run=wandb_run,
        train_data_root=cfg.data.train_data_root,
        val_data_root=cfg.data.val_data_root,
        config=trainer_config,
        tokenizer=enc,
    )

    # check if training from checkpoint
    resume_path = cfg.get("resume_checkpoint", None)

    # train the model
    trainer.train(resume_from_checkpoint=resume_path)

    wandb.finish()


if __name__ == "__main__":
    main()
