import torch
from dataclasses import dataclass
from src.datasets.dataloader import DataLoader
from src.eval.metrics import estimate_loss, evaluate_hella_swag
import time
import math
import os
import json


@dataclass
class TrainerConfig:
    run_name: str = "baseline_test"
    batch_size: int = 64
    grad_accum_steps: int = 4
    block_size: int = 1024
    max_steps: int = 1000
    warmup_steps: int = 100
    min_lr: float = 1e-4
    max_lr: float = 6e-4
    learning_rate: float = 1e-4
    use_muon: bool = True
    muon_backend: str = "custom"
    muon_lr_scale: float = 30.0
    muon_wd: float = 0.1
    weight_decay: float = 0.1
    logging_steps: int = 1
    checkpoint_interval: int = 1000
    generation_interval: int = 50
    eval_interval: int = 200
    eval_steps: int = 200
    eval_batch_size: int = 64
    eval_block_size: int = 1024
    device: str = "cuda"
    lr_schedule: str = "cosine"
    stable_steps: int = 0


class TrainerSingleGPU:
    def __init__(
        self,
        model,
        train_data_root,
        val_data_root,
        config: TrainerConfig,
        tokenizer=None,
        wandb_run=None,
    ):
        self.config = config
        self.model = model
        self.wandb_run = wandb_run

        # Dataloaders
        self.train_loader = DataLoader(
            train_data_root, config.batch_size, config.block_size, "train"
        )
        self.val_loader = DataLoader(
            val_data_root, config.eval_batch_size, config.eval_block_size, "val"
        )

        self.optimizer = self.model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            self.config.device,
            use_muon=self.config.use_muon,
            muon_wd=self.config.muon_wd,
            muon_backend=self.config.muon_backend,
        )
        self.tokenizer = tokenizer
        self.step = 0

    def get_lr(self, it):
        if it < self.config.warmup_steps:
            return self.config.max_lr * it / self.config.warmup_steps
        if self.config.lr_schedule == "wsd":
            stable_end = self.config.warmup_steps + self.config.stable_steps
            if it <= stable_end:
                return self.config.max_lr
            decay_ratio = min(
                (it - stable_end) / max(self.config.max_steps - stable_end, 1), 1.0
            )
            return self.config.max_lr - decay_ratio * (self.config.max_lr - self.config.min_lr)
        # cosine (default)
        if it > self.config.max_steps:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def _train_global_batch(self):
        self.optimizer.zero_grad()
        loss_accum = 0.0

        for step in range(self.config.grad_accum_steps):
            x, y = next(self.train_loader)
            x, y = x.to(self.config.device), y.to(self.config.device)
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                _, loss = self.model(x, y)

            loss = loss / self.config.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if hasattr(self.optimizer, "get_adamw_params"):
            torch.nn.utils.clip_grad_norm_(self.optimizer.get_adamw_params(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss_accum

    def train(self, resume_from_checkpoint=None):
        self.model.train()

        start_step = 1
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)
            start_step = self.step + 1
            print(
                f"---| Resuming training from step {start_step} until {self.config.max_steps} |---"
            )
            self.train_loader.set_step(self.step, self.config.grad_accum_steps)
        else:
            print(f"---| Starting training for {self.config.max_steps} |---")

        best_val_loss = float("inf")
        for step in range(start_step, self.config.max_steps + 1):
            self.step = step
            t0 = time.time()

            adam_lr = self.get_lr(step)
            muon_lr = adam_lr * self.config.muon_lr_scale
            if hasattr(self.optimizer, "set_lrs"):
                self.optimizer.set_lrs(adam_lr, muon_lr)
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = adam_lr

            loss = self._train_global_batch()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            # tokens per second
            tps = (
                self.train_loader.B * self.train_loader.T * self.config.grad_accum_steps
            ) / dt

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train loss": float(loss),
                        "tokens/sec": float(tps),
                        "train step time (ms)": dt * 1000,
                        "adam lr": float(adam_lr),
                        "muon lr": float(muon_lr),
                    },
                    step=step,
                )

            if step % self.config.logging_steps == 0:
                print(
                    f"Step: {step} | loss: {loss:.6f} | dt: {dt * 1000:.4f} ms | tokens/sec: {tps:.4f} | adam lr: {adam_lr:.6e} | muon lr: {muon_lr:.6e}"
                )
                self._log_expert_loads(step)

            if (
                self.tokenizer is not None
                and step % self.config.generation_interval == 0
            ):
                print(f"\n--- Generating text at step {step} ---", flush=True)
                context = "Once upon a time"
                idx = torch.tensor(
                    self.tokenizer.encode(context),
                    dtype=torch.long,
                    device=self.config.device,
                )
                unwrap_model = (
                    self.model._orig_mod
                    if hasattr(self.model, "_orig_mod")
                    else self.model
                )
                unwrap_model.eval()
                with torch.no_grad():
                    out = unwrap_model.generate(idx, max_tokens=64, num_sequences=3)
                unwrap_model.train()
                for i in range(out.size(0)):

                    gen_tokens = out[i].tolist()
                    # Model has padded vocab size (50304), but tokenizer only knows up to 50256
                    gen_tokens = [t if t < 50257 else 50256 for t in gen_tokens]
                    print(f"Gen {i}: {self.tokenizer.decode(gen_tokens)}", flush=True)

                print("--------------------------------------\n", flush=True)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

            val_loss = None

            if step % self.config.eval_interval == 0:
                val_loss = estimate_loss(
                    self.model,
                    self.val_loader,
                    self.config.eval_steps,
                    self.config.device,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(val_loss, step, is_best=True)

                hella_acc = evaluate_hella_swag(self.model, self.config.device)
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"val loss": val_loss, "HellaSwag accuracy": hella_acc},
                        step=step,
                    )

            if step % self.config.checkpoint_interval == 0:
                if val_loss is None:
                    val_loss = estimate_loss(
                        self.model,
                        self.val_loader,
                        self.config.eval_steps,
                        self.config.device,
                    )
                self.save_checkpoint(val_loss, step, is_best=False)

    def _log_expert_loads(self, step):
        """Write per-layer expert token counts to output/experts/RUN_NAME_expert_load.json."""
        unwrap_model = (
            self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        )
        if not hasattr(unwrap_model, "get_expert_loads"):
            return
        loads = unwrap_model.get_expert_loads()
        if not loads:
            return
        log_path = f"output/experts/{self.config.run_name}_expert_load.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "layers": loads}) + "\n")

    def save_checkpoint(self, val_loss, step, is_best=False):
        """Saves a standard PyTorch dictionary checkpoint"""
        if is_best:
            checkpoint_path = f"output/checkpoints/{self.config.run_name}/best_val.pt"
        else:
            checkpoint_path = (
                f"output/checkpoints/{self.config.run_name}/step_{step}.pt"
            )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(f"---| Saving checkpoint to {checkpoint_path} |---")

        unwrap_model = (
            self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        )

        state_dict = {
            "model_state_dict": unwrap_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
        }

        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Loads a standard PyTorch dictionary checkpoint"""
        print(f"---| Loading checkpoint from {checkpoint_path} |---")

        unwrap_model = (
            self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        unwrap_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
