import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
import sys
from scripts.data_prep.hellaswag import iterate_examples, render_example


@torch.no_grad()
def estimate_loss(model, loader, eval_steps, device):
    """
    Estimates the loss on the validation set.
    """
    model.eval()
    losses = torch.zeros(eval_steps, device=device)

    # Iterate eval_steps times from the loader
    for k in range(eval_steps):
        # The loader is infinite, so next() always returns a batch
        try:
            X, Y = next(loader)
        except StopIteration:
            loader.reset()
            X, Y = next(loader)

        X = X.to(device)
        Y = Y.to(device)

        # Use autocast if using cuda/bfloat16
        with torch.autocast(
            device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16
        ):
            # Model forward pass
            _, loss = model(X, Y)

        losses[k] = loss.item()

    out = losses.mean()
    model.train()
    return out.item()


@torch.no_grad()
def evaluate_hella_swag(model, device):
    """
    Evaluates HellaSwag accuracy using the model.
    """
    model.eval()

    # Check strict ddp info
    ddp = dist.is_initialized()
    if ddp:
        ddp_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
    else:
        ddp_rank = 0
        ddp_world_size = 1

    num_correct_norm = 0
    num_total = 0

    total_examples = 10042
    cutoff = (total_examples // ddp_world_size) * ddp_world_size

    # Iterate over validation examples
    # We rely on iterate_examples("val") which reads the jsonl
    for i, example in enumerate(iterate_examples("val")):
        if i >= cutoff:
            break

        # Only process examples belonging to this rank
        if i % ddp_world_size != ddp_rank:
            continue

        # Render example into tokens
        # Returns: data_dict, tokens(4, N), mask(4, N), label(int)
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Forward pass
        with torch.autocast(
            device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16
        ):
            logits, _ = model(tokens)

        # Logits shape: (4, Seq_Len, Vocab)
        # We need to evaluate autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()

        # Calculate loss per token
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Mask out context, only evaluate completion
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # Sum loss per candidate and normalize by length
        sum_loss = masked_shift_losses.sum(dim=1)
        # Prevent division by zero if mask is empty (unlikely)
        div = shift_mask.sum(dim=1)
        div[div == 0] = 1.0
        avg_loss = sum_loss / div

        # Prediction: Lowest loss is the most likely completion
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # Aggregate stats across GPUs
    if ddp:
        stats = torch.tensor(
            [num_total, num_correct_norm], dtype=torch.long, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        num_total = stats[0].item()
        num_correct_norm = stats[1].item()

    acc = num_correct_norm / num_total if num_total > 0 else 0.0

    model.train()
    return acc
