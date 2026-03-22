import os
import torch
import numpy as np


def load_tokens(filename):
    return np.load(filename, mmap_mode="r")


class DataLoader:
    def __init__(
        self,
        data_root,
        batch_size,
        block_size,
        split,
        rank=0,
        world_size=1,
    ):
        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Start at the offset for this specific GPU
        self.current_position = self.B * self.T * self.rank

    def set_step(self, step, grad_accum_steps):
        total_micro_steps = step * grad_accum_steps
        bytes_to_skip = total_micro_steps * (self.B * self.T * self.world_size)
        offset = bytes_to_skip + (self.rank * self.B * self.T)

        for shard_idx, shard_path in enumerate(self.shards):
            meta = np.load(shard_path, mmap_mode="r")
            shard_len = meta.shape[0]

            if offset < shard_len:
                self.current_shard = shard_idx
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = offset
                print(
                    f"Rank {self.rank}: Resuming at Shard {shard_idx}, Index {self.current_position}"
                )
                return
            else:
                offset -= shard_len
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        B, T = self.B, self.T

        # Check if we need to switch shards
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # Reset position: Rank 0 starts at 0, Rank 1 at B*T, etc.
            self.current_position = B * T * self.rank

        buff_np = np.asarray(
            self.tokens[self.current_position : self.current_position + B * T + 1],
            dtype=np.int32,
        )
        buff = torch.from_numpy(buff_np).to(torch.long)

        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)

        # Advance position by total batch size across all GPUs
        self.current_position += B * T * self.world_size

        return x, y
