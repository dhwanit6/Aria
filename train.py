"""
Aria Training Loop.

Features:
- Mixed-precision BF16 training with ternary STE
- Cosine LR schedule with warmup
- Gradient accumulation for effective large batch size
- Gradient checkpointing for memory efficiency
- Checkpoint save/resume (survives Colab disconnects)
- Wandb logging (optional)
- Multi-task loss: LM + draft head + MoE balance
"""
import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model.config import AriaConfig, aria_debug, aria_tiny, aria_small, aria_full
from model.aria import Aria, build_aria


# ──── Dataset ────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Memory-mapped token dataset.
    
    Reads pre-tokenized binary files (uint16 token IDs).
    Each sample is a contiguous chunk of `seq_len + 1` tokens:
        input = tokens[0:seq_len]
        target = tokens[1:seq_len+1]
    """

    def __init__(self, data_path: str, seq_len: int = 2048):
        self.seq_len = seq_len
        data_path = Path(data_path)

        if data_path.is_file():
            # Single binary file
            self.data = torch.from_numpy(
                __import__("numpy").memmap(str(data_path), dtype="uint16", mode="r")
            ).long()
        elif data_path.is_dir():
            # Directory of shard files — concatenate
            shards = sorted(data_path.glob("*.bin"))
            if not shards:
                raise FileNotFoundError(f"No .bin files found in {data_path}")
            import numpy as np
            arrays = [np.memmap(str(s), dtype="uint16", mode="r") for s in shards]
            combined = np.concatenate(arrays)
            self.data = torch.from_numpy(combined.copy()).long()
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // self.seq_len
        print(f"Dataset: {self.n_tokens:,} tokens → {self.n_samples:,} samples "
              f"(seq_len={seq_len})")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]     # target tokens (shifted by 1)
        return x, y


class SyntheticDataset(Dataset):
    """Synthetic random dataset for testing/debugging. No disk I/O needed."""

    def __init__(self, n_samples: int = 1000, seq_len: int = 128, vocab_size: int = 256):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic random based on idx for reproducibility
        gen = torch.Generator().manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
        return tokens[:-1], tokens[1:]


# ──── Learning Rate Schedule ─────────────────────────────────

def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * step / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──── Training Loop ──────────────────────────────────────────

def train(
    config_name: str = "debug",
    data_path: str | None = None,
    output_dir: str = "checkpoints",
    resume_from: str | None = None,
    # Training hyperparameters
    batch_size: int = 4,
    gradient_accumulation: int = 8,
    peak_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 1000,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    # System
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "aria",
    log_interval: int = 10,
    save_interval: int = 1000,
    eval_interval: int = 500,
    gradient_checkpointing: bool = True,
):
    """Main training function."""

    # ──── Setup ────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ──── Config ────
    configs = {"debug": aria_debug, "tiny": aria_tiny, "small": aria_small, "full": aria_full}
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Options: {list(configs.keys())}")
    config = configs[config_name]()

    # ──── Dataset ────
    if data_path is None:
        print("No data_path provided — using synthetic data for testing")
        dataset = SyntheticDataset(
            n_samples=max(1000, batch_size * gradient_accumulation * 100),
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
        )
    else:
        dataset = TokenDataset(data_path, seq_len=config.max_seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ──── Model ────
    model = build_aria(config)
    model = model.to(device)

    if gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing: enabled")

    # ──── Optimizer ────
    # Separate parameters: decay (weights) vs no-decay (norms, biases, RWKV params)
    decay_params = []
    no_decay_params = []
    seen_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue  # Skip shared attention params (already counted)
        seen_ids.add(pid)

        if param.ndim < 2 or "norm" in name or "decay" in name or "eta" in name or "time_mix" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    total_params = sum(p.numel() for p in decay_params) + sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {len(decay_params)} decay groups ({sum(p.numel() for p in decay_params)/1e6:.1f}M), "
          f"{len(no_decay_params)} no-decay groups ({sum(p.numel() for p in no_decay_params)/1e6:.1f}M)")

    # ──── Compute total steps ────
    tokens_per_step = batch_size * gradient_accumulation * config.max_seq_len
    if max_tokens is not None:
        total_steps = max_tokens // tokens_per_step
    elif max_steps is not None:
        total_steps = max_steps
    else:
        total_steps = len(dataloader) // gradient_accumulation
    print(f"Training for {total_steps:,} steps ({total_steps * tokens_per_step / 1e9:.2f}B tokens)")
    print(f"Tokens per step: {tokens_per_step:,} (batch={batch_size} × accum={gradient_accumulation} × seq={config.max_seq_len})")

    # ──── Resume ────
    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step} ({resume_from})")

    # ──── Wandb ────
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config={
            "model": config_name,
            "total_params_M": total_params / 1e6,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "peak_lr": peak_lr,
            "total_steps": total_steps,
            "tokens_per_step": tokens_per_step,
        })

    # ──── Mixed precision context ────
    amp_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = None  # BF16 doesn't need gradient scaling

    # ──── Save config ────
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ──── Training loop ────
    model.train()
    optimizer.zero_grad()

    data_iter = iter(dataloader)
    tokens_seen = start_step * tokens_per_step
    best_loss = float("inf")

    t_start = time.time()
    running_loss = 0.0
    running_lm_loss = 0.0

    for step in range(start_step, total_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, total_steps, peak_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        step_lm_loss = 0.0

        for micro_step in range(gradient_accumulation):
            # Get batch (restart dataloader if exhausted)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward
            with amp_context:
                output = model(x, target_ids=y)
                loss = output.loss / gradient_accumulation

            # Backward
            loss.backward()

            step_loss += loss.item()
            if output.aux_loss is not None:
                step_lm_loss += (output.loss.item() - output.aux_loss.item() -
                                 (output.draft_loss.item() if output.draft_loss else 0))/ gradient_accumulation
            else:
                step_lm_loss += output.loss.item() / gradient_accumulation

        # Gradient clipping
        if grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        tokens_seen += tokens_per_step
        running_loss += step_loss
        running_lm_loss += step_lm_loss

        # ──── Logging ────
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = running_loss / log_interval
            avg_lm_loss = running_lm_loss / log_interval
            tok_per_sec = tokens_per_step * log_interval / elapsed
            ppl = math.exp(min(avg_lm_loss, 20))  # cap to avoid overflow

            print(
                f"step {step+1:>6d}/{total_steps} | "
                f"loss {avg_loss:.4f} | lm_loss {avg_lm_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"tok/s {tok_per_sec:.0f} | "
                f"tokens {tokens_seen/1e9:.3f}B"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lm_loss": avg_lm_loss,
                    "perplexity": ppl,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tok_per_sec,
                    "tokens_seen_B": tokens_seen / 1e9,
                    "step": step + 1,
                })

            running_loss = 0.0
            running_lm_loss = 0.0
            t_start = time.time()

        # ──── Save checkpoint ────
        if (step + 1) % save_interval == 0 or step == total_steps - 1:
            ckpt_path = output_dir / f"step_{step+1:07d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config_dict,
                "tokens_seen": tokens_seen,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

            # Also save as "latest" for easy resume
            latest_path = output_dir / "latest.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config_dict,
                "tokens_seen": tokens_seen,
            }, latest_path)

    print(f"\nTraining complete. {tokens_seen/1e9:.2f}B tokens processed.")
    if use_wandb:
        import wandb
        wandb.finish()


# ──── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aria model")
    parser.add_argument("--config", type=str, default="debug",
                        choices=["debug", "tiny", "small", "full"])
    parser.add_argument("--data", type=str, default=None,
                        help="Path to tokenized data (.bin file or directory of shards)")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=str, default=None,
                        help="Max tokens to train on, e.g. '5B' or '5000000000'")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--no-grad-ckpt", action="store_true")

    args = parser.parse_args()

    # Parse max_tokens shorthand (e.g., "5B", "500M")
    max_tokens = None
    if args.max_tokens:
        s = args.max_tokens.upper().strip()
        if s.endswith("B"):
            max_tokens = int(float(s[:-1]) * 1e9)
        elif s.endswith("M"):
            max_tokens = int(float(s[:-1]) * 1e6)
        else:
            max_tokens = int(s)

    train(
        config_name=args.config,
        data_path=args.data,
        output_dir=args.output,
        resume_from=args.resume,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup,
        max_steps=args.max_steps,
        max_tokens=max_tokens,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        use_wandb=args.wandb,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        gradient_checkpointing=not args.no_grad_ckpt,
    )
