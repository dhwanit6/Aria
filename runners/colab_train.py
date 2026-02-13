"""
Colab/GPU training for Aria-Hindi.

One-shot setup script — run this in a Colab notebook or on any GPU machine.
Handles: environment setup, data upload, tokenizer, preprocessing, training.

Usage (Colab):
    !git clone <your-repo> Aria
    %cd Aria/train
    !python colab_train.py

Usage (local GPU):
    python colab_train.py --data_dir data/processed --skip_setup
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_environment():
    """Install dependencies and verify GPU."""
    print("=" * 60)
    print("ARIA HINDI — TRAINING SETUP")
    print("=" * 60)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Determine optimal batch settings based on GPU
        if gpu_mem >= 70:  # A100/H100 80GB
            batch_size, grad_accum, seq_len = 8, 4, 2048
        elif gpu_mem >= 35:  # A100 40GB
            batch_size, grad_accum, seq_len = 4, 8, 2048
        elif gpu_mem >= 14:  # T4 16GB
            batch_size, grad_accum, seq_len = 2, 16, 1024
        else:
            batch_size, grad_accum, seq_len = 1, 32, 512
        print(f"  Recommended: batch={batch_size}, grad_accum={grad_accum}, seq_len={seq_len}")
        return batch_size, grad_accum, seq_len
    else:
        print("✗ No GPU detected. Training will be very slow.")
        return 1, 8, 256


def prepare_data(data_dir: str = "data/processed", raw_dir: str = "data/raw"):
    """Download and preprocess data if not already done."""
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir)

    # Check if processed data already exists
    train_bin = data_dir / "train.bin"
    if train_bin.exists():
        meta = json.loads((data_dir / "meta.json").read_text())
        print(f"✓ Processed data found: {meta['train_tokens']:,} tokens")
        return str(train_bin)

    print("\nNo processed data found. Running full pipeline...")

    # Step 1: Download
    print("\n[1/3] Downloading Hindi data...")
    os.system(f"python -m data.download_hindi --output_dir {raw_dir} --max_samples 100000")

    # Step 2: Train tokenizer
    merged = raw_dir / "hindi_merged.txt"
    tokenizer_prefix = "data/tokenizer/aria_hindi"
    tokenizer_model = Path(tokenizer_prefix + ".model")
    if not tokenizer_model.exists():
        print("\n[2/3] Training tokenizer...")
        os.system(f"python -m data.train_tokenizer --input {merged} --vocab_size 32000")
    else:
        print(f"✓ Tokenizer exists: {tokenizer_model}")

    # Step 3: Preprocess
    print("\n[3/3] Preprocessing...")
    os.system(f"python -m data.preprocess --input {merged} --tokenizer {tokenizer_model} --output_dir {data_dir}")

    meta = json.loads((data_dir / "meta.json").read_text())
    print(f"\n✓ Data ready: {meta['train_tokens']:,} tokens")
    return str(train_bin)


def get_vocab_size(data_dir: str = "data/processed") -> int:
    """Read vocab size from processed data metadata."""
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())["vocab_size"]
    # Fallback: read directly from tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load("data/tokenizer/aria_hindi.model")
    return sp.get_piece_size()


def create_config(vocab_size: int, seq_len: int):
    """Create Aria-Tiny config with correct vocab size."""
    from model.config import AriaConfig
    return AriaConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_head=64,
        n_kv_heads=4,
        n_shared_experts=1,
        n_routed_experts=2,
        d_expert=512,
        top_k_experts=1,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        attention_window=min(512, seq_len),
        use_draft_head=False,
        use_ternary=True,
    )


def train(
    data_path: str,
    vocab_size: int,
    batch_size: int = 4,
    grad_accum: int = 8,
    seq_len: int = 2048,
    max_steps: int = 5000,
    peak_lr: float = 3e-4,
    output_dir: str = "checkpoints/hindi_tiny",
    resume: str | None = None,
    use_wandb: bool = False,
):
    """Run training with proper GPU settings."""
    import torch
    import torch.nn as nn
    import time
    import math
    from contextlib import nullcontext
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ──
    config = create_config(vocab_size, seq_len)
    from model.aria import Aria
    model = Aria(config).to(device)
    model.set_gradient_checkpointing(True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.1f}M params")
    print(f"Config: d={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"Vocab: {config.vocab_size}, seq_len={config.max_seq_len}")

    # ── Data ──
    from train import TokenDataset
    dataset = TokenDataset(data_path, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ──
    decay_params = []
    no_decay_params = []
    seen_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if param.ndim < 2 or "norm" in name or "decay" in name or "eta" in name or "time_mix" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    print(f"Optimizer: {sum(p.numel() for p in decay_params)/1e6:.1f}M decay, "
          f"{sum(p.numel() for p in no_decay_params)/1e6:.1f}M no-decay")

    # ── Resume ──
    start_step = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"✓ Resumed from step {start_step}")
    else:
        # Auto-resume from latest
        latest = output_dir / "latest.pt"
        if latest.exists():
            ckpt = torch.load(str(latest), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"✓ Auto-resumed from step {start_step}")

    # ── Wandb ──
    if use_wandb:
        import wandb
        wandb.init(project="aria-hindi", config={
            "params_M": n_params / 1e6,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "seq_len": seq_len,
            "peak_lr": peak_lr,
            "max_steps": max_steps,
            "vocab_size": vocab_size,
        })

    # ── Training ──
    tokens_per_step = batch_size * grad_accum * seq_len
    min_lr = peak_lr / 10
    warmup_steps = min(1000, max_steps // 10)

    print(f"\n{'='*60}")
    print(f"TRAINING START")
    print(f"  Steps: {start_step} → {max_steps}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: ~{max_steps * tokens_per_step / 1e9:.2f}B")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  LR: {peak_lr} → {min_lr}")
    print(f"  Grad accum: {grad_accum} (effective batch = {batch_size * grad_accum})")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    data_iter = iter(dataloader)
    tokens_seen = start_step * tokens_per_step
    running_loss = 0.0
    t_start = time.time()

    for step in range(start_step, max_steps):
        # LR schedule
        if step < warmup_steps:
            lr = peak_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                output = model(x, target_ids=y)
                loss = output.loss / grad_accum

            loss.backward()
            step_loss += loss.item()

        # Clip and step
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        tokens_seen += tokens_per_step
        running_loss += step_loss

        # ── Log ──
        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_start
            avg_loss = running_loss / 10
            ppl = math.exp(min(avg_loss, 20))
            tok_s = tokens_per_step * 10 / elapsed

            print(
                f"step {step+1:>5d}/{max_steps} │ "
                f"loss {avg_loss:.4f} │ ppl {ppl:>8.1f} │ "
                f"lr {lr:.2e} │ gnorm {grad_norm:.2f} │ "
                f"{tok_s:,.0f} tok/s │ {tokens_seen/1e6:.1f}M tok"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss, "perplexity": ppl, "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tok_s, "tokens_M": tokens_seen / 1e6,
                }, step=step + 1)

            running_loss = 0.0
            t_start = time.time()

        # ── Save ──
        if (step + 1) % 500 == 0 or step == max_steps - 1:
            ckpt_path = output_dir / f"step_{step+1:06d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {k: v for k, v in config.__dict__.items()},
                "tokens_seen": tokens_seen,
            }, ckpt_path)
            print(f"  → Saved: {ckpt_path}")

            # Always save latest for auto-resume
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {k: v for k, v in config.__dict__.items()},
                "tokens_seen": tokens_seen,
            }, output_dir / "latest.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  {tokens_seen/1e6:.1f}M tokens processed")
    print(f"  Checkpoints in: {output_dir}")
    print(f"{'='*60}")

    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Aria Hindi Training (Colab/GPU)")
    parser.add_argument("--skip_setup", action="store_true", help="Skip data download/preprocessing")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints/hindi_tiny")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="Override auto-detected batch_size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override auto-detected grad_accum")
    parser.add_argument("--seq_len", type=int, default=None, help="Override auto-detected seq_len")
    args = parser.parse_args()

    # Setup
    auto_bs, auto_ga, auto_sl = setup_environment()
    batch_size = args.batch_size or auto_bs
    grad_accum = args.grad_accum or auto_ga
    seq_len = args.seq_len or auto_sl

    # Data
    if not args.skip_setup:
        data_path = prepare_data(args.data_dir)
    else:
        data_path = str(Path(args.data_dir) / "train.bin")

    vocab_size = get_vocab_size(args.data_dir)

    # Train
    train(
        data_path=data_path,
        vocab_size=vocab_size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        max_steps=args.max_steps,
        peak_lr=args.lr,
        output_dir=args.output_dir,
        resume=args.resume,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
