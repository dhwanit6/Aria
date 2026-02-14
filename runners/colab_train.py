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
import subprocess
from pathlib import Path

# Make notebook/script output visible immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))


def _script_paths() -> tuple[Path, Path, Path]:
    """Return (script_dir, train_dir, repo_root)."""
    script_dir = Path(__file__).resolve().parent
    train_dir = script_dir.parent
    repo_root = train_dir.parent
    return script_dir, train_dir, repo_root


def _resolve_path(path_str: str) -> Path:
    """
    Resolve an absolute/relative path against common roots.
    Returns a best-effort path even if it does not exist yet.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    _, train_dir, repo_root = _script_paths()
    candidates = [
        Path.cwd() / p,
        train_dir / p,
        repo_root / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _find_tokenizer_model(data_dir: Path) -> Path | None:
    """Find aria_hindi.model across likely locations."""
    _, train_dir, repo_root = _script_paths()
    candidates = [
        data_dir.parent / "tokenizer" / "aria_hindi.model",
        data_dir / "tokenizer" / "aria_hindi.model",
        train_dir / "data" / "tokenizer" / "aria_hindi.model",
        repo_root / "data" / "tokenizer" / "aria_hindi.model",
        Path.cwd() / "data" / "tokenizer" / "aria_hindi.model",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _run_module(module: str, args: list[str], cwd: Path) -> None:
    """Run `python -m module ...` and raise on failure."""
    cmd = [sys.executable, "-m", module] + args
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


class TokenDataset:
    """Memory-mapped token dataset. Returns (input_ids, target_ids)."""

    def __init__(self, data_path: str, seq_len: int = 2048):
        import numpy as np
        import torch

        self.seq_len = seq_len
        self.data = torch.from_numpy(
            np.memmap(data_path, dtype="uint16", mode="r").copy()
        ).long()
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // self.seq_len
        print(
            f"Dataset: {self.n_tokens:,} tokens -> "
            f"{self.n_samples:,} samples (seq_len={seq_len})"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def setup_environment(allow_cpu: bool = False):
    """Verify runtime type and suggest stable defaults."""
    print("=" * 60)
    print("ARIA HINDI - TRAINING SETUP")
    print("=" * 60)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")

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
    has_xla = False
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        has_xla = True
    except ImportError:
        has_xla = False

    if has_xla:
        msg = (
            "TPU runtime detected. This script is GPU-only; use "
            "`python runners/tpu_train.py` instead."
        )
    else:
        msg = "No CUDA GPU detected."

    if not allow_cpu:
        raise RuntimeError(msg + " Pass --allow_cpu to run this script on CPU.")

    print(f"[WARN] {msg} Running on CPU because --allow_cpu was set.")
    return 1, 8, 256


def prepare_data(data_dir: str = "data/processed", raw_dir: str = "data/raw"):
    """Download and preprocess data if not already done."""
    _, train_dir, _ = _script_paths()
    data_dir = _resolve_path(data_dir)
    raw_dir = _resolve_path(raw_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if processed data already exists
    train_bin = data_dir / "train.bin"
    meta_path = data_dir / "meta.json"
    if train_bin.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[OK] Processed data found: {meta.get('train_tokens', 0):,} tokens")
        return str(train_bin)
    if train_bin.exists():
        print("[WARN] train.bin exists but meta.json is missing; will rebuild metadata/data.")
        return str(train_bin)

    print("\nNo processed data found. Running full pipeline...")

    merged = raw_dir / "hindi_merged.txt"
    tokenizer_dir = train_dir / "data" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_model = _find_tokenizer_model(data_dir)
    if tokenizer_model is None:
        tokenizer_model = tokenizer_dir / "aria_hindi.model"

    # Step 1: Download (only if needed)
    if not merged.exists():
        print("\n[1/3] Downloading Hindi data...")
        _run_module(
            "data.download_hindi",
            ["--output_dir", str(raw_dir), "--max_samples", "100000"],
            cwd=train_dir,
        )
    else:
        print(f"\n[1/3] Raw corpus exists: {merged}")

    # Step 2: Train tokenizer
    if not tokenizer_model.exists():
        print("\n[2/3] Training tokenizer...")
        tokenizer_prefix = str((tokenizer_model.parent / "aria_hindi").resolve())
        _run_module(
            "data.train_tokenizer",
            ["--input", str(merged), "--vocab_size", "32000", "--output_prefix", tokenizer_prefix],
            cwd=train_dir,
        )
    else:
        print(f"[OK] Tokenizer exists: {tokenizer_model}")

    # Step 3: Preprocess
    print("\n[3/3] Preprocessing...")
    _run_module(
        "data.preprocess",
        ["--input", str(merged), "--tokenizer", str(tokenizer_model), "--output_dir", str(data_dir)],
        cwd=train_dir,
    )

    if not train_bin.exists():
        raise RuntimeError(f"Preprocess completed but missing {train_bin}")
    if not meta_path.exists():
        raise RuntimeError(f"Preprocess completed but missing {meta_path}")

    meta = json.loads(meta_path.read_text())
    print(f"\n[OK] Data ready: {meta['train_tokens']:,} tokens")
    return str(train_bin)


def get_vocab_size(data_dir: str = "data/processed") -> int:
    """Read vocab size from processed data metadata."""
    data_dir_p = _resolve_path(data_dir)
    meta_path = data_dir_p / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "vocab_size" in meta:
            return int(meta["vocab_size"])
    # Fallback: read directly from tokenizer
    import sentencepiece as spm
    tokenizer_path = _find_tokenizer_model(data_dir_p)
    if tokenizer_path is None:
        raise FileNotFoundError(
            f"No vocab metadata ({meta_path}) and no tokenizer found for data_dir={data_dir_p}"
        )
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
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
    log_interval: int = 10,
    log_first_n_steps: int = 10,
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
    model.set_gradient_checkpointing(device.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.1f}M params")
    print(f"Config: d={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"Vocab: {config.vocab_size}, seq_len={config.max_seq_len}")

    # ── Data ──
    dataset = TokenDataset(data_path, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset has no samples for seq_len={seq_len}. "
            "Use a smaller --seq_len or provide more tokens."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
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
        print(f"[OK] Resumed from step {start_step}")
    else:
        # Auto-resume from latest
        latest = output_dir / "latest.pt"
        if latest.exists():
            ckpt = torch.load(str(latest), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"[OK] Auto-resumed from step {start_step}")

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
    print(f"  Steps: {start_step} -> {max_steps}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: ~{max_steps * tokens_per_step / 1e9:.2f}B")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  LR: {peak_lr} -> {min_lr}")
    print(f"  Grad accum: {grad_accum} (effective batch = {batch_size * grad_accum})")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    data_iter = iter(dataloader)
    tokens_seen = start_step * tokens_per_step
    running_loss = 0.0
    steps_since_log = 0
    t_start = time.time()

    for step in range(start_step, max_steps):
        # LR schedule
        if step < warmup_steps:
            lr = peak_lr * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
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
        steps_since_log += 1
        should_log = (
            (step - start_step + 1) <= log_first_n_steps
            or (step + 1) % max(log_interval, 1) == 0
        )
        if should_log:
            elapsed = time.time() - t_start
            avg_loss = running_loss / max(steps_since_log, 1)
            ppl = math.exp(min(avg_loss, 20))
            tok_s = tokens_per_step * steps_since_log / max(elapsed, 0.01)

            print(
                f"step {step+1:>5d}/{max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:>8.1f} | "
                f"lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                f"{tok_s:,.0f} tok/s | {tokens_seen/1e6:.1f}M tok"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss, "perplexity": ppl, "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tok_s, "tokens_M": tokens_seen / 1e6,
                }, step=step + 1)

            running_loss = 0.0
            steps_since_log = 0
            t_start = time.time()

        # ── Save ──
        if (step + 1) % 500 == 0 or step == max_steps - 1:
            ckpt_path = output_dir / f"step_{step+1:06d}.pt"
            try:
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {k: v for k, v in config.__dict__.items()},
                    "tokens_seen": tokens_seen,
                }, ckpt_path)
                print(f"  -> Saved: {ckpt_path}")

                # Always save latest for auto-resume
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {k: v for k, v in config.__dict__.items()},
                    "tokens_seen": tokens_seen,
                }, output_dir / "latest.pt")
            except Exception as exc:
                print(f"[WARN] Checkpoint save failed at step {step+1}: {exc}")

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
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Allow CPU training when no CUDA GPU is available.")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps after warmup logging.")
    parser.add_argument("--log_first_n_steps", type=int, default=10,
                        help="Always log the first N training steps.")
    args = parser.parse_args()

    # Setup
    try:
        auto_bs, auto_ga, auto_sl = setup_environment(allow_cpu=args.allow_cpu)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    batch_size = args.batch_size or auto_bs
    grad_accum = args.grad_accum or auto_ga
    seq_len = args.seq_len or auto_sl

    # Data
    data_dir = _resolve_path(args.data_dir)
    train_bin = data_dir / "train.bin"

    if args.skip_setup and train_bin.exists():
        data_path = str(train_bin)
        print(f"[OK] Using existing data: {data_path}")
    else:
        if args.skip_setup and not train_bin.exists():
            print(
                f"[WARN] --skip_setup was set, but {train_bin} is missing. "
                "Running data setup automatically."
            )
        data_path = prepare_data(str(data_dir))

    try:
        vocab_size = get_vocab_size(str(data_dir))
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        print("[INFO] Re-running data setup to build tokenizer/meta...")
        data_path = prepare_data(str(data_dir))
        vocab_size = get_vocab_size(str(Path(data_path).parent))

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
        log_interval=args.log_interval,
        log_first_n_steps=args.log_first_n_steps,
    )


if __name__ == "__main__":
    main()
