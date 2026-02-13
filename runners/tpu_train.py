"""
Aria Hindi Training — TPU v5e1 (Google Colab).

Uses PyTorch/XLA for TPU support. Run in Colab with TPU runtime:
    1. Runtime → Change runtime type → TPU v5e1
    2. !pip install torch torch_xla sentencepiece datasets tqdm wandb
    3. !python tpu_train.py --data_dir data/processed --max_steps 5000

Key XLA differences from CUDA:
    - Lazy execution: ops are traced, not executed immediately
    - xm.mark_step() triggers execution of the traced graph
    - xm.optimizer_step() wraps optimizer.step() + mark_step()
    - No mixed-precision needed: TPU v5e natively runs BF16 matmuls
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

# ── XLA imports ──
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    print("WARNING: torch_xla not found. Falling back to CPU/CUDA.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import AriaConfig
from model.aria import Aria


# ──────────────────────────────────────────────────────────────
# Dataset (inlined to avoid import issues on Colab)
# ──────────────────────────────────────────────────────────────

class TokenDataset(torch.utils.data.Dataset):
    """Memory-mapped token dataset. Returns (input_ids, target_ids)."""

    def __init__(self, data_path: str, seq_len: int = 2048):
        import numpy as np
        self.seq_len = seq_len
        self.data = torch.from_numpy(
            np.memmap(data_path, dtype="uint16", mode="r").copy()
        ).long()
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // self.seq_len
        print(f"Dataset: {self.n_tokens:,} tokens → {self.n_samples:,} samples (seq_len={seq_len})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

def make_config(vocab_size: int, seq_len: int) -> AriaConfig:
    """Aria-Tiny config (~95M params) with correct vocab."""
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


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

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
    # ── Device ──
    if HAS_XLA:
        device = xm.xla_device()
        print(f"✓ TPU device: {device}")
        print(f"  XLA version: {torch_xla.__version__}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("✗ CPU only (very slow)")

    is_tpu = HAS_XLA and "xla" in str(device)

    # ── Model ──
    config = make_config(vocab_size, seq_len)
    model = Aria(config)

    # Cast model to BF16 for TPU (TPU does BF16 matmuls natively)
    if is_tpu:
        model = model.to(torch.bfloat16)

    model = model.to(device)
    model.set_gradient_checkpointing(True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.1f}M params")
    print(f"Config: d={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"Ternary: {config.use_ternary}")

    # ── Data ──
    dataset = TokenDataset(data_path, seq_len=seq_len)

    # XLA-compatible dataloader (no pin_memory, no persistent_workers)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # ── Optimizer ──
    decay_params, no_decay_params = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or any(k in name for k in ("norm", "decay", "eta", "time_mix")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    # No fused=True on TPU
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)

    print(f"Optimizer: {sum(p.numel() for p in decay_params)/1e6:.1f}M decay, "
          f"{sum(p.numel() for p in no_decay_params)/1e6:.1f}M no-decay")

    # ── Resume ──
    start_step = 0
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = out_dir / "latest.pt"
    resume_path = resume or (str(latest) if latest.exists() else None)
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"✓ Resumed from step {start_step}")
        # Re-move model to device after loading
        model = model.to(device)

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
            "device": "tpu" if is_tpu else str(device),
        })

    # ── LR Schedule ──
    min_lr = peak_lr / 10
    warmup_steps = min(500, max_steps // 10)
    tokens_per_step = batch_size * grad_accum * seq_len

    def get_lr(step):
        if step < warmup_steps:
            return peak_lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # ── AMP context (only for CUDA, TPU does BF16 natively) ──
    if not is_tpu and device.type == "cuda":
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"TRAINING START ({'TPU' if is_tpu else device})")
    print(f"  Steps: {start_step} → {max_steps}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: ~{max_steps * tokens_per_step / 1e9:.2f}B")
    print(f"  Warmup: {warmup_steps} steps, LR: {peak_lr} → {min_lr}")
    print(f"  Grad accum: {grad_accum} (effective batch = {batch_size * grad_accum})")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()
    data_iter = iter(dataloader)
    tokens_seen = start_step * tokens_per_step
    running_loss = 0.0
    t_start = time.time()

    for step in range(start_step, max_steps):
        # LR
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        step_loss = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            with amp_ctx:
                output = model(x, target_ids=y)
                loss = output.loss / grad_accum

            loss.backward()
            step_loss += loss.item()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step (XLA-aware)
        if is_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        optimizer.zero_grad()

        tokens_seen += tokens_per_step
        running_loss += step_loss

        # ── Log every 10 steps ──
        if (step + 1) % 10 == 0:
            # On TPU, .item() triggers execution — use it sparingly
            if is_tpu:
                xm.mark_step()

            elapsed = time.time() - t_start
            avg_loss = running_loss / 10
            ppl = math.exp(min(avg_loss, 20))
            tok_s = tokens_per_step * 10 / max(elapsed, 0.01)

            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            print(
                f"step {step+1:>5d}/{max_steps} │ "
                f"loss {avg_loss:.4f} │ ppl {ppl:>8.1f} │ "
                f"lr {lr:.2e} │ gnorm {grad_norm_val:.2f} │ "
                f"{tok_s:,.0f} tok/s │ {tokens_seen/1e6:.1f}M tok"
            )

            if use_wandb:
                import wandb
                wandb.log({
                    "loss": avg_loss, "perplexity": ppl, "lr": lr,
                    "grad_norm": grad_norm_val,
                    "tokens_per_sec": tok_s,
                    "tokens_M": tokens_seen / 1e6,
                }, step=step + 1)

            running_loss = 0.0
            t_start = time.time()

        # ── Save every 500 steps ──
        if (step + 1) % 500 == 0 or step == max_steps - 1:
            # On TPU, move model to CPU for saving to avoid XLA serialization issues
            if is_tpu:
                xm.mark_step()
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                cpu_state = model.state_dict()

            ckpt = {
                "step": step,
                "model": cpu_state,
                "optimizer": optimizer.state_dict(),
                "config": {k: v for k, v in config.__dict__.items()},
                "tokens_seen": tokens_seen,
            }

            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, out_dir / "latest.pt")
            print(f"  → Saved: {ckpt_path}")

            # Backup to Drive if mounted
            drive_dir = Path("/content/drive/MyDrive/Aria/checkpoints")
            if drive_dir.parent.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, drive_dir / "latest.pt")
                print(f"  → Backed up to Google Drive")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {tokens_seen/1e6:.1f}M tokens processed")
    print(f"{'='*60}")

    if use_wandb:
        import wandb
        wandb.finish()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def find_file(name: str, search_dirs: list[str]) -> str | None:
    """Search for a file across multiple directories."""
    for d in search_dirs:
        p = Path(d) / name
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Aria Hindi — TPU Training")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dir containing train.bin and meta.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/hindi_tiny")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # ── Robust path resolution ──
    # Script can be run from /content/Aria, /content/Aria/train, or anywhere
    script_dir = Path(__file__).parent.resolve()        # runners/
    train_dir = script_dir.parent.resolve()              # train/
    repo_root = train_dir.parent.resolve()               # Aria/

    search_dirs = [
        str(Path(args.data_dir)) if args.data_dir else "",
        str(train_dir / "data" / "processed"),           # train/data/processed/
        str(repo_root / "data" / "processed"),            # Aria/data/processed/
        "data/processed",                                 # relative to CWD
        str(Path.cwd() / "data" / "processed"),
    ]

    # Find train.bin
    data_path = find_file("train.bin", search_dirs)
    if data_path is None:
        print("Data not found. Auto-downloading Hindi corpus...")
        # Run full pipeline from train/ directory
        os.chdir(str(train_dir))
        print(f"  Working dir: {os.getcwd()}")

        # Step 1: Download
        print("\n[1/3] Downloading Hindi Wikipedia...")
        ret = os.system("python -m data.download_hindi --output_dir data/raw --max_samples 100000")
        if ret != 0:
            print("ERROR: Download failed.")
            sys.exit(1)

        # Step 2: Train tokenizer
        tok_model = train_dir / "data" / "tokenizer" / "aria_hindi.model"
        if not tok_model.exists():
            print("\n[2/3] Training tokenizer...")
            os.system("python -m data.train_tokenizer --input data/raw/hindi_merged.txt --vocab_size 32000")
        else:
            print(f"\n[2/3] Tokenizer exists: {tok_model}")

        # Step 3: Preprocess
        print("\n[3/3] Preprocessing to binary...")
        os.system(f"python -m data.preprocess --input data/raw/hindi_merged.txt --tokenizer {tok_model} --output_dir data/processed")

        # Now find the data
        data_path = str(train_dir / "data" / "processed" / "train.bin")
        if not Path(data_path).exists():
            print(f"ERROR: Pipeline completed but {data_path} not found.")
            sys.exit(1)
        print(f"\n✓ Pipeline complete: {data_path}")

    data_dir = str(Path(data_path).parent)
    print(f"✓ Found data: {data_path}")

    # Find meta.json (same dir as train.bin)
    meta_path = Path(data_dir) / "meta.json"
    if meta_path.exists():
        vocab_size = json.loads(meta_path.read_text())["vocab_size"]
        print(f"✓ Vocab size: {vocab_size} (from {meta_path})")
    else:
        # Try to find tokenizer
        tok_search = [
            str(Path(data_dir).parent / "tokenizer"),    # data/tokenizer/
            str(train_dir / "data" / "tokenizer"),
            str(repo_root / "data" / "tokenizer"),
            "data/tokenizer",
        ]
        tok_path = find_file("aria_hindi.model", tok_search)
        if tok_path is None:
            print("ERROR: No meta.json and no tokenizer found. Cannot determine vocab size.")
            sys.exit(1)
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(tok_path)
        vocab_size = sp.get_piece_size()
        print(f"✓ Vocab size: {vocab_size} (from {tok_path})")

    train(
        data_path=data_path,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        peak_lr=args.lr,
        output_dir=args.output_dir,
        resume=args.resume,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
