"""
Train Aria on Hindi Wikipedia data — first real validation.

Uses the debug config (64-dim, 4 layers) on real tokenized Hindi data
to validate that the architecture converges on a natural language corpus.
"""
import sys
sys.path.insert(0, ".")
import json
import torch
import torch.nn as nn
from pathlib import Path
from model.config import AriaConfig
from model.aria import Aria
from data.dataset import BinaryTokenDataset
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
TOKENIZER_DIR = Path("data/tokenizer")
CHECKPOINT_DIR = Path("checkpoints/hindi_debug")

# Load metadata
meta = json.loads((DATA_DIR / "meta.json").read_text())
VOCAB_SIZE = meta["vocab_size"]
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Train tokens: {meta['train_tokens']:,}")

# Model config — debug size but with real vocab
SEQ_LEN = 128
cfg = AriaConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    n_kv_heads=2,
    n_shared_experts=1,
    n_routed_experts=2,
    d_expert=128,
    top_k_experts=1,
    vocab_size=VOCAB_SIZE,
    max_seq_len=SEQ_LEN,
    attention_window=64,
    use_draft_head=False,
    use_ternary=True,
)

# ── Model ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = Aria(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params")

# ── Data ────────────────────────────────────────────────────
train_ds = BinaryTokenDataset(str(DATA_DIR / "train.bin"), seq_len=SEQ_LEN)
val_ds = BinaryTokenDataset(str(DATA_DIR / "val.bin"), seq_len=SEQ_LEN)
print(f"Train: {train_ds}")
print(f"Val:   {val_ds}")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

# ── Training ────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_DIR / "aria_hindi.model"))

model.train()
best_val_loss = float("inf")
total_steps = 0
MAX_STEPS = 500

print(f"\n{'='*60}")
print(f"Training Aria on Hindi Wikipedia")
print(f"  {MAX_STEPS} steps, batch_size=4, seq_len={SEQ_LEN}")
print(f"{'='*60}\n")

for epoch in range(100):  # loop until MAX_STEPS
    for batch in train_loader:
        if total_steps >= MAX_STEPS:
            break

        x = batch["input_ids"].to(device)
        y = batch["target_ids"].to(device)

        out = model(x, target_ids=y)
        loss = out.loss

        if loss.isnan():
            print(f"Step {total_steps}: NaN loss! Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_steps += 1

        if total_steps % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {total_steps:4d} | loss={loss.item():.4f} | lr={lr:.2e}")

        if total_steps % 100 == 0:
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, vbatch in enumerate(val_loader):
                    if i >= 20:  # 20 val batches
                        break
                    vx = vbatch["input_ids"].to(device)
                    vy = vbatch["target_ids"].to(device)
                    vout = model(vx, target_ids=vy)
                    val_losses.append(vout.loss.item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
            print(f"  ► Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
                print(f"  ► Saved best checkpoint (val_loss={val_loss:.4f})")

            # Generate a sample
            model.eval()
            with torch.no_grad():
                # Start with a Hindi prompt token
                prompt_text = "भारत"
                prompt_ids = sp.encode(prompt_text, out_type=int)
                prompt = torch.tensor([prompt_ids], device=device)
                generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
                gen_text = sp.decode(generated[0].tolist())
                print(f"  ► Sample: {gen_text[:200]}")

            model.train()

    if total_steps >= MAX_STEPS:
        break
    if total_steps > 0 and loss.isnan():
        break

print(f"\n{'='*60}")
print(f"Training complete: {total_steps} steps")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"{'='*60}")
