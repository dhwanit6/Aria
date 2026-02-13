"""
Binary token dataset for Aria training.

Loads pre-tokenized binary files (uint16) and serves fixed-length chunks
for language model training.
"""
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class BinaryTokenDataset(Dataset):
    """
    Memory-mapped dataset over a binary file of uint16 token IDs.
    
    Each sample is a contiguous chunk of (seq_len + 1) tokens:
      - input:  tokens[i : i + seq_len]
      - target: tokens[i + 1 : i + seq_len + 1]
    
    Uses memory-mapping for zero-copy reads — works with corpora
    larger than RAM.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        stride: int | None = None,
    ):
        """
        Args:
            data_path: Path to .bin file of uint16 tokens
            seq_len: Sequence length for training
            stride: Step between consecutive samples (default: seq_len, no overlap)
        """
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Memory-map the binary file
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # Number of complete samples
        # Need seq_len + 1 tokens per sample (input + 1 shifted target)
        usable = self.n_tokens - seq_len - 1
        self.n_samples = max(0, usable // self.stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len + 1

        # Read chunk and convert to int64 (PyTorch embedding requirement)
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input tokens
        y = torch.from_numpy(chunk[1:])    # target tokens (shifted by 1)

        return {"input_ids": x, "target_ids": y}

    def __repr__(self) -> str:
        return (
            f"BinaryTokenDataset(tokens={self.n_tokens:,}, "
            f"samples={self.n_samples:,}, seq_len={self.seq_len})"
        )


def create_dataloaders(
    data_dir: str,
    seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create train and val dataloaders from preprocessed binary data.
    
    Returns:
        (train_loader, val_loader) — val_loader may be None if no val.bin exists
    """
    data_dir = Path(data_dir)

    # Load metadata
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Dataset: {meta['train_tokens']:,} train, {meta['val_tokens']:,} val tokens")
        print(f"Vocab size: {meta['vocab_size']}")

    # Train loader
    train_path = data_dir / "train.bin"
    assert train_path.exists(), f"Missing {train_path}"
    train_ds = BinaryTokenDataset(str(train_path), seq_len=seq_len)
    print(f"Train: {train_ds}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val loader (optional)
    val_loader = None
    val_path = data_dir / "val.bin"
    if val_path.exists() and val_path.stat().st_size > 0:
        val_ds = BinaryTokenDataset(str(val_path), seq_len=seq_len)
        print(f"Val:   {val_ds}")
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

    return train_loader, val_loader
