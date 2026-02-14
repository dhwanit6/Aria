"""
Preprocess tokenized text into binary format for fast training.

Converts raw text → tokenized IDs → binary .bin files.
Each .bin file is a flat array of uint16 token IDs.

Usage:
    python -m data.preprocess --input data/raw/hindi_merged.txt \
                              --tokenizer data/tokenizer/aria_hindi.model \
                              --output_dir data/processed
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def preprocess(
    input_file: str,
    tokenizer_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    val_split: float = 0.02,
):
    """Tokenize text and save as binary chunks."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    vocab_size = sp.get_piece_size()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    print(f"Tokenizer: {tokenizer_path} (vocab={vocab_size})")
    print(f"Input:     {input_file}")
    print(f"BOS={bos_id}, EOS={eos_id}")

    assert vocab_size < 65536, f"Vocab {vocab_size} too large for uint16"

    # Read and tokenize entire corpus
    print("\nTokenizing corpus...")
    all_ids = []
    doc_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        current_doc = []
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line:
                # End of document
                if current_doc:
                    doc_text = " ".join(current_doc)
                    ids = sp.encode(doc_text, out_type=int)
                    # Add BOS/EOS markers per document
                    all_ids.append(bos_id)
                    all_ids.extend(ids)
                    all_ids.append(eos_id)
                    doc_count += 1
                    current_doc = []
            else:
                current_doc.append(line)

        # Handle last document (no trailing blank line)
        if current_doc:
            doc_text = " ".join(current_doc)
            ids = sp.encode(doc_text, out_type=int)
            all_ids.append(bos_id)
            all_ids.extend(ids)
            all_ids.append(eos_id)
            doc_count += 1

    total_tokens = len(all_ids)
    print(f"\nTokenized {doc_count} documents -> {total_tokens:,} tokens")
    print(f"Avg tokens/doc: {total_tokens / max(doc_count, 1):.0f}")

    # Convert to numpy array
    all_ids = np.array(all_ids, dtype=np.uint16)

    # Split into train/val
    val_size = int(total_tokens * val_split)
    train_ids = all_ids[:-val_size] if val_size > 0 else all_ids
    val_ids = all_ids[-val_size:] if val_size > 0 else np.array([], dtype=np.uint16)

    # Save as binary files
    train_file = out_dir / "train.bin"
    val_file = out_dir / "val.bin"

    train_ids.tofile(str(train_file))
    if len(val_ids) > 0:
        val_ids.tofile(str(val_file))

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "doc_count": doc_count,
        "dtype": "uint16",
        "tokenizer": tokenizer_path,
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    train_mb = train_ids.nbytes / 1e6
    val_mb = val_ids.nbytes / 1e6
    print(f"\nSaved:")
    print(f"  {train_file}: {len(train_ids):,} tokens ({train_mb:.1f} MB)")
    print(f"  {val_file}:   {len(val_ids):,} tokens ({val_mb:.1f} MB)")
    print(f"  {out_dir / 'meta.json'}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess text to binary tokens")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer .model path")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--val_split", type=float, default=0.02)
    args = parser.parse_args()

    preprocess(args.input, args.tokenizer, args.output_dir, val_split=args.val_split)


if __name__ == "__main__":
    main()
