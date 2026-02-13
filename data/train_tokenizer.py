"""
Train a SentencePiece BPE tokenizer for Hindi + Hinglish + English.

Handles:
  - Devanagari script (Hindi/Sanskrit)
  - Latin script (Hinglish/English)
  - Numbers, punctuation, special tokens

Usage:
    python -m data.train_tokenizer --input data/raw/hindi_merged.txt --vocab_size 32000
"""
import argparse
from pathlib import Path

import sentencepiece as spm


def train_tokenizer(
    input_file: str,
    output_prefix: str = "data/tokenizer/aria_hindi",
    vocab_size: int = 32000,
    character_coverage: float = 0.9999,
):
    """Train a BPE tokenizer optimized for Hindi + Hinglish."""
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training tokenizer:")
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_prefix}.model")
    print(f"  Vocab:    {vocab_size}")
    print(f"  Coverage: {character_coverage}")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        # Normalization: NFC for Devanagari (important for combining marks)
        normalization_rule_name="nfkc",
        # Byte-fallback ensures NO unknown tokens ever
        byte_fallback=True,
        # Treat whitespace as part of tokens (like GPT-style)
        allow_whitespace_only_pieces=True,
        # Split digits individually for better number handling
        split_digits=True,
        # Special tokens
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<|pad|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        unk_piece="<|unk|>",
        # Training params
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
        # Input sentence size for sampling (use all for small corpora)
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    print(f"\nTokenizer saved to: {output_prefix}.model")
    return output_prefix + ".model"


def validate_tokenizer(model_path: str):
    """Validate tokenizer on Hindi + Hinglish + English samples."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    print(f"\nTokenizer validation:")
    print(f"  Vocab size: {sp.get_piece_size()}")
    print(f"  BOS id: {sp.bos_id()}")
    print(f"  EOS id: {sp.eos_id()}")
    print(f"  PAD id: {sp.pad_id()}")

    test_cases = [
        # Devanagari Hindi
        ("Hindi", "भारत एक विशाल देश है। यहाँ की संस्कृति बहुत पुरानी है।"),
        ("Hindi-2", "मशीन लर्निंग एक कृत्रिम बुद्धिमत्ता की शाखा है जो डेटा से सीखती है।"),
        # Hinglish (romanized)
        ("Hinglish", "yaar aaj mausam bahut accha hai, chal bahar chalte hain"),
        ("Hinglish-2", "bhai mujhe coding seekhni hai, kahan se shuru karun?"),
        # English
        ("English", "The quick brown fox jumps over the lazy dog."),
        # Mixed Hindi-English
        ("Mixed", "मुझे machine learning में interest है, python सीख रहा हूँ।"),
        # Numbers
        ("Numbers", "आज 13 फरवरी 2026 है, temperature 25°C है।"),
    ]

    for name, text in test_cases:
        tokens = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        match = decoded == text
        print(f"\n  [{name}]")
        print(f"    Text:    {text}")
        print(f"    Tokens:  {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"    IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"    Count:   {len(tokens)}")
        print(f"    Round-trip: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"    Decoded: {decoded}")


import os


def main():
    parser = argparse.ArgumentParser(description="Train Aria Hindi tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument(
        "--output_prefix", type=str, default="data/tokenizer/aria_hindi",
        help="Output model prefix",
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()

    model_path = train_tokenizer(args.input, args.output_prefix, args.vocab_size)
    validate_tokenizer(model_path)


if __name__ == "__main__":
    main()
