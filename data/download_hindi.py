"""
Download and prepare Hindi + Hinglish training data.

Sources:
  1. CulturaX Hindi subset (CC-licensed, cleaned web text)
  2. Hindi Wikipedia (clean, factual)
  3. Hinglish conversations (romanized Hindi)

Usage:
    python -m data.download_hindi --output_dir data/raw --max_samples 500000
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def is_devanagari(char: str) -> bool:
    """Check if a character is Devanagari."""
    try:
        return "DEVANAGARI" in unicodedata.name(char, "")
    except ValueError:
        return False


def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic chars that are Devanagari."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if is_devanagari(c)) / len(alpha_chars)


def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace, remove control chars."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Remove control characters (keep newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize whitespace (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_quality_hindi(text: str, min_length: int = 50) -> bool:
    """Filter for quality Hindi text."""
    if len(text) < min_length:
        return False
    # Must have substantial Devanagari content (>40% for mixed Hindi-English)
    if devanagari_ratio(text) < 0.3:
        return False
    # Filter out mostly-URL or mostly-number text
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True


def is_quality_hinglish(text: str, min_length: int = 30) -> bool:
    """Filter for quality Hinglish text (romanized Hindi + English mix)."""
    if len(text) < min_length:
        return False
    # Hinglish should be mostly Latin script
    if devanagari_ratio(text) > 0.3:
        return False  # This is Devanagari Hindi, not Hinglish
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    return True


def download_hindi_web(output_dir: Path, max_samples: int) -> int:
    """Download Hindi web text. Tries multiple sources in order."""
    print("\n[1/3] Downloading Hindi web text...")
    out_file = output_dir / "hindi_web.txt"

    if out_file.exists():
        line_count = sum(1 for _ in open(out_file, "r", encoding="utf-8"))
        print(f"  Already exists: {out_file} ({line_count} docs)")
        return line_count

    # Try sources in order of preference
    ds = None
    source_name = ""
    sources = [
        ("CulturaX", "uonlp/CulturaX", {"name": "hi", "split": "train", "streaming": True}),
        ("Sangraha", "ai4bharat/sangraha", {"name": "verified_hi", "split": "train", "streaming": True}),
        ("IndicCorp", "ai4bharat/IndicCorp", {"name": "hi", "split": "train", "streaming": True}),
        ("CC100-Hindi", "cc100", {"name": "hi", "split": "train", "streaming": True}),
        ("OSCAR-Hindi", "oscar-corpus/OSCAR-2301", {"name": "hi", "split": "train", "streaming": True}),
    ]

    for name, dataset_id, kwargs in sources:
        try:
            print(f"  Trying {name}...")
            ds = load_dataset(dataset_id, **kwargs)
            source_name = name
            print(f"  Connected to {name}!")
            break
        except Exception as e:
            print(f"  {name} failed: {str(e)[:80]}")
            continue

    if ds is None:
        print("  All web sources failed. Proceeding with Wikipedia only.")
        return 0

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=max_samples, desc=f"  {source_name}"):
            text = clean_text(sample.get("text", ""))
            if is_quality_hindi(text):
                f.write(text + "\n\n")
                count += 1
                if count >= max_samples:
                    break

    print(f"  Saved {count} documents from {source_name} to {out_file}")
    return count


def download_hindi_wikipedia(output_dir: Path, max_samples: int) -> int:
    """Download Hindi Wikipedia articles."""
    print("\n[2/3] Downloading Hindi Wikipedia...")
    out_file = output_dir / "hindi_wiki.txt"

    if out_file.exists():
        line_count = sum(1 for line in open(out_file, "r", encoding="utf-8") if line.strip())
        print(f"  Already exists: {out_file} ({line_count} docs)")
        return line_count

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.hi",
        split="train",
    )

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=min(max_samples, len(ds)), desc="  Hindi Wiki"):
            text = clean_text(sample.get("text", ""))
            if is_quality_hindi(text, min_length=100):
                f.write(text + "\n\n")
                count += 1
                if count >= max_samples:
                    break

    print(f"  Saved {count} documents to {out_file}")
    return count


def download_hinglish(output_dir: Path, max_samples: int) -> int:
    """Download Hinglish (romanized Hindi) data."""
    print("\n[3/3] Downloading Hinglish data...")
    out_file = output_dir / "hinglish.txt"

    if out_file.exists():
        line_count = sum(1 for line in open(out_file, "r", encoding="utf-8") if line.strip())
        print(f"  Already exists: {out_file} ({line_count} docs)")
        return line_count

    # Try multiple Hinglish sources
    count = 0
    sources_tried = []

    # Source 1: LinCE Hinglish
    try:
        ds = load_dataset(
            "lince",
            "sa_en",
            split="train",
            trust_remote_code=True,
        )
        sources_tried.append("LinCE")
        with open(out_file, "w", encoding="utf-8") as f:
            for sample in tqdm(ds, total=min(max_samples, len(ds)), desc="  LinCE Hinglish"):
                # LinCE has token-level data, reconstruct sentences
                tokens = sample.get("tokens", [])
                if tokens:
                    text = " ".join(tokens)
                    text = clean_text(text)
                    if len(text) > 20:
                        f.write(text + "\n")
                        count += 1
                        if count >= max_samples:
                            break
    except Exception as e:
        print(f"  LinCE failed: {e}")

    # Source 2: Generate synthetic Hinglish from Hindi patterns
    # Common Hinglish patterns for bootstrapping
    if count < 1000:
        print("  Generating synthetic Hinglish examples...")
        hinglish_examples = [
            "yaar aaj mausam bahut accha hai, chal bahar chalte hain",
            "bhai tu sahi bol raha hai, mujhe bhi lagta hai ki yeh kaam hona chahiye",
            "mujhe ek coffee chahiye, aur thoda sa kuch khane ko bhi",
            "kal main market gaya tha, bahut sari cheezein dekhi",
            "tum kab aa rahe ho ghar pe? sab log wait kar rahe hain",
            "yeh movie bahut acchi hai, tum zaroor dekhna",
            "main kal se padhai kar raha hoon, exam pass aane wala hai",
            "aaj office mein bahut kaam tha, thak gaya main",
            "kya plan hai weekend ka? kahin bahar chalein?",
            "mere phone ki battery khatam ho gayi, charger de do please",
            "yeh khana bahut tasty hai, recipe batao na",
            "traffic bahut zyada hai aaj, late ho jaunga shayad",
            "bhai coding seekhni hai mujhe, kahan se shuru karun?",
            "machine learning bahut interesting topic hai yaar",
            "mujhe lagta hai hum yeh project complete kar lenge time pe",
            "subah jaldi uthna padega kal, meeting hai 9 baje",
            "yaar yeh AI models kitne powerful ho gaye hain",
            "college mein aaj bahut maza aaya, professor ne accha padhaya",
            "ghar pe sab theek hain? mummy papa ko mera salaam bolna",
            "winter mein chai peene ka alag hi maza hai",
        ]
        with open(out_file, "a" if count > 0 else "w", encoding="utf-8") as f:
            for ex in hinglish_examples:
                f.write(ex + "\n")
                count += 1

    print(f"  Saved {count} Hinglish samples to {out_file}")
    return count


def create_merged_corpus(output_dir: Path) -> Path:
    """Merge all downloaded files into one training corpus."""
    print("\nMerging all data into training corpus...")
    merged_file = output_dir / "hindi_merged.txt"

    total_chars = 0
    total_docs = 0
    with open(merged_file, "w", encoding="utf-8") as out:
        for txt_file in sorted(output_dir.glob("*.txt")):
            if txt_file.name == "hindi_merged.txt":
                continue
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                total_chars += len(content)
                total_docs += content.count("\n\n") + 1
                print(f"  Added {txt_file.name}: {len(content)/1e6:.1f}MB")

    size_mb = total_chars / 1e6
    print(f"\nMerged corpus: {merged_file}")
    print(f"  Total: {size_mb:.1f} MB, ~{total_docs} documents")
    print(f"  Estimated tokens: ~{total_chars // 4 / 1e6:.1f}M")
    return merged_file


def main():
    parser = argparse.ArgumentParser(description="Download Hindi training data")
    parser.add_argument(
        "--output_dir", type=str, default="data/raw",
        help="Output directory for raw text files",
    )
    parser.add_argument(
        "--max_samples", type=int, default=500_000,
        help="Max documents per source",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Aria Hindi Data Pipeline")
    print("=" * 60)

    # Download each source
    n_web = download_hindi_web(output_dir, args.max_samples)
    n_wiki = download_hindi_wikipedia(output_dir, args.max_samples)
    n_hinglish = download_hinglish(output_dir, args.max_samples)

    print(f"\n{'='*60}")
    print(f"Download complete:")
    print(f"  Web text:           {n_web} docs")
    print(f"  Wikipedia:          {n_wiki} docs")
    print(f"  Hinglish:           {n_hinglish} docs")
    print(f"  Total:              {n_web + n_wiki + n_hinglish} docs")

    # Merge into single corpus
    merged = create_merged_corpus(output_dir)
    print(f"\nNext step: python -m data.train_tokenizer --input {merged}")


if __name__ == "__main__":
    main()
