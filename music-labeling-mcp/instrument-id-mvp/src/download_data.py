"""
Download and extract a subset of the IRMAS dataset.

IRMAS-TrainingData: one label per audio file (instrument name = folder name)
IRMAS-TestingData:  multi-label (each .wav has a companion .txt with labels)

Usage:
    python src/download_data.py --max-per-class 50
    python src/download_data.py --max-per-class 50 --include-test
"""

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_BASE = "https://zenodo.org/api/records/1290750/files"
TRAIN_URL   = f"{ZENODO_BASE}/IRMAS-TrainingData.zip/content"
TEST_URLS   = [
    f"{ZENODO_BASE}/IRMAS-TestingData-Part1.zip/content",
    f"{ZENODO_BASE}/IRMAS-TestingData-Part2.zip/content",
    f"{ZENODO_BASE}/IRMAS-TestingData-Part3.zip/content",
]

DATA_DIR = Path(__file__).parent.parent / "data" / "IRMAS"


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, out_dir)


def limit_training_files(train_dir: Path, max_per_class: int, seed: int = 42):
    """Keep only max_per_class .wav files per instrument folder."""
    random.seed(seed)
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        if len(wavs) <= max_per_class:
            continue
        to_remove = random.sample(wavs, len(wavs) - max_per_class)
        for f in to_remove:
            f.unlink()
        print(f"  {class_dir.name}: kept {max_per_class}/{len(wavs)} files")


def find_training_root(extracted_dir: Path) -> Path:
    """IRMAS zip may have a top-level folder; find the dir with instrument subfolders."""
    # Direct instrument folders (e.g. cel, cla …)
    instrument_names = {"cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"}
    if any((extracted_dir / name).is_dir() for name in instrument_names):
        return extracted_dir
    # One extra nesting level
    for child in extracted_dir.iterdir():
        if child.is_dir() and any((child / name).is_dir() for name in instrument_names):
            return child
    raise RuntimeError(f"Cannot locate instrument folders under {extracted_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download IRMAS dataset subset")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Max .wav files to keep per instrument class (default: 50)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download IRMAS-TestingData1 (multi-label audio for evaluate.py)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Root directory for IRMAS data",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Training data ────────────────────────────────────────────────────────
    train_zip = data_dir / "IRMAS-TrainingData.zip"
    train_out = data_dir / "training"

    if not train_out.exists():
        if not train_zip.exists():
            download_file(TRAIN_URL, train_zip)
        extract_zip(train_zip, train_out)

    train_root = find_training_root(train_out)
    print(f"Training root: {train_root}")
    classes = sorted(d.name for d in train_root.iterdir() if d.is_dir())
    print(f"Found classes: {classes}")

    if args.max_per_class:
        print(f"\nLimiting to {args.max_per_class} files per class …")
        limit_training_files(train_root, args.max_per_class)

    total = sum(len(list((train_root / c).glob("*.wav"))) for c in classes)
    print(f"\nTraining files after pruning: {total}")

    # ── Testing data (optional, 3 parts) ─────────────────────────────────────
    if args.include_test:
        test_out = data_dir / "testing"
        test_out.mkdir(exist_ok=True)
        for i, url in enumerate(TEST_URLS, start=1):
            zip_name = f"IRMAS-TestingData-Part{i}.zip"
            test_zip = data_dir / zip_name
            part_extracted = data_dir / f"testing_part{i}_done"
            if not part_extracted.exists():
                if not test_zip.exists():
                    download_file(url, test_zip)
                extract_zip(test_zip, test_out)
                part_extracted.touch()   # marker so we don't re-extract
        test_wavs = list(test_out.rglob("*.wav"))
        print(f"Testing files: {len(test_wavs)}")

    print("\nDone. Dataset ready at:", data_dir)


if __name__ == "__main__":
    main()
