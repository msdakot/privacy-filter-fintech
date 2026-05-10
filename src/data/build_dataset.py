"""
Main orchestrator: loads all sources, harmonizes, validates span alignment,
splits 80/10/10, and pushes to HF Hub as msdakot/fintech-privacy-pii.
"""

import argparse
import random
import sys

import yaml
from datasets import Dataset, DatasetDict

from src.data.load_gretel import load_gretel
from src.data.load_nemotron import load_nemotron
from src.data.harmonize import harmonize
from src.data.synthetic import generate_synthetic


def validate_spans(records: list[dict]) -> tuple[list[dict], int]:
    valid, dropped = [], 0
    for rec in records:
        text = rec["text"]
        ok = True
        for span in rec["spans"]:
            s, e = span["start"], span["end"]
            if e > len(text) or s < 0 or s >= e:
                ok = False
                break
        if ok:
            valid.append(rec)
        else:
            dropped += 1
    return valid, dropped


def dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        key = r["text"][:120]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def split(records: list[dict], train: float, val: float) -> tuple[list, list, list]:
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train)
    n_val = int(n * val)
    return records[:n_train], records[n_train:n_train + n_val], records[n_train + n_val:]


def to_hf_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list([
        {
            "text": r["text"],
            "spans": r["spans"],
            "source": r["source"],
            "language": r.get("language", "en"),
        }
        for r in records
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip HF push")
    args = parser.parse_args()

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)

    token = args.hf_token
    hf_repo = cfg["output"]["hf_repo"]
    train_frac = cfg["output"]["train_split"]
    val_frac = cfg["output"]["val_split"]

    print("Loading Gretel...")
    gretel = load_gretel(token=token)
    print(f"  {len(gretel)} records")

    print("Loading Nemotron...")
    nemotron = load_nemotron(token=token)
    print(f"  {len(nemotron)} records (finance-domain filtered)")

    print("Generating synthetic examples...")
    synthetic = generate_synthetic(n_per_label=65)
    print(f"  {len(synthetic)} records")

    print("Harmonizing labels...")
    all_records = harmonize(gretel + nemotron + synthetic)

    print("Deduplicating...")
    all_records = dedup(all_records)
    print(f"  {len(all_records)} records after dedup")

    print("Validating span alignment...")
    all_records, dropped = validate_spans(all_records)
    if dropped:
        print(f"  WARNING: dropped {dropped} records with invalid spans", file=sys.stderr)
    print(f"  {len(all_records)} records pass validation")

    print("Splitting 80/10/10...")
    train_recs, val_recs, test_recs = split(all_records, train_frac, val_frac)
    print(f"  train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    dd = DatasetDict({
        "train": to_hf_dataset(train_recs),
        "validation": to_hf_dataset(val_recs),
        "test": to_hf_dataset(test_recs),
    })

    if args.dry_run:
        print("Dry run — skipping HF push. Dataset preview:")
        print(dd)
        return

    print(f"Pushing to HF Hub: {hf_repo} ...")
    dd.push_to_hub(hf_repo, token=token)
    print("Done.")


if __name__ == "__main__":
    main()
