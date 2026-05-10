"""
Converts our HF dataset format to opf JSONL format required by `opf train`.

Our format:  {"text": "...", "spans": [{"start": 18, "end": 40, "label": "iban"}]}
opf format:  {"text": "...", "spans": {"iban": [[18, 40]]}}
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def record_to_opf(record: dict) -> dict:
    spans_dict: dict[str, list[list[int]]] = {}
    for span in record.get("spans", []):
        label = span["label"]
        spans_dict.setdefault(label, []).append([span["start"], span["end"]])
    return {"text": record["text"], "spans": spans_dict}


def export_split(ds_split, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in ds_split:
            opf_record = record_to_opf(record)
            f.write(json.dumps(opf_record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="msdakot/fintech-privacy-pii")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output-dir", default="data/opf")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Loading {args.hf_repo} from HF Hub...")
    ds = load_dataset(args.hf_repo, token=args.hf_token)

    for split in ("train", "validation", "test"):
        path = out / f"{split}.jsonl"
        n = export_split(ds[split], path)
        print(f"  {split}: {n} records → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
