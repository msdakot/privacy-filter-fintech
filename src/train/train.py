"""
Thin wrapper around `opf train` CLI. Runs from configs/train_config.yaml.
Actual execution happens in Colab (notebooks/02_train_colab.ipynb).
"""

import argparse
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--train-jsonl", required=True, help="Local opf-format train JSONL")
    parser.add_argument("--val-jsonl", required=True, help="Local opf-format validation JSONL")
    parser.add_argument("--checkpoint-dir", required=True, help="Local base model checkpoint dir")
    parser.add_argument("--output-dir", required=True, help="Local output dir for trained checkpoint")
    args = parser.parse_args()

    with open(args.train_config) as f:
        cfg = yaml.safe_load(f)

    t = cfg["training"]
    cmd = [
        "opf", "train", args.train_jsonl,
        "--validation-dataset", args.val_jsonl,
        "--checkpoint", args.checkpoint_dir,
        "--label-space-json", "configs/label_space.json",
        "--output-dir", args.output_dir,
        "--overwrite-output",
        "--epochs", str(t["epochs"]),
        "--batch-size", str(t["batch_size"]),
        "--grad-accum-steps", str(t["grad_accumulation_steps"]),
        "--learning-rate", str(t["learning_rate"]),
        "--weight-decay", str(t["weight_decay"]),
        "--output-param-dtype", t["dtype"],
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
