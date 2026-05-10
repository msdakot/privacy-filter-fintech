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
    args = parser.parse_args()

    with open(args.train_config) as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]["base"]
    dataset = cfg["dataset"]["hf_path"]
    output = cfg["output"]["hf_repo"]
    lr = cfg["training"]["learning_rate"]
    epochs = cfg["training"]["epochs"]
    bs = cfg["training"]["batch_size"]
    grad_accum = cfg["training"]["grad_accumulation_steps"]
    dtype = cfg["training"]["dtype"]

    cmd = [
        "opf", "train",
        "--model", model,
        "--dataset", dataset,
        "--output", output,
        "--learning-rate", str(lr),
        "--epochs", str(epochs),
        "--batch-size", str(bs),
        "--grad-accumulation-steps", str(grad_accum),
        "--dtype", dtype,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
