import argparse
import json
import os
from pathlib import Path

import yaml

from src.eval.compare import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    with open(args.eval_config) as f:
        cfg = yaml.safe_load(f)

    results = compare(args.eval_config, token=args.hf_token)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nBaseline F1: {results['baseline']['f1']:.4f}")
    print(f"Trained F1:  {results['trained']['f1']:.4f}")
    print(f"\nTrained report:\n{results['trained']['report']}")


if __name__ == "__main__":
    main()
