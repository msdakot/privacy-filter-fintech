"""Side-by-side F1 comparison: trained model vs OpenMed/privacy-filter-nemotron baseline."""

import yaml
from datasets import load_dataset
from transformers import pipeline

from src.eval.metrics import compute_metrics


def run_model(model_id: str, texts: list[str], token: str | None = None) -> list[list[dict]]:
    pipe = pipeline(
        "token-classification",
        model=model_id,
        aggregation_strategy="simple",
        token=token,
    )
    results = []
    for text in texts:
        preds = pipe(text)
        spans = [
            {"start": p["start"], "end": p["end"], "label": p["entity_group"]}
            for p in preds
        ]
        results.append(spans)
    return results


def compare(cfg_path: str = "configs/eval_config.yaml", token: str | None = None) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = load_dataset(cfg["dataset"]["hf_path"], split=cfg["dataset"]["split"], token=token)
    texts = ds["text"]
    true_spans = ds["spans"]

    print(f"Evaluating {cfg['models']['trained']} ...")
    trained_preds = run_model(cfg["models"]["trained"], texts, token=token)
    trained_metrics = compute_metrics(true_spans, trained_preds, texts)

    print(f"Evaluating {cfg['models']['baseline']} ...")
    baseline_preds = run_model(cfg["models"]["baseline"], texts, token=token)
    baseline_metrics = compute_metrics(true_spans, baseline_preds, texts)

    return {"trained": trained_metrics, "baseline": baseline_metrics}
