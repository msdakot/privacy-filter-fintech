# Training Recipe

Documents the hyperparameters, infrastructure, and decisions for the training run.

---

## Base model

`openai/privacy-filter` — the canonical opf-format checkpoint (8 labels, v2:33 BIOES).

**Why not OpenMed/privacy-filter-nemotron**: opf train validates that the checkpoint's `num_labels` matches a known set (v2:33, v4:57, v7:101). OpenMed has 221 labels (55 entities × 4 BIOES + O) which is not in the allowed set. The error is not patchable — opf only accepts checkpoints it originally produced. Label space is expanded 8→65 via `--label-space-json`.

## Hyperparameters

Based on Vasanth's recipe (the configuration used to train OpenMed-nemotron, which achieved F1 0.95 across 55 labels):

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 1e-4 | Proven recipe from OpenMed-nemotron training |
| `epochs` | 3 | Balance between convergence and T4 budget |
| `batch_size` | 2 | T4 VRAM constraint (16GB) |
| `grad_accumulation_steps` | 4 | Effective batch size = 8 |
| `dtype` | bf16 | T4 supports bf16; reduces memory vs fp32 |
| `weight_decay` | 0.0 | Default; not tuned |

## Infrastructure

- **Hardware**: Google Colab T4 (16GB VRAM)
- **Estimated runtime**: 2–4 hours
- **Checkpoint persistence**: Google Drive mount at `/content/drive/MyDrive/fintech-pii-checkpoints`
- **Disconnect recovery**: Resume from latest Drive checkpoint

## Training framework

`opf train` CLI from `https://github.com/openai/privacy-filter`

## Architectural decision: full fine-tuning vs LoRA

**LoRA** (Low-Rank Adaptation) was considered as a parameter-efficient alternative. Decision to use full fine-tuning instead:

| Factor | Reasoning |
|---|---|
| Framework support | `opf train` natively implements full fine-tuning; LoRA would require a custom training loop outside the `opf` ecosystem |
| Label delta | Only 10 new labels on top of 55 — a small output-head expansion. Catastrophic forgetting risk is low and explicitly measured in eval (general PII F1 guardrail) |
| Learning goal | The project goal is an end-to-end fine-tuning workflow. Full fine-tuning is the more foundational technique to understand first |
| Compute | 3 epochs on T4 (~3 hrs) is within budget with full fine-tuning; LoRA's compute advantage matters more for larger models or tighter constraints |

If a future iteration targets a much larger base model or requires faster iteration cycles, LoRA via `peft` would be the right revisit point.

---

## Actual results (filled in after training)

| Metric | Value |
|---|---|
| Training runtime | TBD |
| Final train loss | TBD |
| Final val loss | TBD |
| F1 on fintech labels | TBD |
| F1 on shared labels | TBD |
| F1 on general PII | TBD |
