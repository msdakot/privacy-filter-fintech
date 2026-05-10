"""Span-level F1, precision, recall using seqeval. Per-label breakdown."""

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def spans_to_bio(text: str, spans: list[dict]) -> list[str]:
    """Convert character-level spans to token-level BIO tags (whitespace tokenization)."""
    tokens = text.split()
    tags = ["O"] * len(tokens)

    pos = 0
    token_offsets = []
    for tok in tokens:
        start = text.find(tok, pos)
        end = start + len(tok)
        token_offsets.append((start, end))
        pos = end

    for span in sorted(spans, key=lambda s: s["start"]):
        label = span["label"]
        s_char, e_char = span["start"], span["end"]
        inside = False
        for i, (t_start, t_end) in enumerate(token_offsets):
            if t_start >= s_char and t_end <= e_char:
                tags[i] = f"{'B' if not inside else 'I'}-{label}"
                inside = True

    return tags


def compute_metrics(
    true_spans_list: list[list[dict]],
    pred_spans_list: list[list[dict]],
    texts: list[str],
) -> dict:
    y_true = [spans_to_bio(t, s) for t, s in zip(texts, true_spans_list)]
    y_pred = [spans_to_bio(t, s) for t, s in zip(texts, pred_spans_list)]

    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred),
    }
