import torch
from typing import Dict


@torch.no_grad()
def run_inference(
    feature_extractor,
    concept_head,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass.
    Returns logits tensor [1, n_concepts]
    """
    feats = feature_extractor(x)
    logits = concept_head(feats)
    return logits


def apply_thresholds(
    logits: torch.Tensor,
    thresholds: Dict[str, float],
    concept_order: list[str],
) -> Dict:
    """
    Converts logits → probabilities → decisions
    """
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = {}
    final_ok = True

    for i, cname in enumerate(concept_order):
        p_ok = float(probs[i])
        thr = thresholds[cname]
        pred_ok = p_ok >= thr

        results[cname] = {
            "p_ok": p_ok,
            "threshold": thr,
            "pred_ok": bool(pred_ok),
        }

        final_ok &= bool(pred_ok)

    return {
        "per_concept": results,
        "final_ok": final_ok,
    }
