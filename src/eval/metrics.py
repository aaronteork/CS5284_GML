import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def classification_metrics(scores, labels, groups=None, top_frac=0.01):
    """Metrics for "did this road have an accident".

    `scores` are probabilities (or any risk score), `labels` are 0/1. Top-k precision
    and recall flag the `top_frac` highest-scoring roads; with `groups` (e.g. month ids)
    they're computed per group and averaged, matching how a monthly risk list is used.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    metrics = {
        "AUROC": roc_auc_score(labels, scores),
        "PR-AUC": average_precision_score(labels, scores),
        "precision@0.5": precision_score(labels, scores > 0.5, zero_division=0),
        "recall@0.5": recall_score(labels, scores > 0.5, zero_division=0),
    }

    groups = np.zeros(len(scores), dtype=int) if groups is None else np.asarray(groups)
    precisions, recalls = [], []
    for g in np.unique(groups):
        s, y = scores[groups == g], labels[groups == g]
        if y.sum() == 0:
            continue
        k = max(1, int(np.ceil(top_frac * len(s))))
        hits = y[np.argsort(-s)[:k]].sum()
        precisions.append(hits / k)
        recalls.append(hits / y.sum())
    pct = f"{top_frac:.0%}"
    metrics[f"precision@top{pct}"] = float(np.mean(precisions))
    metrics[f"recall@top{pct}"] = float(np.mean(recalls))
    return {k: float(v) for k, v in metrics.items()}


def regression_metrics(preds, counts):
    """Metrics for "how many accidents", overall and split by whether the road had any."""
    preds, counts = np.asarray(preds, dtype=float), np.asarray(counts, dtype=float)
    error = np.abs(preds - counts)
    positive = counts > 0
    return {
        "MAE": float(error.mean()),
        "MSE": float(((preds - counts) ** 2).mean()),
        "MAE (accident roads)": float(error[positive].mean()) if positive.any() else float("nan"),
        "MAE (no-accident roads)": float(error[~positive].mean()) if (~positive).any() else float("nan"),
    }
