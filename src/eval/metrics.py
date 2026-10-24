import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def classification_metrics(scores, labels, groups=None, top_frac=0.01):
    """Metrics for "did this road have an accident".

    `scores` are probabilities (or any risk score), `labels` are 0/1. Top-k precision
    and recall flag the `top_frac` highest-scoring roads; with `groups` (e.g. month ids)
    they're computed per group and averaged, matching how a monthly risk list is used.
    AUROC is computed per group and weighted by the positive-road count, following
    the paper's aggregation. Pooled AUROC is retained as a supplementary metric.
    Other classification metrics use the pooled examples.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    nonempty = len(labels) > 0
    has_positive = bool(labels.any())
    metrics = {
        "AUROC": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan"),
        "PR-AUC": (average_precision_score(labels, scores) if has_positive else 0.0)
                  if nonempty else float("nan"),
        "precision@0.5": precision_score(labels, scores > 0.5, zero_division=0)
                         if nonempty else float("nan"),
        "recall@0.5": recall_score(labels, scores > 0.5, zero_division=0)
                      if nonempty else float("nan"),
    }

    groups = np.zeros(len(scores), dtype=int) if groups is None else np.asarray(groups)
    monthly_auc, auc_weights = [], []
    precisions, recalls = [], []
    for g in np.unique(groups):
        s, y = scores[groups == g], labels[groups == g]
        if len(np.unique(y)) == 2:
            monthly_auc.append(roc_auc_score(y, s))
            auc_weights.append(y.sum())
        k = max(1, int(np.ceil(top_frac * len(s))))
        hits = y[np.argsort(-s)[:k]].sum()
        precisions.append(hits / k)
        # Precision is zero in an accident-free month; recall is undefined.
        if y.sum() > 0:
            recalls.append(hits / y.sum())
    pct = f"{top_frac:.0%}"
    metrics["AUROC (pooled)"] = metrics["AUROC"]
    metrics["AUROC"] = float(np.average(monthly_auc, weights=auc_weights)) if auc_weights else float("nan")
    metrics[f"precision@top{pct}"] = float(np.mean(precisions)) if precisions else float("nan")
    metrics[f"recall@top{pct}"] = float(np.mean(recalls)) if recalls else float("nan")
    return {k: float(v) for k, v in metrics.items()}


def regression_metrics(preds, counts):
    """Metrics for "how many accidents", overall and split by whether the road had any."""
    preds, counts = np.asarray(preds, dtype=float), np.asarray(counts, dtype=float)
    error = np.abs(preds - counts)
    positive = counts > 0
    return {
        "MAE": float(error.mean()) if len(error) else float("nan"),
        "MSE": float(((preds - counts) ** 2).mean()) if len(error) else float("nan"),
        "MAE (accident roads)": float(error[positive].mean()) if positive.any() else float("nan"),
        "MAE (no-accident roads)": float(error[~positive].mean()) if (~positive).any() else float("nan"),
    }
