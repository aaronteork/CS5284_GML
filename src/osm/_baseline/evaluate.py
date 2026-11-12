# Pinned comparison baseline; extracted from osm_baseline_comparison.ipynb, cell 11.
import numpy as np
import torch

from .metrics import classification_metrics, regression_metrics

PROTOCOLS = ("all", "balanced")


def balanced_road_ids(y, year, month):
    """Every accident road plus an equal number of accident-free roads.

    The sample is seeded by the month, so every model is scored on the same roads.
    This is a supplementary diagnostic; the paper's filtered evaluation population
    is different from both our balanced sample and our full-network population.
    """
    positive = torch.nonzero(y > 0).squeeze(1).cpu().numpy()
    negative = torch.nonzero(y == 0).squeeze(1).cpu().numpy()
    rng = np.random.default_rng(year * 12 + month)
    sampled = rng.choice(negative, size=min(len(positive), len(negative)), replace=False)
    return torch.from_numpy(np.concatenate([positive, sampled]))


@torch.no_grad()
def predict(model, data, task):
    """Predictions for every road: accident probability or expected count."""
    model.eval()
    out = model(data)
    return torch.sigmoid(out) if task == "classification" else out.clamp(min=0)


def evaluate(model, dataset, months, task, device, return_predictions=False):
    """Score a model on `months` under both protocols.

    - "all": every road in every month, i.e. how the model would be used.
    - "balanced": accident roads plus an equal sample of accident-free roads,
      reported as a supplementary diagnostic.

    Classification metrics are always reported (a predicted count is also a risk
    score); regression metrics are added for the regression task.

    Zero-accident months contribute to "all" metrics and prediction exports.
    They have no balanced examples, since there are no positives to balance.
    """
    collected = {p: {"pred": [], "y": [], "group": []} for p in PROTOCOLS}
    all_preds, all_y, evaluated_months = [], [], []

    for group, (year, month) in enumerate(months):
        data = dataset.monthly(year, month).to(device)
        pred, y = predict(model, data, task).cpu(), data.y.cpu()

        ids = balanced_road_ids(y, year, month)
        for protocol, (p, t) in {"all": (pred, y), "balanced": (pred[ids], y[ids])}.items():
            collected[protocol]["pred"].append(p)
            collected[protocol]["y"].append(t)
            collected[protocol]["group"].append(torch.full((len(p),), group))

        if return_predictions:
            all_preds.append(pred)
            all_y.append(y)
            evaluated_months.append((year, month))

    results = {}
    for protocol, c in collected.items():
        pred, y, group = (torch.cat(c[k]).numpy() for k in ("pred", "y", "group"))
        metrics = classification_metrics(pred, y > 0, groups=group)
        if task == "regression":
            metrics.update(regression_metrics(pred, y))
        results[protocol] = metrics

    if not return_predictions:
        return results
    predictions = {
        "months": evaluated_months,
        "pred": torch.stack(all_preds),            # [num_months, num_roads]
        "y": torch.stack(all_y),                   # [num_months, num_roads]
        "road_index": dataset.edge_index[:, dataset.road_edge],  # [2, num_roads]
        "target_unit": getattr(dataset, "target_unit", "physical"),
    }
    if getattr(dataset, "node_coordinates", None) is not None:
        predictions["node_coordinates"] = dataset.node_coordinates  # [N, 2], longitude then latitude
    return results, predictions
