"""Delaware accident-history analyses and figures used in the final report."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.history import PhysicalNeighbours


def history_feature_catalog():
    return pd.DataFrame([
        ("acc_lag_1m", "Own-road count at T−1", "Immediate persistence"),
        ("acc_mean_3m", "Own-road mean over T−3…T−1", "Smooth recent noise"),
        ("acc_mean_6m", "Own-road mean over T−6…T−1", "Medium-term baseline"),
        ("acc_mean_12m", "Own-road mean over T−12…T−1", "Annual-scale baseline"),
        ("acc_lag_12m", "Own-road count at T−12", "Same-season reference"),
        ("acc_nonzero_fraction_12m", "Share of prior 12 months with count > 0", "Recurring vs isolated risk"),
        ("neighbor_acc_lag_1m_mean", "Adjacent-road mean at T−1", "Recent local context"),
        ("neighbor_acc_mean_3m", "Adjacent-road mean over T−3…T−1", "Smoothed local context"),
        ("neighbor_count", "Distinct physical roads sharing an endpoint", "Neighbourhood support"),
        ("history_months_available_12m", "Usable source months in prior year", "Coverage context"),
    ], columns=["Feature", "Definition", "Reason for inclusion"])


def load_training_road_counts(state_dir):
    adjacency = torch.load(state_dir / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
    edge_index = adjacency.indices().numpy()
    num_nodes = adjacency.shape[0]
    directed_keys = edge_index[0] * num_nodes + edge_index[1]
    physical_keys = np.minimum(edge_index[0], edge_index[1]) * num_nodes + np.maximum(edge_index[0], edge_index[1])
    road_keys = np.unique(physical_keys)
    edge_to_road = np.searchsorted(road_keys, physical_keys)
    road_ids = pd.read_csv(
        state_dir / "accidents_monthly.csv",
        usecols=["year", "month", "node_1_idx", "node_2_idx", "acc_count"],
    )
    raw_months = pd.PeriodIndex(
        pd.to_datetime(dict(year=road_ids["year"], month=road_ids["month"], day=1)),
        freq="M",
    )
    observed_months = set(raw_months)
    usable_months_in_source = sum(
        month in observed_months
        for month in pd.period_range("2009-01", "2022-09", freq="M")
    )
    october_2022_count = int(road_ids.loc[
        (road_ids["year"] == 2022) & (road_ids["month"] == 10), "acc_count"
    ].sum())
    print(
        f"Source coverage: {len(observed_months)} months have rows; "
        f"{usable_months_in_source} are usable through 2022-09 under the current cutoff."
    )
    print(
        f"2022-10 recorded count: {october_2022_count}; "
        f"2022-11/12 have rows: "
        f"{any(month in observed_months for month in [pd.Period('2022-11'), pd.Period('2022-12')])}."
    )

    label_directed_keys = (
        road_ids["node_1_idx"].to_numpy() * num_nodes
        + road_ids["node_2_idx"].to_numpy()
    )
    label_edge_ids = np.searchsorted(directed_keys, label_directed_keys)
    if not np.array_equal(directed_keys[label_edge_ids], label_directed_keys):
        raise ValueError("Accident labels do not align with graph edges.")
    month_ids = (
        (road_ids["year"].to_numpy() - 2009) * 12
        + road_ids["month"].to_numpy() - 1
    )
    in_training = (month_ids >= 0) & (month_ids < 48)
    directed_counts = np.zeros((48, len(directed_keys)), dtype=np.int32)
    np.add.at(
        directed_counts,
        (month_ids[in_training], label_edge_ids[in_training]),
        road_ids["acc_count"].to_numpy(dtype=np.int32)[in_training],
    )
    counts = np.zeros((48, len(road_keys)), dtype=np.float32)
    np.add.at(
        counts,
        (month_ids[in_training], edge_to_road[label_edge_ids[in_training]]),
        road_ids["acc_count"].to_numpy(dtype=np.float32)[in_training],
    )
    usable_months = np.isin(np.arange(48), month_ids[in_training])
    physical_pairs = np.column_stack((road_keys // num_nodes, road_keys % num_nodes))
    neighbours = PhysicalNeighbours(physical_pairs, num_nodes)
    forward = edge_index[0] < edge_index[1]
    reverse = edge_index[0] > edge_index[1]
    has_forward = np.zeros(len(road_keys), dtype=bool)
    has_reverse = np.zeros(len(road_keys), dtype=bool)
    np.logical_or.at(has_forward, edge_to_road, forward)
    np.logical_or.at(has_reverse, edge_to_road, reverse)
    two_way = has_forward & has_reverse
    forward_counts = np.zeros((48, len(road_keys)), dtype=np.int32)
    reverse_counts = np.zeros_like(forward_counts)
    np.add.at(forward_counts.T, edge_to_road[forward], directed_counts[:, forward].T)
    np.add.at(reverse_counts.T, edge_to_road[reverse], directed_counts[:, reverse].T)
    two_way_with_crashes = counts[:, two_way].sum(axis=0) > 0
    both_directions_cumulative = (
        (forward_counts[:, two_way].sum(axis=0) > 0)
        & (reverse_counts[:, two_way].sum(axis=0) > 0)
    )
    both_directions_same_month = (
        (forward_counts[:, two_way] > 0) & (reverse_counts[:, two_way] > 0)
    ).sum()
    two_way_positive_road_months = (counts[:, two_way] > 0).sum()
    two_way_total = int(counts[:, two_way].sum())
    lower_to_higher_total = int(directed_counts[:, forward & two_way[edge_to_road]].sum())
    print(
        f"Training direction audit: {two_way.sum():,} two-way pairs; "
        f"{two_way_with_crashes.sum():,} with crashes, "
        f"{(two_way_with_crashes & ~both_directions_cumulative).sum():,} with records on one orientation only."
    )
    print(
        f"No two-way positive road-month has crashes on both orientations "
        f"({both_directions_same_month:,}/{two_way_positive_road_months:,}); "
        f"{lower_to_higher_total / two_way_total:.2%} of two-way counts are on lower-index-to-higher-index edges."
    )

    return counts, usable_months, neighbours, edge_to_road


def history_feature_profile(train_features):
    selected = [
        ("Own-road lag 1 month", 0),
        ("Own-road mean 3 months", 1),
        ("Own-road mean 12 months", 3),
        ("Own-road nonzero fraction, 12 months", 5),
        ("Neighbour lag 1 month mean", 6),
        ("Neighbour mean 3 months", 7),
        ("Number of neighbours", 8),
    ]
    profile_rows = []
    for label, column in selected:
        values = train_features[:, :, column]
        finite = np.isfinite(values)
        observed = values[finite]
        profile_rows.append({
            "Feature": label,
            "Missing (%)": 100 * (~finite).mean(),
            "Zero (%)": 100 * np.isclose(observed, 0).mean(),
            "Median": np.median(observed),
            "P90": np.quantile(observed, 0.90),
            "P99": np.quantile(observed, 0.99),
        })
    history_feature_profile = pd.DataFrame(profile_rows)
    for column in ["Missing (%)", "Zero (%)"]:
        history_feature_profile[column] = np.floor(
            history_feature_profile[column] * 10 + 0.5
        ) / 10
    history_feature_profile[["Median", "P90", "P99"]] = history_feature_profile[
        ["Median", "P90", "P99"]
    ].round(3)
    return history_feature_profile


def history_window_correlations(train_features):
    eligible_features = train_features[12:48, :, :8].reshape(-1, 8)
    sample_ids = np.random.default_rng(42).choice(
        len(eligible_features), size=100_000, replace=False
    )
    correlation_columns = [1, 2, 3, 5]
    correlation_labels = ["Mean 3m", "Mean 6m", "Mean 12m", "Nonzero fraction 12m"]
    correlation = pd.DataFrame(
        eligible_features[sample_ids][:, correlation_columns],
        columns=correlation_labels,
    ).corr(method="spearman")
    return correlation


def plot_history_profile(train_features, history_feature_profile, correlation):
    correlation_labels = correlation.columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    y = np.arange(len(history_feature_profile))
    axes[0].barh(y, history_feature_profile["Zero (%)"], color="#d98958")
    axes[0].set_yticks(y, history_feature_profile["Feature"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Finite road-months equal to zero (%)")
    axes[0].set_title("Zero inflation")
    positive_own = train_features[:, :, 1]
    positive_own = positive_own[np.isfinite(positive_own) & (positive_own > 0)]
    positive_neighbour = train_features[:, :, 7]
    positive_neighbour = positive_neighbour[np.isfinite(positive_neighbour) & (positive_neighbour > 0)]
    upper = max(np.quantile(positive_own, 0.99), np.quantile(positive_neighbour, 0.99))
    axes[1].hist(positive_own[positive_own <= upper], bins=28, density=True,
                 alpha=0.62, color="#3569a8", label="Own-road mean 3m")
    axes[1].hist(positive_neighbour[positive_neighbour <= upper], bins=28, density=True,
                 alpha=0.62, color="#d16b32", label="Neighbour mean 3m")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Positive feature value")
    axes[1].set_ylabel("Density (log scale)")
    axes[1].set_title("Positive-value distributions")
    axes[1].legend(frameon=False)
    image = axes[2].imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    axes[2].set_xticks(range(len(correlation_labels)), correlation_labels, rotation=35, ha="right")
    axes[2].set_yticks(range(len(correlation_labels)), correlation_labels)
    for row in range(len(correlation_labels)):
        for col in range(len(correlation_labels)):
            axes[2].text(col, row, f"{correlation.iloc[row, col]:.2f}",
                         ha="center", va="center", color="black", fontsize=9)
    axes[2].set_title("Overlapping-window Spearman correlation")
    fig.colorbar(image, ax=axes[2], shrink=0.78, label="Spearman rho")
    fig.suptitle("History-feature EDA on Delaware training data")
    return fig


def history_association_tables(counts, neighbours):
    history_labels = ["0", "1", "2", "3–5", "6+"]
    road_months = np.zeros(5, dtype=np.int64)
    next_month_positive = np.zeros(5, dtype=np.int64)
    next_month_count = np.zeros(5, dtype=np.float64)
    neighbor_rows = np.zeros(3, dtype=np.int64)
    neighbor_positive = np.zeros(3, dtype=np.int64)
    neighbor_count_sum = np.zeros(3, dtype=np.float64)

    for t in range(12, 48):
        prior_3m = counts[t - 3:t].sum(axis=0)
        target = counts[t]
        group = np.digitize(prior_3m, [1, 2, 3, 6])
        road_months += np.bincount(group, minlength=5)
        next_month_positive += np.bincount(group, weights=(target > 0), minlength=5).astype(np.int64)
        next_month_count += np.bincount(group, weights=target, minlength=5)

        recent_neighbor_share = neighbours.mean((prior_3m > 0).astype(np.float32))
        degree = neighbours.degree
        neighbor_group = np.full(len(prior_3m), -1, dtype=np.int8)
        neighbor_group[(prior_3m == 0) & (degree == 0)] = 0
        neighbor_group[(prior_3m == 0) & (degree > 0) & (recent_neighbor_share == 0)] = 1
        neighbor_group[(prior_3m == 0) & (degree > 0) & (recent_neighbor_share > 0)] = 2
        for j in range(3):
            selected = neighbor_group == j
            neighbor_rows[j] += selected.sum()
            neighbor_positive[j] += (target[selected] > 0).sum()
            neighbor_count_sum[j] += target[selected].sum()

    history_risk = pd.DataFrame({
        "Prior 3-month accidents": history_labels,
        "Road-months": road_months,
        "Next-month incidence (%)": 100 * next_month_positive / road_months,
        "Mean next-month count": next_month_count / road_months,
    })
    neighbor_risk = pd.DataFrame({
        "Neighbour history, own prior 3-month total = 0": [
            "No neighbours",
            "Neighbours, none with recent crashes",
            "At least one neighbour with recent crashes",
        ],
        "Road-months": neighbor_rows,
        "Next-month incidence (%)": 100 * neighbor_positive / neighbor_rows,
        "Mean next-month count": neighbor_count_sum / neighbor_rows,
    })
    return history_risk, neighbor_risk


def plot_history_associations(history_risk, neighbor_risk):
    history_labels = history_risk.iloc[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(history_labels, history_risk["Next-month incidence (%)"], color="#3977a8")
    axes[0].set_xlabel("Accidents on the road in prior 3 months")
    axes[0].set_ylabel("Next-month incidence (%)")
    axes[0].set_title("Own-road history")
    for i, row in history_risk.iterrows():
        axes[0].annotate(f"n={row['Road-months']:,.0f}", (i, row["Next-month incidence (%)"]),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    neighbor_labels = ["Isolated", "No recent", "Recent"]
    axes[1].bar(neighbor_labels, neighbor_risk["Next-month incidence (%)"],
                color=["#9aa6b2", "#79a78d", "#d78355"])
    axes[1].set_ylabel("Next-month incidence (%)")
    axes[1].set_title("Neighbour history when own history is zero")
    axes[1].set_xlabel("Recent crashes on neighbouring roads")
    for i, row in neighbor_risk.iterrows():
        axes[1].annotate(f"n={row['Road-months']:,.0f}", (i, row["Next-month incidence (%)"]),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Training-only associations, 2010–2012 after 12-month warm-up")
    return fig


def load_history_results(results_root, full_history_root=None):
    model_names = ["mlp", "graphsage", "gatedgcn", "graph_transformer"]
    display_names = ["MLP", "GraphSAGE", "GatedGCN", "Graph Transformer"]
    records = []
    for model, display_name in zip(model_names, display_names):
        for history_mode in ["none", "all"]:
            for task in ["classification", "regression"]:
                task_candidates = (
                    [results_root / f"DE_{model}_{task}_n2v",
                     results_root / f"DE_{model}_{task}_n2v_hist_none"]
                    if history_mode == "none" else
                    [results_root / f"DE_{model}_{task}_n2v_hist_all"]
                )
                if history_mode == "all" and full_history_root is not None:
                    run_dir = full_history_root / f"v1_DE_{model}_{task}_n2v_hist_all"
                    runs = json.loads((run_dir / "metrics.json").read_text())["runs"]
                else:
                    run_dir = next((p for p in task_candidates if p.is_dir() and list(p.glob("seed*_metrics.json"))), None)
                    if run_dir is None:
                        raise FileNotFoundError(f"Missing {history_mode} {task} metrics for {model}: {task_candidates}")
                    runs = [json.loads(path.read_text()) for path in sorted(run_dir.glob("seed*_metrics.json"))]
                metric_rows = []
                for run in runs:
                    test = run["test"]["all"]
                    metric_rows.append({
                        "AUROC": test["AUROC"],
                        "AP": test["PR-AUC"],
                        "MAE": test.get("MAE", np.nan),
                        "Precision@top1%": test["precision@top1%"],
                    })
                values = pd.DataFrame(metric_rows)
                if len(values) != 3:
                    raise ValueError(f"Expected three seeds in {run_dir}; found {len(values)}")
                for metric in values.columns:
                    records.append({
                        "Model": display_name, "Task": task, "History": history_mode,
                        "Metric": metric, "Mean": values[metric].mean(),
                        "SD": values[metric].std(ddof=0), "Seeds": len(values),
                    })
    return pd.DataFrame(records)


def history_comparison_table(history_result_summary, task):
    metrics = ["AUROC", "AP", "Precision@top1%"]
    if task == "regression":
        metrics.insert(2, "MAE")
    display_names = history_result_summary["Model"].unique()
    rows = []
    for model in display_names:
        row = {"Model": model}
        for metric in metrics:
            vals = history_result_summary[
                history_result_summary["Task"].eq(task)
                & history_result_summary["Metric"].eq(metric)
                & history_result_summary["Model"].eq(model)
            ].set_index("History")
            baseline = vals.loc["none"]
            history = vals.loc["all"]
            difference = history["Mean"] - baseline["Mean"]
            row[metric] = (
                f"{baseline['Mean']:.4f} ± {baseline['SD']:.4f} → "
                f"{history['Mean']:.4f} ± {history['SD']:.4f} ({difference:+.4f})"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def history_checkpoints(full_history_root, task):
    metric = "AUROC" if task == "classification" else "MAE"
    checkpoints = []
    for model, label in (("mlp", "MLP"), ("graphsage", "GraphSAGE"),
                         ("gatedgcn", "GatedGCN"), ("graph_transformer", "Graph Transformer")):
        run_dir = full_history_root / f"v1_DE_{model}_{task}_n2v_hist_all"
        runs = json.loads((run_dir / "metrics.json").read_text())["runs"]
        for run in sorted(runs, key=lambda row: row["seed"]):
            seed = run["seed"]
            score = run["valid"]["all"][metric]
            checkpoints.append({"Model": label, "Seed": seed, "Epoch": run["best_epoch"],
                                f"Validation {metric}": score, f"Test {metric}": run["test"]["all"][metric]})
    return pd.DataFrame(checkpoints)
