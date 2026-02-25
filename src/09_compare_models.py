"""
Cross-paper model comparison for 30-day readmission prediction.

Run as:
  python -m src.09_compare_models
or
  python src/09_compare_models.py

Outputs:
  - results/model_comparison.csv
  - results/model_comparison.json
  - results/wins_vs_baseline.csv
  - results/wins_vs_baseline.json
  - results/wins_summary.csv
  - figures/model_comparison_core_metrics.png
  - figures/model_comparison_heatmap.png
  - figures/model_comparison_radar.png
  - figures/model_comparison_wins_matrix.png
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import fix: works as package module and direct script.
try:
    from .config import RESULTS_DIR, FIGURES_DIR
    from .plot_style import apply_publication_style, save_publication_figure
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import RESULTS_DIR, FIGURES_DIR
    from plot_style import apply_publication_style, save_publication_figure


CORE_METRICS: List[str] = [
    "auroc",
    "auprc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
    "mcc",
]

ALL_METRICS: List[str] = CORE_METRICS + [
    "npv",
    "brier_score",
    "log_loss",
]
apply_publication_style()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def load_our_model_metrics() -> Dict[str, float]:
    report_path = os.path.join(RESULTS_DIR, "training_report.json")
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Missing report: {report_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    pred_df = pd.read_csv(pred_path)
    required = {"y_true", "pred"}
    if not required.issubset(set(pred_df.columns)):
        raise ValueError(
            f"test_predictions.csv must contain columns {sorted(required)}, "
            f"found {sorted(pred_df.columns.tolist())}"
        )

    y_true = pred_df["y_true"].astype(int).to_numpy()
    y_pred = pred_df["pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "auroc": float(report.get("auroc_calibrated", float("nan"))),
        "auprc": float(report.get("auprc", float("nan"))),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "mcc": mcc,
        "npv": npv,
        "brier_score": float(report.get("brier_score", float("nan"))),
        "log_loss": float(report.get("log_loss", float("nan"))),
    }


def _load_training_report() -> Dict[str, object]:
    report_path = os.path.join(RESULTS_DIR, "training_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Missing report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _our_operating_point_rows(report: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ops = report.get("operating_points", {}) or {}
    if not isinstance(ops, dict):
        return rows

    auroc = float(report.get("auroc_calibrated", float("nan")))
    auprc = float(report.get("auprc", float("nan")))
    brier = float(report.get("brier_score", float("nan")))
    ll = float(report.get("log_loss", float("nan")))

    for op_name, m in ops.items():
        if not isinstance(m, dict):
            continue
        rows.append(
            {
                "model": f"TRANCE ({op_name} op)",
                "source": "Local run: training_report.json operating_points",
                "threshold": float(m.get("threshold", float("nan"))),
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": float(m.get("accuracy", float("nan"))),
                "f1": float(m.get("f1", float("nan"))),
                "precision": float(m.get("precision", float("nan"))),
                "recall": float(m.get("recall", float("nan"))),
                "specificity": float(m.get("specificity", float("nan"))),
                "mcc": float(m.get("mcc", float("nan"))),
                "brier_score": brier,
                "log_loss": ll,
            }
        )
    return rows


def get_paper_baselines() -> List[Dict[str, object]]:
    """
    Paper baselines taken from the two PDFs provided by the user.
    """
    return [
        {
            "model": "ClinicalT5 + LGBM",
            "source": "PLOS ONE (2025) Table 7",
            "auroc": 0.68,
            "accuracy": 0.63,
            "recall": 0.66,
            "precision": 0.63,
            "specificity": 0.60,
            "f1": 0.64,
            "mcc": 0.26,
        },
        {
            "model": "ClinicalT5 + VotingClassifier",
            "source": "PLOS ONE (2025) Table 7",
            "auroc": 0.68,
            "accuracy": 0.63,
            "recall": 0.68,
            "precision": 0.63,
            "specificity": 0.58,
            "f1": 0.65,
            "mcc": 0.26,
        },
        {
            "model": "ClinicalT5 + XGBClassifier",
            "source": "PLOS ONE (2025) Table 7",
            "auroc": 0.66,
            "accuracy": 0.62,
            "recall": 0.66,
            "precision": 0.62,
            "specificity": 0.58,
            "f1": 0.64,
            "mcc": 0.24,
        },
        {
            "model": "PubMedBERT (Text only)",
            "source": "PLOS ONE (2025) Table 7",
            "auroc": 0.64,
            "accuracy": 0.60,
            "recall": 0.69,
            "precision": 0.60,
            "specificity": 0.50,
            "f1": 0.64,
            "mcc": 0.20,
        },
        {
            "model": "ClinicalT5 + Neural Network",
            "source": "PLOS ONE (2025) Table 7",
            "auroc": 0.62,
            "accuracy": 0.58,
            "recall": 0.76,
            "precision": 0.56,
            "specificity": 0.40,
            "f1": 0.65,
            "mcc": 0.17,
        },
        {
            "model": "MM-STGNN (Internal)",
            "source": "IEEE JBHI (2023) main text / Table II summary",
            "auroc": 0.79,
            "auprc": 0.64,  # AP reported as 64%
            "recall": 0.80,  # at operating point used in paper narrative
            "specificity": 0.60,  # at 80% sensitivity
        },
        {
            "model": "MM-STGNN (MIMIC-IV)",
            "source": "IEEE JBHI (2023) main text / Fig.2 narrative",
            "auroc": 0.79,
            "auprc": 0.64,  # AP reported as 64%
            "recall": 0.80,  # at operating point used in paper narrative
            "specificity": 0.54,  # at 80% sensitivity
        },
        {
            "model": "LACE+ (Internal subset)",
            "source": "IEEE JBHI (2023) main text",
            "auroc": 0.61,
            "recall": 0.81,  # in reported operating point
            "specificity": 0.38,
        },
    ]


def build_comparison_df() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    report = _load_training_report()

    # Our implementation row.
    our_metrics = load_our_model_metrics()
    rows.append(
        {
            "model": "TRANCE (Our Implementation)",
            "source": "Local run: training_report.json + test_predictions.csv",
            "threshold": float(report.get("best_threshold", float("nan"))),
            **our_metrics,
        }
    )
    rows.extend(_our_operating_point_rows(report))

    # Paper baselines.
    rows.extend(get_paper_baselines())

    df = pd.DataFrame(rows)
    if "threshold" not in df.columns:
        df["threshold"] = np.nan
    for metric in ALL_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    ordered_cols = ["model", "source", "threshold"] + ALL_METRICS
    return df[ordered_cols]


def save_tabular_outputs(df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    json_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"Saved comparison table -> {csv_path}")
    print(f"Saved comparison JSON  -> {json_path}")


def build_wins_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare each paper baseline against the best TRANCE variant per metric.
    """
    our_df = df[df["model"].str.startswith("TRANCE (", na=False)].copy()
    base_df = df[~df["model"].str.startswith("TRANCE (", na=False)].copy()

    rows: List[Dict[str, object]] = []
    for _, b in base_df.iterrows():
        for metric in CORE_METRICS:
            b_val = b.get(metric, np.nan)
            if pd.isna(b_val):
                continue
            candidates = our_df[["model", metric]].dropna(subset=[metric]).copy()
            if candidates.empty:
                continue
            best_idx = candidates[metric].astype(float).idxmax()
            best_model = str(candidates.loc[best_idx, "model"])
            our_val = float(candidates.loc[best_idx, metric])
            rows.append(
                {
                    "baseline_model": str(b["model"]),
                    "baseline_source": str(b["source"]),
                    "metric": metric,
                    "our_variant": best_model,
                    "our_value": our_val,
                    "baseline_value": float(b_val),
                    "delta": float(our_val - float(b_val)),
                    "win": bool(our_val > float(b_val)),
                }
            )

    wins_df = pd.DataFrame(rows)
    if wins_df.empty:
        summary_df = pd.DataFrame(columns=["baseline_model", "wins", "compared_metrics", "win_rate"])
        return wins_df, summary_df

    summary_df = (
        wins_df.groupby("baseline_model", as_index=False)
        .agg(
            wins=("win", "sum"),
            compared_metrics=("win", "count"),
        )
    )
    summary_df["win_rate"] = (summary_df["wins"] / summary_df["compared_metrics"]).round(4)
    return wins_df, summary_df


def save_wins_outputs(wins_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    wins_csv = os.path.join(RESULTS_DIR, "wins_vs_baseline.csv")
    wins_json = os.path.join(RESULTS_DIR, "wins_vs_baseline.json")
    summary_csv = os.path.join(RESULTS_DIR, "wins_summary.csv")

    wins_df.to_csv(wins_csv, index=False)
    wins_df.to_json(wins_json, orient="records", indent=2)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved wins table    -> {wins_csv}")
    print(f"Saved wins json     -> {wins_json}")
    print(f"Saved wins summary  -> {summary_csv}")


def _ordered_models(df: pd.DataFrame) -> List[str]:
    our = "TRANCE (Our Implementation)"
    others = [m for m in df["model"].tolist() if m != our]
    return [our] + others if our in set(df["model"]) else df["model"].tolist()


def plot_core_metrics(df: pd.DataFrame) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    core = df[["model"] + CORE_METRICS].copy()
    long_df = core.melt(id_vars="model", var_name="metric", value_name="value")
    long_df = long_df.dropna(subset=["value"])

    if long_df.empty:
        return

    long_df["metric"] = long_df["metric"].str.upper()
    model_order = _ordered_models(df)
    metric_order = [m.upper() for m in CORE_METRICS]

    fig = plt.figure(figsize=(15, 7))
    if HAS_SEABORN:
        sns.barplot(
            data=long_df,
            x="metric",
            y="value",
            hue="model",
            order=metric_order,
            hue_order=model_order,
            errorbar=None,
        )
    else:
        pivot = long_df.pivot_table(index="metric", columns="model", values="value")
        pivot = pivot.reindex(metric_order)
        x = np.arange(len(pivot.index))
        n_models = len(pivot.columns)
        width = 0.8 / max(n_models, 1)
        for i, col in enumerate(pivot.columns):
            plt.bar(x + (i - n_models / 2) * width + width / 2, pivot[col], width, label=col)
        plt.xticks(x, pivot.index, rotation=20)

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title("Model Comparison Across Core Metrics")
    plt.xticks(rotation=20)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    out = os.path.join(FIGURES_DIR, "model_comparison_core_metrics.png")
    save_publication_figure(fig, out)
    print(f"Saved figure -> {out}")


def plot_heatmap(df: pd.DataFrame) -> None:
    score_df = df.set_index("model")[ALL_METRICS].copy()
    # Keep only metrics with at least one non-null.
    score_df = score_df.loc[:, score_df.notna().any(axis=0)]
    if score_df.empty:
        return

    fig = plt.figure(figsize=(12, max(5, int(0.65 * len(score_df.index)))))
    if HAS_SEABORN:
        sns.heatmap(
            score_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Score"},
            vmin=0.0,
            vmax=1.0,
        )
    else:
        plt.imshow(score_df.fillna(0.0).to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Score")
        plt.yticks(np.arange(len(score_df.index)), score_df.index)
        plt.xticks(np.arange(len(score_df.columns)), score_df.columns, rotation=35, ha="right")

    plt.title("Model vs Metric Heatmap (N/A = not reported)")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    out = os.path.join(FIGURES_DIR, "model_comparison_heatmap.png")
    save_publication_figure(fig, out)
    print(f"Saved figure -> {out}")


def plot_radar(df: pd.DataFrame) -> None:
    radar_metrics = [m for m in CORE_METRICS if df[m].notna().sum() >= 2]
    if len(radar_metrics) < 3:
        return

    radar_df = df[["model"] + radar_metrics].copy().set_index("model")
    # Fill missing values with metric-wise mean for shape completeness.
    radar_df = radar_df.apply(lambda col: col.fillna(col.mean()), axis=0)

    labels = [m.upper() for m in radar_metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)

    for model in _ordered_models(df):
        if model not in radar_df.index:
            continue
        values = radar_df.loc[model, radar_metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Radar View of Core Metrics")
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)
    out = os.path.join(FIGURES_DIR, "model_comparison_radar.png")
    save_publication_figure(fig, out)
    print(f"Saved figure -> {out}")


def plot_wins_matrix(wins_df: pd.DataFrame) -> None:
    if wins_df.empty:
        return

    matrix = (
        wins_df.pivot_table(index="baseline_model", columns="metric", values="win", aggfunc="max")
        .fillna(False)
        .astype(int)
    )
    matrix = matrix.reindex(columns=CORE_METRICS, fill_value=0)

    fig = plt.figure(figsize=(11, max(4.5, 0.65 * len(matrix.index))))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No", "Yes"])
    cbar.set_label("TRANCE beats baseline")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([m.upper() for m in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Wins Matrix: TRANCE Variants vs Baselines by Metric")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Baseline Model")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix.iat[i, j])
            ax.text(j, i, "W" if val == 1 else "-", ha="center", va="center", fontsize=9, color="black")

    out = os.path.join(FIGURES_DIR, "model_comparison_wins_matrix.png")
    save_publication_figure(fig, out)
    print(f"Saved figure -> {out}")


def main() -> None:
    df = build_comparison_df()
    # Show concise table in terminal.
    print("\nModel comparison (head):")
    print(df.to_string(index=False))

    save_tabular_outputs(df)
    wins_df, summary_df = build_wins_tables(df)
    save_wins_outputs(wins_df, summary_df)
    plot_core_metrics(df)
    plot_heatmap(df)
    plot_radar(df)
    plot_wins_matrix(wins_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
