"""
Generate publication-ready figures from the paper-level CSV tables.

Source of truth: the CSVs in docs/paper_tables/ (provided tables).
Outputs: PNG + PDF figures (vector) into a chosen outdir (default docs/paper_figures/).

Charts:
- Dataset-wise contribution (questions, papers) as horizontal bars
- Academic level distribution (donut + bar)
- Domain distribution (horizontal bar)

Validation:
- Dataset contribution: sum questions == 13248, sum papers == 2676, percents consistent
- Academic level: sum papers == 1104, sum questions == 13248, percents consistent
- Domain: sum papers == 1104, percents consistent
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
except Exception:
    plt.style.use("default")


EXPECTED_TOTAL_QUESTIONS = 13248
EXPECTED_TOTAL_PAPERS = 1104
EXPECTED_DATASET_PAPERS = 2676  # papers-using-dataset (multi-dataset sum)
PERCENT_TOL = 0.15  # allowable pct-point difference when validating provided percentages


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(outdir: str, name: str) -> None:
    for ext in ("png", "pdf"):  # png for preview, pdf for publication
        plt.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


def validate_sum(series: pd.Series, expected: float, label: str) -> None:
    total = float(series.sum())
    if abs(total - expected) > 1e-6:
        raise ValueError(f"{label} total {total} != expected {expected}")


def validate_percent(series_val: pd.Series, series_pct: pd.Series, expected_total: float, label: str) -> None:
    # Compare provided pct to recomputed pct of expected_total
    recomputed = (series_val / expected_total) * 100.0
    diffs = (series_pct - recomputed).abs()
    if (diffs > PERCENT_TOL).any():
        rows = series_pct.index[diffs > PERCENT_TOL].tolist()
        raise ValueError(f"{label} percent mismatch for rows: {rows}, diffs={diffs[diffs > PERCENT_TOL].tolist()}")


def auto_label_barh(ax, values: Iterable[float], fmt: str = "{:.0f}", offset: float = 0.01) -> None:
    for rect, val in zip(ax.patches, values):
        width = rect.get_width()
        ax.text(
            width + max(width * offset, 0.02 * max(values)),
            rect.get_y() + rect.get_height() / 2,
            fmt.format(val),
            va="center",
            fontsize=9,
        )


def make_dataset_charts(df: pd.DataFrame, outdir: str) -> None:
    df = df.copy()
    df["Dataset"] = df["Dataset"].astype(str)
    df["Question Count"] = df["Question Count"].astype(float)
    df["Papers using Dataset"] = df["Papers using Dataset"].astype(float)
    df["Percentage of Total questions"] = df["Percentage of Total questions"].astype(float)
    df["Percentage of papers"] = df["Percentage of papers"].astype(float)

    # Drop Total row for validation/plotting
    df_plot = df[df["Dataset"].str.lower() != "total"].copy()

    # Validate totals & percentages (using rows without Total)
    validate_sum(df_plot["Question Count"], EXPECTED_TOTAL_QUESTIONS, "Dataset questions")
    validate_sum(df_plot["Papers using Dataset"], EXPECTED_DATASET_PAPERS, "Dataset papers using dataset")
    validate_percent(df_plot["Question Count"], df_plot["Percentage of Total questions"], EXPECTED_TOTAL_QUESTIONS, "Dataset questions pct")
    validate_percent(df_plot["Papers using Dataset"], df_plot["Percentage of papers"], EXPECTED_TOTAL_PAPERS, "Dataset papers pct")

    # Questions by dataset
    q_sorted = df_plot.sort_values("Question Count", ascending=True)
    plt.figure(figsize=(9.5, 6.5))
    ax = plt.gca()
    bars = ax.barh(q_sorted["Dataset"], q_sorted["Question Count"], color="#4C78A8")
    auto_label_barh(ax, q_sorted["Question Count"], fmt="{:.0f}")
    ax.set_xlabel("Number of questions")
    ax.set_ylabel("Dataset")
    ax.set_title("Questions by dataset (provided table)")
    plt.tight_layout()
    savefig(outdir, "dataset_questions_barh")
    plt.close()

    # Papers using dataset
    p_sorted = df_plot.sort_values("Papers using Dataset", ascending=True)
    plt.figure(figsize=(9.5, 6.5))
    ax = plt.gca()
    bars = ax.barh(p_sorted["Dataset"], p_sorted["Papers using Dataset"], color="#E45756")
    auto_label_barh(ax, p_sorted["Papers using Dataset"], fmt="{:.0f}")
    ax.set_xlabel("Number of papers using dataset")
    ax.set_ylabel("Dataset")
    ax.set_title("Papers using dataset (provided table)")
    plt.tight_layout()
    savefig(outdir, "dataset_papers_barh")
    plt.close()


def make_level_charts(df: pd.DataFrame, outdir: str) -> None:
    df = df.copy()
    df["Academic Level"] = df["Academic Level"].astype(str)
    df["No. of Papers"] = df["No. of Papers"].astype(float)
    df["No. of Questions"] = df["No. of Questions"].astype(float)
    df["Percentage"] = df["Percentage"].astype(float)

    df_plot = df[df["Academic Level"].str.lower() != "total"].copy()

    validate_sum(df_plot["No. of Papers"], EXPECTED_TOTAL_PAPERS, "Level papers")
    validate_sum(df_plot["No. of Questions"], EXPECTED_TOTAL_QUESTIONS, "Level questions")
    validate_percent(df_plot["No. of Papers"], df_plot["Percentage"], EXPECTED_TOTAL_PAPERS, "Level pct")
    df_plot = df_plot.sort_values("No. of Papers", ascending=False)

    # Donut
    plt.figure(figsize=(6.2, 4.8))
    wedges, texts, autotexts = plt.pie(
        df_plot["No. of Papers"],
        labels=df_plot["Academic Level"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        textprops={"fontsize": 10},
        colors=["#F58518", "#4C78A8", "#54A24B"][: len(df_plot)],
    )
    centre = plt.Circle((0, 0), 0.58, fc="white")
    plt.gca().add_artist(centre)
    plt.title("Papers by academic level")
    plt.tight_layout()
    savefig(outdir, "level_donut")
    plt.close()

    # Bar
    plt.figure(figsize=(6.2, 4.4))
    ax = plt.gca()
    bars = ax.bar(df_plot["Academic Level"], df_plot["No. of Papers"], color="#4C78A8")
    ax.set_xlabel("Academic level")
    ax.set_ylabel("Number of papers")
    ax.set_title("Papers by academic level")
    for rect, val in zip(bars, df_plot["No. of Papers"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(val * 0.01, 3), f"{int(val)}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    savefig(outdir, "level_bar")
    plt.close()


def make_domain_chart(df: pd.DataFrame, outdir: str) -> None:
    df = df.copy()
    df["Domain"] = df["Domain"].astype(str)
    df["No. of Papers"] = df["No. of Papers"].astype(float)
    df["Percentage"] = df["Percentage"].astype(float)

    df_plot = df[df["Domain"].str.lower() != "total"].copy()
    df_plot = df_plot.sort_values("No. of Papers", ascending=True)

    plt.figure(figsize=(9.5, 8.0))
    ax = plt.gca()
    bars = ax.barh(df_plot["Domain"], df_plot["No. of Papers"], color="#4C78A8")
    auto_label_barh(ax, df_plot["No. of Papers"], fmt="{:.0f}")
    ax.set_xlabel("Number of papers")
    ax.set_ylabel("Domain")
    ax.set_title("Papers by domain")
    plt.tight_layout()
    savefig(outdir, "domain_barh")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from provided CSV tables.")
    parser.add_argument("--tables_dir", default=os.path.join("docs", "paper_tables"), help="Directory containing paper tables CSVs.")
    parser.add_argument("--outdir", default=os.path.join("docs", "paper_figures"), help="Output directory for charts.")
    args = parser.parse_args()

    tables_dir = args.tables_dir
    outdir = args.outdir
    ensure_dir(outdir)

    dataset_csv = os.path.join(tables_dir, "dataset_contribution.csv")
    level_csv = os.path.join(tables_dir, "academic_level_distribution.csv")
    domain_csv = os.path.join(tables_dir, "domain_distribution.csv")

    dataset_df = pd.read_csv(dataset_csv)
    level_df = pd.read_csv(level_csv)
    domain_df = pd.read_csv(domain_csv)

    make_dataset_charts(dataset_df, outdir)
    make_level_charts(level_df, outdir)
    make_domain_chart(domain_df, outdir)

    print(f"Saved charts to: {outdir}")
    print("Tip: use the PDF versions in your paper for sharp vector graphics.")


if __name__ == "__main__":
    main()

