# Copyright 2024 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0

"""Plot thesis aggregated rankings from thesis/my_results."""

from collections import Counter
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "my_results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set(font_scale=1.3)
sns.set_style("white")
cmap = sns.diverging_palette(30, 255, l=60, as_cmap=True)

DATASETS = {
    "linearly-separable": {
        "folder": "linearly_separable",
        "values": range(2, 21),
        "stem": lambda n: f"linearly_separable_{n}d",
    },
    "hmm-diff": {
        "folder": "hidden_manifold_diff",
        "values": range(2, 21),
        "stem": lambda n: f"hidden_manifold-10d-{n}manifold",
    },
}

MODELS = [
    "CircuitCentricClassifier",
    "CircuitCentricClassifierHalfSeparableFirst50",
    "CircuitCentricClassifierHalfSeparableRandom50",
    "CircuitCentricClassifierSeparable",
]

MODELS_NO_FIRST50 = [
    model for model in MODELS if model != "CircuitCentricClassifierHalfSeparableFirst50"
]


def load_results(models) -> pd.DataFrame:
    frames = []
    for dataset_name, cfg in DATASETS.items():
        dataset_root = RESULTS_DIR / cfg["folder"]
        for model in models:
            model_dir = dataset_root / model
            if not model_dir.exists():
                continue

            for value in cfg["values"]:
                dataset_stem = cfg["stem"](value)
                result_file = model_dir / f"{model}_{dataset_stem}_GridSearchCV-best-hyperparams-results.csv"
                if not result_file.exists():
                    continue

                df_new = pd.read_csv(result_file)
                if "test_acc" not in df_new.columns:
                    continue

                df_new = df_new.copy()
                df_new["Model"] = model
                df_new["dataset"] = dataset_stem
                df_new["dataset_family"] = dataset_name
                frames.append(df_new)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_ranking_table(df: pd.DataFrame, models) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "Model"], as_index=False).mean(numeric_only=True)
    grouped["rank"] = grouped.groupby("dataset")["test_acc"].rank(method="min", ascending=False)
    grouped["rank_pct"] = grouped.groupby("dataset")["test_acc"].rank(method="min", ascending=False, pct=True)

    stats = {}
    order_score = {}
    for model in models:
        ranks = grouped[grouped["Model"] == model]["rank"].tolist()
        rank_pcts = grouped[grouped["Model"] == model]["rank_pct"].tolist()
        counts = dict(Counter(ranks))

        stats[model] = {f"rank {i}": counts.get(i, 0) for i in range(1, len(models) + 1)}
        order_score[model] = np.mean(rank_pcts) if rank_pcts else np.nan

    df_plot = pd.DataFrame(stats).transpose()
    df_plot = df_plot[df_plot.columns[::-1]]
    df_plot["average_pct"] = df_plot.index.map(lambda model: order_score[model])
    df_plot = df_plot.sort_values(axis=0, by="average_pct", ascending=False)
    return df_plot.drop(columns=["average_pct"])


def save_ranking_plot(df_plot: pd.DataFrame, outpath: Path, title=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    df_plot.plot.barh(
        ax=ax,
        stacked=True,
        cmap=cmap,
        width=0.8,
        legend=False,
        edgecolor="None",
    )

    sns.despine()
    ax.set_xlabel("number of rankings")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def create_ranking_set(models, suffix: str) -> None:
    df = load_results(models)
    if df.empty:
        print(f"No thesis ranking data found for suffix '{suffix}'.")
        return

    suffix_part = f"-{suffix}" if suffix else ""

    df_plot = build_ranking_table(df, models)
    save_ranking_plot(df_plot, FIGURES_DIR / f"ranking-all{suffix_part}.png")

    for dataset_name in DATASETS:
        df_dataset = df[df["dataset_family"] == dataset_name].copy()
        if df_dataset.empty:
            continue
        df_dataset_plot = build_ranking_table(df_dataset, models)
        save_ranking_plot(
            df_dataset_plot,
            FIGURES_DIR / f"ranking-{dataset_name}{suffix_part}.png",
            title=dataset_name,
        )


def main() -> None:
    create_ranking_set(MODELS, suffix="")
    create_ranking_set(MODELS_NO_FIRST50, suffix="no-first50")


if __name__ == "__main__":
    main()
