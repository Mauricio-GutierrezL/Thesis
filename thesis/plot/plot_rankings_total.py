# Copyright 2024 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0

"""Plot aggregated CCC rankings from thesis/my_results for all completed benchmark families."""

from collections import Counter
from pathlib import Path
import re
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
        "stem_regex": r"linearly_separable_(\d+)d",
    },
    "hidden-manifold": {
        "folder": "hidden_manifold",
        "stem_regex": r"hidden_manifold-6manifold-(\d+)d",
    },
    "hmm-diff": {
        "folder": "hidden_manifold_diff",
        "stem_regex": r"hidden_manifold-10d-(\d+)manifold",
    },
    "hyperplanes-diff": {
        "folder": "hyperplanes_diff",
        "stem_regex": r"hyperplanes-10d-from3d-(\d+)n",
    },
    "bars-and-stripes": {
        "folder": "bars_and_stripes",
        "stem_regex": r"bars_and_stripes_(\d+)_x_\d+_0\.5noise",
    },
    "mnist-pca": {
        "folder": "mnist_pca",
        "stem_regex": r"mnist_3-5_(\d+)d",
    },
    "mnist-pca-small": {
        "folder": "mnist_pca-",
        "stem_regex": r"mnist_3-5_(\d+)d-250",
    },
    "two-curves-diff": {
        "folder": "two_curves_diff",
        "stem_regex": r"two_curves-10d-(\d+)degree",
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


def iter_result_files(model_dir: Path):
    for candidate in sorted(model_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
        yield candidate
    results_dir = model_dir / "results"
    if results_dir.exists():
        for candidate in sorted(results_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
            yield candidate


def parse_dataset_stem(model_name: str, result_file: Path, dataset_cfg: dict):
    prefix = f"{model_name}_"
    suffix = "_GridSearchCV-best-hyperparams-results.csv"
    name = result_file.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    dataset_stem = name[len(prefix) : -len(suffix)]
    if re.fullmatch(dataset_cfg["stem_regex"], dataset_stem) is None:
        return None
    return dataset_stem


def load_results(models) -> pd.DataFrame:
    frames = []
    for dataset_name, cfg in DATASETS.items():
        dataset_root = RESULTS_DIR / cfg["folder"]
        for model in models:
            model_dir = dataset_root / model
            if not model_dir.exists():
                continue

            for result_file in iter_result_files(model_dir):
                dataset_stem = parse_dataset_stem(model, result_file, cfg)
                if dataset_stem is None:
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

    def ranking_name(base: str) -> str:
        # The thesis figures now only contain the no-First50 comparison set,
        # so keep the output names clean and stable.
        return f"ranking-{base}.png" if suffix == "no-first50" else f"ranking-{base}{suffix_part}.png"

    df_plot = build_ranking_table(df, models)
    save_ranking_plot(df_plot, FIGURES_DIR / ranking_name("all"))

    for dataset_name in DATASETS:
        df_dataset = df[df["dataset_family"] == dataset_name].copy()
        if df_dataset.empty:
            continue
        df_dataset_plot = build_ranking_table(df_dataset, models)
        save_ranking_plot(
            df_dataset_plot,
            FIGURES_DIR / ranking_name(dataset_name),
            title=dataset_name,
        )


def main() -> None:
    # Only emit the no-First50 ranking figures for the thesis comparison set.
    create_ranking_set(MODELS_NO_FIRST50, suffix="no-first50")


if __name__ == "__main__":
    main()
