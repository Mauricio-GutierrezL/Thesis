# Copyright 2024 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0

"""Plot CCC train/test accuracies from thesis/my_results for all completed benchmark families."""

from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import yaml


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR.parent / "my_results"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set(rc={"figure.figsize": (8, 4)})
sns.set(font_scale=1.3)
sns.set_style("white")

DATASETS = {
    "LINEARLY SEPARABLE": {
        "folder": "linearly_separable",
        "xlabel": "number of features",
        "stem_regex": r"linearly_separable_(\d+)d",
    },
    "HIDDEN MANIFOLD": {
        "folder": "hidden_manifold",
        "xlabel": "number of features",
        "stem_regex": r"hidden_manifold-6manifold-(\d+)d",
    },
    "HIDDEN MANIFOLD DIFF": {
        "folder": "hidden_manifold_diff",
        "xlabel": "number of manifolds",
        "stem_regex": r"hidden_manifold-10d-(\d+)manifold",
    },
    "HYPERPLANES DIFF": {
        "folder": "hyperplanes_diff",
        "xlabel": "number of hyperplanes",
        "stem_regex": r"hyperplanes-10d-from3d-(\d+)n",
    },
    "BARS & STRIPES": {
        "folder": "bars_and_stripes",
        "xlabel": "grid width",
        "stem_regex": r"bars_and_stripes_(\d+)_x_\d+_0\.5noise",
    },
    "MNIST PCA": {
        "folder": "mnist_pca",
        "xlabel": "number of features",
        "stem_regex": r"mnist_3-5_(\d+)d",
    },
    "MNIST PCA-": {
        "folder": "mnist_pca-",
        "xlabel": "number of features",
        "stem_regex": r"mnist_3-5_(\d+)d-250",
    },
    "TWO CURVES DIFF": {
        "folder": "two_curves_diff",
        "xlabel": "degree",
        "stem_regex": r"two_curves-10d-(\d+)degree",
    },
}

MODEL_ORDER = [
    "CircuitCentricClassifier",
    "CircuitCentricClassifierHalfSeparableRandom50",
    "CircuitCentricClassifierSeparable",
]


def load_plotting_config():
    with open(BASE_DIR / "plotting_standards.yaml", "r") as stream:
        return yaml.safe_load(stream)


def model_style(model_name, plotting_config, fallback_index):
    base_alias = {
        "CircuitCentricClassifierHalfSeparableFirst50": "CircuitCentricClassifierHalfSeparable",
        "CircuitCentricClassifierHalfSeparableRandom50": "CircuitCentricClassifierHalfSeparable",
    }.get(model_name, model_name)

    fallback_palette = sns.color_palette("tab10", n_colors=max(len(MODEL_ORDER), 4))
    fallback_markers = ["o", "s", "^", "D", "P", "X"]
    fallback_dashes = [(), (4, 2), (1, 2), (2, 2)]

    color_map = plotting_config.get("color", {})
    marker_map = plotting_config.get("marker", {})
    dashes_map = plotting_config.get("dashes", {})

    color = color_map.get(model_name, color_map.get(base_alias, fallback_palette[fallback_index % len(fallback_palette)]))
    marker = marker_map.get(model_name, marker_map.get(base_alias, fallback_markers[fallback_index % len(fallback_markers)]))
    dashes_raw = dashes_map.get(model_name, dashes_map.get(base_alias, str(fallback_dashes[fallback_index % len(fallback_dashes)])))
    dashes = eval(dashes_raw) if isinstance(dashes_raw, str) else dashes_raw

    if model_name.endswith("First50"):
        marker = "<"
        dashes = (4, 2)
    elif model_name.endswith("Random50"):
        marker = ">"
        dashes = (2, 2)

    return color, marker, dashes


def iter_result_files(model_dir: Path):
    for candidate in sorted(model_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
        yield candidate
    results_dir = model_dir / "results"
    if results_dir.exists():
        for candidate in sorted(results_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
            yield candidate


def parse_dataset_value(model_name: str, result_file: Path, dataset_cfg: dict):
    prefix = f"{model_name}_"
    suffix = "_GridSearchCV-best-hyperparams-results.csv"
    name = result_file.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None, None
    dataset_stem = name[len(prefix) : -len(suffix)]
    match = re.fullmatch(dataset_cfg["stem_regex"], dataset_stem)
    if not match:
        return None, None
    return dataset_stem, int(match.group(1))


def collect_dataset_frame(dataset_cfg, plotting_config):
    dataset_root = RESULTS_DIR / dataset_cfg["folder"]
    if not dataset_root.exists():
        return pd.DataFrame(), []

    model_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    ordered_models = [m for m in MODEL_ORDER if (dataset_root / m).is_dir()]
    ordered_models.extend([p.name for p in model_dirs if p.name not in ordered_models])

    frames = []
    for model_index, model_name in enumerate(ordered_models):
        model_dir = dataset_root / model_name
        for result_file in iter_result_files(model_dir):
            dataset_stem, value = parse_dataset_value(model_name, result_file, dataset_cfg)
            if dataset_stem is None:
                continue

            df_new = pd.read_csv(result_file)
            if "train_acc" not in df_new.columns or "test_acc" not in df_new.columns:
                continue

            color, marker, dashes = model_style(model_name, plotting_config, model_index)
            df_new = df_new.copy()
            df_new["Model"] = model_name
            df_new["n"] = value
            df_new["dataset_stem"] = dataset_stem
            df_new["color"] = color
            df_new["marker"] = str(marker)
            df_new["dashes"] = str(dashes)
            frames.append(df_new)

    if not frames:
        return pd.DataFrame(), ordered_models

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["n", "Model"]).reset_index(drop=True)
    return df, ordered_models


def plot_dataset(dataset_name, dataset_cfg, plotting_config):
    df, model_order = collect_dataset_frame(dataset_cfg, plotting_config)
    if df.empty:
        print(f"No data found for {dataset_name}")
        return

    palette = {}
    markers = {}
    dashes = {}
    for idx, model_name in enumerate(model_order):
        color, marker, dash = model_style(model_name, plotting_config, idx)
        palette[model_name] = color
        markers[model_name] = marker
        dashes[model_name] = dash

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, tight_layout=True)
    axes[0].set_title("train")
    axes[1].set_title("test")

    sns.lineplot(
        ax=axes[0],
        data=df,
        x="n",
        y="train_acc",
        hue="Model",
        style="Model",
        hue_order=model_order,
        style_order=model_order,
        palette=palette,
        markers=markers,
        dashes=dashes,
    )

    sns.lineplot(
        ax=axes[1],
        data=df,
        x="n",
        y="test_acc",
        hue="Model",
        style="Model",
        hue_order=model_order,
        style_order=model_order,
        palette=palette,
        markers=markers,
        dashes=dashes,
    )

    sns.despine()

    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylim((0.45, 1.05))
        axis.set_ylabel("accuracy")
        axis.set_xlabel(dataset_cfg["xlabel"])
        axis.grid(axis="y")

    fig.suptitle(dataset_name, fontsize=15, y=0.9)

    handles, labels = axes[1].get_legend_handles_labels()
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    legend = fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        frameon=False,
    )

    out_name = dataset_cfg["folder"]
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(
        FIGURES_DIR / f"score-{out_name}-qnn.png",
        dpi=200,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / f'score-{out_name}-qnn.png'}")



def main():
    plotting_config = load_plotting_config()
    for dataset_name, dataset_cfg in DATASETS.items():
        plot_dataset(dataset_name, dataset_cfg, plotting_config)


if __name__ == "__main__":
    main()
