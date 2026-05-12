from collections import OrderedDict
from math import sqrt
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover
    student_t = None


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "my_results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BASE_MODEL = "CircuitCentricClassifier"
HALF_MODEL = "CircuitCentricClassifierHalfSeparableRandom50"
SEP_MODEL = "CircuitCentricClassifierSeparable"

DISPLAY_NAME_MAP = {
    HALF_MODEL: "CircuitCentricClassifierHalfSeparable",
}

DATASETS = OrderedDict(
    [
        (
            "LINEARLY SEPARABLE",
            {
                "folder": "linearly_separable",
                "stem_regex": r"linearly_separable_(\d+)d",
            },
        ),
        (
            "HIDDEN MANIFOLD",
            {
                "folder": "hidden_manifold",
                "stem_regex": r"hidden_manifold-6manifold-(\d+)d",
            },
        ),
        (
            "HIDDEN MANIFOLD DIFF",
            {
                "folder": "hidden_manifold_diff",
                "stem_regex": r"hidden_manifold-10d-(\d+)manifold",
            },
        ),
        (
            "HYPERPLANES DIFF",
            {
                "folder": "hyperplanes_diff",
                "stem_regex": r"hyperplanes-10d-from3d-(\d+)n",
            },
        ),
        (
            "MNIST PCA",
            {
                "folder": "mnist_pca",
                "stem_regex": r"mnist_3-5_(\d+)d",
            },
        ),
        (
            "MNIST PCA-",
            {
                "folder": "mnist_pca-",
                "stem_regex": r"mnist_3-5_(\d+)d-250",
            },
        ),
        (
            "TWO CURVES",
            {
                "folder": "two_curves",
                "stem_regex": r"two_curves-5degree-0\.1offset-(\d+)d",
            },
        ),
        (
            "TWO CURVES DIFF",
            {
                "folder": "two_curves_diff",
                "stem_regex": r"two_curves-10d-(\d+)degree",
            },
        ),
    ]
)

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

sns.set_theme(style="whitegrid", context="talk")


def display_name(model_name: str) -> str:
    return DISPLAY_NAME_MAP.get(model_name, model_name)


def load_plotting_config():
    with open(BASE_DIR / "plotting_standards.yaml", "r") as stream:
        return yaml.safe_load(stream)


def iter_result_files(model_dir: Path):
    for candidate in sorted(model_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
        yield candidate
    results_dir = model_dir / "results"
    if results_dir.exists():
        for candidate in sorted(results_dir.glob("*_GridSearchCV-best-hyperparams-results.csv")):
            yield candidate


def parse_dataset_stem(model_name: str, result_file: Path, stem_regex: str):
    prefix = f"{model_name}_"
    suffix = "_GridSearchCV-best-hyperparams-results.csv"
    name = result_file.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    dataset_stem = name[len(prefix) : -len(suffix)]
    if re.fullmatch(stem_regex, dataset_stem) is None:
        return None
    return dataset_stem


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if student_t is not None:
        return float(student_t.ppf(0.975, df))
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    return 1.96


def collect_ccc_means():
    rows = []
    for family_name, cfg in DATASETS.items():
        dataset_root = RESULTS_DIR / cfg["folder"]
        if not dataset_root.exists():
            continue
        for model_name in [BASE_MODEL, HALF_MODEL, SEP_MODEL]:
            model_dir = dataset_root / model_name
            if not model_dir.exists():
                continue
            for result_file in iter_result_files(model_dir):
                dataset_stem = parse_dataset_stem(model_name, result_file, cfg["stem_regex"])
                if dataset_stem is None:
                    continue
                df = pd.read_csv(result_file)
                if "test_acc" not in df.columns:
                    continue
                rows.append(
                    {
                        "family": family_name,
                        "dataset_stem": dataset_stem,
                        "model": model_name,
                        "mean_test_acc": float(df["test_acc"].mean()),
                    }
                )

    return pd.DataFrame(rows)


def summarize_differences(means_df: pd.DataFrame, variant_model: str):
    summaries = []
    all_diffs = []
    all_base_means = []

    for family_name in DATASETS:
        family_df = means_df[means_df["family"] == family_name]
        if family_df.empty:
            continue

        pivot = family_df.pivot_table(
            index="dataset_stem",
            columns="model",
            values="mean_test_acc",
            aggfunc="mean",
        )
        if BASE_MODEL not in pivot.columns or variant_model not in pivot.columns:
            continue

        paired = pivot[[BASE_MODEL, variant_model]].dropna()
        if paired.empty:
            continue

        diffs = paired[variant_model] - paired[BASE_MODEL]
        base_means = paired[BASE_MODEL]
        summaries.append(make_summary_row(family_name, diffs, base_means))
        all_diffs.extend(diffs.tolist())
        all_base_means.extend(base_means.tolist())

    if all_diffs:
        summaries.insert(0, make_summary_row("ALL BENCHMARKS", pd.Series(all_diffs), pd.Series(all_base_means)))

    return pd.DataFrame(summaries)


def make_summary_row(label: str, diffs: pd.Series, base_means: pd.Series):
    n = int(diffs.shape[0])
    mean_diff = float(diffs.mean())
    ccc_mean = float(base_means.mean())

    if n > 1:
        sd = float(diffs.std(ddof=1))
        se = sd / sqrt(n)
        margin = t_critical_95(n - 1) * se
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
    else:
        sd = float("nan")
        se = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "family": label,
        "n_benchmarks": n,
        "mean_diff": mean_diff,
        "ccc_mean_test_acc": ccc_mean,
        "sd_diff": sd,
        "se_diff": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def plot_forest(summary_df: pd.DataFrame, variant_model: str, plotting_config: dict, output_name: str):
    color = plotting_config["color"][variant_model]
    title_map = {
        HALF_MODEL: "Mean Test Accuracy Difference: CCC Half-Separable vs CCC",
        SEP_MODEL: "Mean Test Accuracy Difference: CCC Separable vs CCC",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2), tight_layout=True)
    y_positions = list(range(len(summary_df)))[::-1]
    means = summary_df["mean_diff"].to_numpy()
    ci_low = summary_df["ci_low"].to_numpy()
    ci_high = summary_df["ci_high"].to_numpy()

    lower_err = means - ci_low
    upper_err = ci_high - means
    lower_err = [0.0 if pd.isna(x) else x for x in lower_err]
    upper_err = [0.0 if pd.isna(x) else x for x in upper_err]

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(
        means,
        y_positions,
        xerr=[lower_err, upper_err],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=2,
        capsize=4,
        markersize=8,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df["family"])
    ax.set_xlabel("Mean test accuracy difference")
    ax.set_title(title_map.get(variant_model, f"{variant_model} - {BASE_MODEL}"))

    x_min = min([0.0] + summary_df["ci_low"].dropna().tolist() + summary_df["mean_diff"].tolist())
    x_max = max([0.0] + summary_df["ci_high"].dropna().tolist() + summary_df["mean_diff"].tolist())
    pad = max(0.01, 0.08 * (x_max - x_min if x_max > x_min else 1.0))
    ax.set_xlim(x_min - pad, x_max + 3.5 * pad)

    for y, (_, row) in zip(y_positions, summary_df.iterrows()):
        ax.text(row["mean_diff"], y + 0.18, f"{row['mean_diff']:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(
            x_max + 1.2 * pad,
            y,
            f"n={int(row['n_benchmarks'])}, CCC mean={row['ccc_mean_test_acc']:.3f}",
            va="center",
            fontsize=10,
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    plotting_config = load_plotting_config()
    means_df = collect_ccc_means()
    if means_df.empty:
        raise RuntimeError(f"No CCC results found in {RESULTS_DIR}")

    half_summary = summarize_differences(means_df, HALF_MODEL)
    sep_summary = summarize_differences(means_df, SEP_MODEL)

    summary_path = FIGURES_DIR / "ccc_forest_summary.csv"
    pd.concat(
        [
            half_summary.assign(comparison=f"{HALF_MODEL} - {BASE_MODEL}"),
            sep_summary.assign(comparison=f"{SEP_MODEL} - {BASE_MODEL}"),
        ],
        ignore_index=True,
    ).assign(
        comparison=lambda df: df["comparison"].replace(
            {
                f"{HALF_MODEL} - {BASE_MODEL}": f"{display_name(HALF_MODEL)} - {display_name(BASE_MODEL)}",
                f"{SEP_MODEL} - {BASE_MODEL}": f"{display_name(SEP_MODEL)} - {display_name(BASE_MODEL)}",
            }
        )
    ).to_csv(summary_path, index=False)

    plot_forest(
        half_summary,
        HALF_MODEL,
        plotting_config,
        "forest-ccc-half-minus-ccc.png",
    )
    plot_forest(
        sep_summary,
        SEP_MODEL,
        plotting_config,
        "forest-ccc-separable-minus-ccc.png",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {FIGURES_DIR / 'forest-ccc-half-minus-ccc.png'}")
    print(f"Wrote {FIGURES_DIR / 'forest-ccc-separable-minus-ccc.png'}")


if __name__ == "__main__":
    main()
