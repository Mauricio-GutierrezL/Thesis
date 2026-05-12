"""Create an overview figure of the thesis benchmark families.

The figure uses representative train/test CSVs from ``thesis/datasets_tests``
and shows either the first two features directly or a 2D PCA projection for
high-dimensional datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


DATASETS_ROOT = ROOT / "thesis" / "datasets_tests"
FIGURES_DIR = ROOT / "thesis" / "plot" / "figures"
PANELS_DIR = FIGURES_DIR / "benchmark_examples_panels"
SNIPPET_PATH = ROOT / "thesis" / "Latex" / "figures" / "benchmark_examples_snippet.tex"

OUTPUT_PNG = FIGURES_DIR / "benchmark_examples.png"
OUTPUT_PDF = FIGURES_DIR / "benchmark_examples.pdf"

CLASS_COLORS = {-1: "#1f4e79", 1: "#b44b37"}
SPLIT_STYLES = {
    "train": {"marker": "o", "alpha": 0.8, "linewidths": 0.45, "s": 22},
    "test": {"marker": "^", "alpha": 0.9, "linewidths": 0.55, "s": 28},
}

CAPTION = (
    "Illustrative examples of datasets created by the different "
    "data-generation procedures. For the scatter plots, the two classes are "
    "shown in blue and orange, and training points are shown as circles while "
    "test points are shown as triangles. High-dimensional benchmark families "
    "are displayed through two-dimensional PCA projections, while the MNIST "
    "families are shown in three dimensions together with example images of "
    "digits 3 and 5. The figure includes representative panels for the "
    "linearly separable, hyperplanes diff, hidden manifold, hidden manifold "
    "diff, two curves, two curves diff, MNIST PCA, and MNIST PCA small "
    "benchmarks."
)

MNIST_CACHE_CANDIDATES = [
    ROOT / "mnist_original" / "MNIST" / "raw",
    Path.home() / ".keras" / "datasets" / "mnist.npz",
]

NON_MNIST_PANEL_CONFIGS = [
    {
        "title": "Linearly Separable",
        "folder": "linearly_separable",
        "stem": "linearly_separable_2d",
        "projection_note": "",
    },
    {
        "title": "Hyperplanes Diff",
        "folder": "hyperplanes_diff",
        "stem": "hyperplanes-10d-from3d-6n",
        "projection_note": "Illustrative PCA projection",
    },
    {
        "title": "Two Curves Diff",
        "folder": "two_curves_diff",
        "stem": "two_curves-10d-12degree",
        "projection_note": "PCA projection",
    },
    {
        "title": "Hidden Manifold",
        "folder": "hidden_manifold",
        "stem": "hidden_manifold-6manifold-10d",
        "projection_note": "PCA projection",
    },
    {
        "title": "Two Curves",
        "folder": "two_curves",
        "stem": "two_curves-5degree-0.1offset-2d",
        "projection_note": "",
    },
    {
        "title": "Hidden Manifold Diff",
        "folder": "hidden_manifold_diff",
        "stem": "hidden_manifold-10d-2manifold",
        "projection_note": "PCA projection",
    },
]

MNIST_PANEL_CONFIGS = [
    {
        "title": "MNIST PCA 3D",
        "folder": "mnist_pca",
        "stem": "mnist_3-5_3d",
        "projection_note": "",
        "force_3d": True,
    },
    {
        "title": "MNIST PCA Small 3D",
        "folder": "mnist_pca-",
        "stem": "mnist_3-5_3d-250",
        "projection_note": "",
        "force_3d": True,
    },
]

PANEL_FILENAME_MAP = {
    "Linearly Separable": "linearly_separable",
    "Hyperplanes Diff": "hyperplanes_diff",
    "Two Curves Diff": "two_curves_diff",
    "Hidden Manifold": "hidden_manifold",
    "Two Curves": "two_curves",
    "Hidden Manifold Diff": "hidden_manifold_diff",
    "MNIST PCA 3D": "mnist_pca_3d",
    "MNIST PCA Small 3D": "mnist_pca_small_3d",
}


def infer_feature_and_label_columns(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a 2D array with at least two columns, got shape {data.shape}")

    last_col = data[:, -1]
    unique_values = np.unique(last_col)
    if unique_values.size <= 10:
        return data[:, :-1], last_col.astype(int)

    first_col = data[:, 0]
    unique_values = np.unique(first_col)
    if unique_values.size <= 10:
        return data[:, 1:], first_col.astype(int)

    raise ValueError("Could not infer the label column automatically.")


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data)
    return infer_feature_and_label_columns(data)


def compute_pca_projection(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_x = np.vstack([train_x, test_x])
    mean = np.mean(all_x, axis=0, keepdims=True)
    centered = all_x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    projected = centered @ components
    return projected[: len(train_x)], projected[len(train_x) :]


def project_to_2d(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    if train_x.shape[1] >= 2 and test_x.shape[1] >= 2 and train_x.shape[1] == 2:
        return train_x[:, :2], test_x[:, :2], "Feature 1", "Feature 2"

    train_proj, test_proj = compute_pca_projection(train_x, test_x)
    return train_proj, test_proj, "PC 1", "PC 2"


def choose_existing_stem(folder: Path, preferred_stem: str) -> str:
    train_path = folder / f"{preferred_stem}_train.csv"
    test_path = folder / f"{preferred_stem}_test.csv"
    if train_path.exists() and test_path.exists():
        return preferred_stem

    candidates = sorted(path.name.removesuffix("_train.csv") for path in folder.glob("*_train.csv"))
    if not candidates:
        raise FileNotFoundError(f"No train CSV files found in {folder}")
    return candidates[0]


def add_3d_panel(
    ax: plt.Axes,
    panel_cfg: dict[str, str],
    train_plot: np.ndarray,
    test_plot: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> None:
    for split_name, plot_x, plot_y in (
        ("train", train_plot, train_y),
        ("test", test_plot, test_y),
    ):
        style = SPLIT_STYLES[split_name]
        for class_label in sorted(np.unique(plot_y)):
            mask = plot_y == class_label
            ax.scatter(
                plot_x[mask, 0],
                plot_x[mask, 1],
                plot_x[mask, 2],
                c=CLASS_COLORS[int(class_label)],
                edgecolors="black",
                marker=style["marker"],
                s=style["s"],
                alpha=style["alpha"],
                linewidths=style["linewidths"],
                depthshade=False,
            )

    ax.set_title(panel_cfg["title"], fontsize=11, color="black")
    ax.set_xlabel("Feature 1", color="black", labelpad=4)
    ax.set_ylabel("Feature 2", color="black", labelpad=4)
    ax.set_zlabel("Feature 3", color="black", labelpad=4)
    ax.tick_params(axis="both", colors="black", labelsize=7, pad=0)
    ax.tick_params(axis="z", colors="black", labelsize=7, pad=0)
    ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("black")
    ax.grid(False)


def add_panel(ax: plt.Axes, panel_cfg: dict[str, str]) -> None:
    folder = DATASETS_ROOT / panel_cfg["folder"]
    stem = choose_existing_stem(folder, panel_cfg["stem"])
    train_x, train_y = load_xy(folder / f"{stem}_train.csv")
    test_x, test_y = load_xy(folder / f"{stem}_test.csv")

    if panel_cfg.get("force_3d") and train_x.shape[1] >= 3 and test_x.shape[1] >= 3:
        add_3d_panel(ax, panel_cfg, train_x[:, :3], test_x[:, :3], train_y, test_y)
        return

    train_plot, test_plot, xlabel, ylabel = project_to_2d(train_x, test_x)

    for split_name, plot_x, plot_y in (
        ("train", train_plot, train_y),
        ("test", test_plot, test_y),
    ):
        style = SPLIT_STYLES[split_name]
        for class_label in sorted(np.unique(plot_y)):
            mask = plot_y == class_label
            ax.scatter(
                plot_x[mask, 0],
                plot_x[mask, 1],
                c=CLASS_COLORS[int(class_label)],
                edgecolors="black",
                marker=style["marker"],
                s=style["s"],
                alpha=style["alpha"],
                linewidths=style["linewidths"],
            )

    title = panel_cfg["title"]
    if panel_cfg["projection_note"]:
        title = f"{title}\n({panel_cfg['projection_note']})"
    title_fontsize = 9
    if panel_cfg["title"] in {"Hidden Manifold", "Hidden Manifold Diff"}:
        title_fontsize = 8
    ax.set_title(title, fontsize=title_fontsize, color="black", pad=8)
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel(ylabel, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", colors="black", labelsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")


def add_mnist_examples(
    ax_top: plt.Axes, ax_bottom: plt.Axes, title_prefix: str, sample_offset: int
) -> None:
    try:
        train_images, train_labels = load_local_mnist_examples()
    except Exception as exc:
        print(f"Skipping MNIST thumbnails: {exc}")
        ax_top.axis("off")
        ax_bottom.axis("off")
        return

    examples = []
    for digit in (3, 5):
        matches = np.where(train_labels == digit)[0]
        if len(matches) <= sample_offset:
            continue
        examples.append((digit, train_images[matches[sample_offset]]))

    if len(examples) != 2:
        ax_top.axis("off")
        ax_bottom.axis("off")
        return

    for axis, (digit, image) in zip((ax_top, ax_bottom), examples):
        axis.imshow(image, cmap="gray_r", interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{title_prefix} digit {digit}", fontsize=8.5, pad=3, color="black")
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)


def load_local_mnist_examples() -> tuple[np.ndarray, np.ndarray]:
    for candidate in MNIST_CACHE_CANDIDATES:
        if candidate.is_dir():
            return load_mnist_from_raw_dir(candidate)
        if candidate.is_file():
            return load_mnist_from_npz(candidate)
    raise FileNotFoundError("No local MNIST cache found.")


def load_mnist_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["x_train"], data["y_train"]


def load_mnist_from_raw_dir(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images = read_idx_images(raw_dir / "train-images-idx3-ubyte")
    labels = read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    return images, labels


def read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(16), dtype=">u4")
        _, count, rows, cols = header
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(8), dtype=">u4")
        _, count = header
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data[:count]


def add_legend_panel(ax: plt.Axes) -> None:
    ax.axis("off")
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_COLORS[-1],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label="Class -1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_COLORS[1],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label="Class +1",
        ),
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            marker=SPLIT_STYLES["train"]["marker"],
            linestyle="",
            color="black",
            markersize=8,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            marker=SPLIT_STYLES["test"]["marker"],
            linestyle="",
            color="black",
            markersize=8,
            label="Test",
        ),
    ]
    ax.legend(
        handles=class_handles + split_handles,
        loc="upper center",
        frameon=False,
        fontsize=10,
        handletextpad=0.8,
        labelspacing=0.9,
        borderaxespad=0.2,
    )
    ax.text(
        0.5,
        0.06,
        "High-dimensional families use 2D projections.\nMNIST is shown in 3D.",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        transform=ax.transAxes,
    )


def export_legend_panel(output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 1.6))
    add_legend_panel(ax)
    style_figure(fig, [ax])
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.92)
    fig.savefig(PANELS_DIR / f"{output_name}.pdf")
    plt.close(fig)


def export_panel(panel_cfg: dict[str, str], output_stem: str) -> None:
    is_3d = panel_cfg.get("force_3d", False)
    if is_3d:
        fig = plt.figure(figsize=(4.2, 3.1))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(4.2, 3.1))
    add_panel(ax, panel_cfg)
    style_figure(fig, [ax])
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90)
    fig.savefig(PANELS_DIR / f"{output_stem}.pdf")
    plt.close(fig)


def export_mnist_digit_image(title_prefix: str, digit: int, sample_offset: int, output_name: str) -> None:
    train_images, train_labels = load_local_mnist_examples()
    matches = np.where(train_labels == digit)[0]
    if len(matches) <= sample_offset:
        raise ValueError(f"Not enough MNIST samples for digit {digit} with offset {sample_offset}")

    fig, ax = plt.subplots(figsize=(1.35, 1.75))
    ax.imshow(train_images[matches[sample_offset]], cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title_prefix} digit {digit}", fontsize=8.5, pad=3, color="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    style_figure(fig, [ax])
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.88)
    fig.savefig(PANELS_DIR / f"{output_name}.png", dpi=300)
    plt.close(fig)


def write_latex_snippet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snippet = rf"""\begin{{figure}}[H]
    \centering
    \makebox[\textwidth][c]{{
    \begin{{minipage}}[t]{{0.66\textwidth}}
        \centering
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/linearly_separable.pdf}}\hfill
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/hyperplanes_diff.pdf}}

        \vspace{{0.8em}}
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/two_curves_diff.pdf}}\hfill
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/hidden_manifold.pdf}}

        \vspace{{0.8em}}
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/two_curves.pdf}}\hfill
        \includegraphics[width=0.49\linewidth]{{../plot/figures/benchmark_examples_panels/hidden_manifold_diff.pdf}}
    \end{{minipage}}\hspace{{0.012\textwidth}}
    \begin{{minipage}}[t]{{0.50\textwidth}}
        \centering
        \begin{{tabular}}[t]{{@{{}}c@{{\hspace{{0.012\linewidth}}}}c@{{\hspace{{0.008\linewidth}}}}c@{{}}}}
            \includegraphics[width=0.72\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_3d.pdf}} &
            \includegraphics[width=0.12\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_digit_3.png}} &
            \includegraphics[width=0.12\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_digit_5.png}} \\
            [0.8em]
            \includegraphics[width=0.72\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_small_3d.pdf}} &
            \includegraphics[width=0.12\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_small_digit_3.png}} &
            \includegraphics[width=0.12\linewidth]{{../plot/figures/benchmark_examples_panels/mnist_pca_small_digit_5.png}}
        \end{{tabular}}

        \vspace{{0.5em}}
        \includegraphics[width=0.86\linewidth]{{../plot/figures/benchmark_examples_panels/legend.pdf}}
    \end{{minipage}}
    }}
    \caption{{{CAPTION}}}
    \label{{fig:benchmark-examples}}
\end{{figure}}
"""
    path.write_text(snippet, encoding="ascii")


def style_figure(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.2, 10.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.08)

    left_grid = outer[0, 0].subgridspec(3, 2, wspace=0.16, hspace=0.34)
    left_axes = [fig.add_subplot(left_grid[row, col]) for row in range(3) for col in range(2)]
    for ax, panel_cfg in zip(left_axes, NON_MNIST_PANEL_CONFIGS):
        add_panel(ax, panel_cfg)

    right_grid = outer[0, 1].subgridspec(
        3,
        3,
        width_ratios=[1.2, 0.42, 0.42],
        height_ratios=[1.0, 1.0, 0.38],
        wspace=0.08,
        hspace=0.18,
    )
    mnist_axes = [
        fig.add_subplot(right_grid[0, 0], projection="3d"),
        fig.add_subplot(right_grid[1, 0], projection="3d"),
    ]
    for ax, panel_cfg in zip(mnist_axes, MNIST_PANEL_CONFIGS):
        add_panel(ax, panel_cfg)

    digit_top_ax_1 = fig.add_subplot(right_grid[0, 1])
    digit_top_ax_2 = fig.add_subplot(right_grid[0, 2])
    add_mnist_examples(digit_top_ax_1, digit_top_ax_2, "PCA", sample_offset=0)

    digit_bottom_ax_1 = fig.add_subplot(right_grid[1, 1])
    digit_bottom_ax_2 = fig.add_subplot(right_grid[1, 2])
    add_mnist_examples(digit_bottom_ax_1, digit_bottom_ax_2, "PCA-", sample_offset=1)

    legend_ax = fig.add_subplot(right_grid[2, :])
    add_legend_panel(legend_ax)

    all_axes = left_axes + mnist_axes + [digit_top_ax_1, digit_top_ax_2, digit_bottom_ax_1, digit_bottom_ax_2, legend_ax]
    style_figure(fig, all_axes)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.08, top=0.97)
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

    for panel_cfg in NON_MNIST_PANEL_CONFIGS + MNIST_PANEL_CONFIGS:
        export_panel(panel_cfg, PANEL_FILENAME_MAP[panel_cfg["title"]])

    export_mnist_digit_image("PCA", 3, 0, "mnist_pca_digit_3")
    export_mnist_digit_image("PCA", 5, 0, "mnist_pca_digit_5")
    export_mnist_digit_image("PCA-", 3, 1, "mnist_pca_small_digit_3")
    export_mnist_digit_image("PCA-", 5, 1, "mnist_pca_small_digit_5")
    export_legend_panel("legend")

    write_latex_snippet(SNIPPET_PATH)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Wrote {SNIPPET_PATH}")


if __name__ == "__main__":
    main()
