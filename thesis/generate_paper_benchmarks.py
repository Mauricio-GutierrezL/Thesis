from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from qml_benchmarks.data import (
    generate_bars_and_stripes,
    generate_hidden_manifold_model,
    generate_hyperplanes_parity,
    generate_linearly_separable,
    generate_two_curves,
)
from qml_benchmarks.data.mnist import generate_mnist


def save_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.c_[x, y]
    np.savetxt(path, data, delimiter=",")


def gen_linearly_separable(root: Path) -> None:
    np.random.seed(42)
    out = root / "linearly_separable"
    n_samples = 300

    for n_features in range(2, 21):
        margin = 0.02 * n_features
        x, y = generate_linearly_separable(n_samples, n_features, margin)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"linearly_separable_{n_features}d_train.csv", x_train, y_train)
        save_xy(out / f"linearly_separable_{n_features}d_test.csv", x_test, y_test)


def gen_hidden_manifold(root: Path) -> None:
    np.random.seed(3)
    out = root / "hidden_manifold"
    manifold_dimension = 6
    n_samples = 300

    for n_features in range(2, 21):
        x, y = generate_hidden_manifold_model(n_samples, n_features, manifold_dimension)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"hidden_manifold-{manifold_dimension}manifold-{n_features}d_train.csv", x_train, y_train)
        save_xy(out / f"hidden_manifold-{manifold_dimension}manifold-{n_features}d_test.csv", x_test, y_test)


def gen_hidden_manifold_diff(root: Path) -> None:
    np.random.seed(3)
    out = root / "hidden_manifold_diff"
    n_features = 10
    n_samples = 300

    for manifold_dimension in range(2, 21):
        x, y = generate_hidden_manifold_model(n_samples, n_features, manifold_dimension)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"hidden_manifold-10d-{manifold_dimension}manifold_train.csv", x_train, y_train)
        save_xy(out / f"hidden_manifold-10d-{manifold_dimension}manifold_test.csv", x_test, y_test)


def gen_bars_and_stripes(root: Path) -> None:
    out = root / "bars_and_stripes"
    n_samples_train = 1000
    n_samples_test = 200
    noise_std = 0.5

    for size in [4, 8, 16, 32]:
        np.random.seed(42)
        x_train, y_train = generate_bars_and_stripes(n_samples_train, size, size, noise_std)
        x_test, y_test = generate_bars_and_stripes(n_samples_test, size, size, noise_std)

        save_xy(
            out / f"bars_and_stripes_{size}_x_{size}_{noise_std}noise_train.csv",
            np.reshape(x_train, [n_samples_train, -1]),
            y_train,
        )
        save_xy(
            out / f"bars_and_stripes_{size}_x_{size}_{noise_std}noise_test.csv",
            np.reshape(x_test, [n_samples_test, -1]),
            y_test,
        )


def gen_hyperplanes_diff(root: Path) -> None:
    np.random.seed(1)
    out = root / "hyperplanes_diff"
    n_features = 10
    dim_hyperplanes = 3
    n_samples = 300

    for n_hyperplanes in range(2, 21):
        x, y = generate_hyperplanes_parity(n_samples, n_features, n_hyperplanes, dim_hyperplanes)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_train.csv", x_train, y_train)
        save_xy(out / f"hyperplanes-10d-from{dim_hyperplanes}d-{n_hyperplanes}n_test.csv", x_test, y_test)


def gen_two_curves_diff(root: Path) -> None:
    np.random.seed(3)
    out = root / "two_curves_diff"
    n_samples = 300
    noise = 0.01

    degree = 5
    offset = 0.1
    for n_features in range(2, 21):
        x, y = generate_two_curves(n_samples, n_features, degree, offset, noise)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"two_curves-5degree-0.1offset-{n_features}d_train.csv", x_train, y_train)
        save_xy(out / f"two_curves-5degree-0.1offset-{n_features}d_test.csv", x_test, y_test)

    n_features = 10
    for degree in range(2, 21):
        offset = 1 / (2 * degree)
        x, y = generate_two_curves(n_samples, n_features, degree, offset, noise)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        save_xy(out / f"two_curves-10d-{degree}degree_train.csv", x_train, y_train)
        save_xy(out / f"two_curves-10d-{degree}degree_test.csv", x_test, y_test)


def gen_mnist_pca(root: Path) -> None:
    np.random.seed(42)
    out = root / "mnist_pca"
    digit_a = 3
    digit_b = 5

    for n_features in range(2, 21):
        x_train, x_test, y_train, y_test = generate_mnist(
            digit_a, digit_b, preprocessing="pca", n_features=n_features
        )
        save_xy(out / f"mnist_{digit_a}-{digit_b}_{n_features}d_train.csv", x_train, y_train)
        save_xy(out / f"mnist_{digit_a}-{digit_b}_{n_features}d_test.csv", x_test, y_test)


def gen_mnist_pca_small(root: Path) -> None:
    np.random.seed(42)
    out = root / "mnist_pca-"
    digit_a = 3
    digit_b = 5

    for n_features in range(2, 21):
        x_train, x_test, y_train, y_test = generate_mnist(
            digit_a, digit_b, preprocessing="pca-", n_features=n_features, n_samples=250
        )
        save_xy(out / f"mnist_{digit_a}-{digit_b}_{n_features}d-250_train.csv", x_train, y_train)
        save_xy(out / f"mnist_{digit_a}-{digit_b}_{n_features}d-250_test.csv", x_test, y_test)


def gen_mnist_cg(root: Path) -> None:
    import torch

    torch.manual_seed(42)
    out = root / "mnist_cg"
    digit_a = 3
    digit_b = 5

    for height in [4, 8, 16, 32]:
        x_train, x_test, y_train, y_test = generate_mnist(
            digit_a, digit_b, preprocessing="cg", height=height
        )
        save_xy(out / f"mnist_pixels_{digit_a}-{digit_b}_{height}x{height}_train.csv", x_train, y_train)
        save_xy(out / f"mnist_pixels_{digit_a}-{digit_b}_{height}x{height}_test.csv", x_test, y_test)


GENERATORS = {
    "linearly_separable": gen_linearly_separable,
    "hidden_manifold": gen_hidden_manifold,
    "hidden_manifold_diff": gen_hidden_manifold_diff,
    "bars_and_stripes": gen_bars_and_stripes,
    "hyperplanes_diff": gen_hyperplanes_diff,
    "two_curves_diff": gen_two_curves_diff,
    "mnist_pca": gen_mnist_pca,
    "mnist_pca-": gen_mnist_pca_small,
    "mnist_cg": gen_mnist_cg,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis datasets mirroring the paper benchmark scripts.")
    parser.add_argument(
        "--root",
        default="thesis/datasets_tests",
        help="Target root directory for generated datasets",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(GENERATORS.keys()),
        default=sorted(GENERATORS.keys()),
        help="Benchmark folders to generate",
    )
    args = parser.parse_args()

    root = Path(args.root)
    for name in args.benchmarks:
        print(f"Generating {name}...")
        GENERATORS[name](root)
    print("DONE")


if __name__ == "__main__":
    main()
