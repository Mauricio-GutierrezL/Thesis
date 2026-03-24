# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to reproduce the decision boundaries of selected models
on the 2d linearly separable dataset.
"""

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from qml_benchmarks import models
from qml_benchmarks.hyperparam_search_utils import read_data, csv_to_dict


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TRAIN_PATH = (
    PROJECT_ROOT
    / "thesis"
    / "datasets_tests"
    / "linearly_separable"
    / "linearly_separable_2d_train.csv"
)
TEST_PATH = (
    PROJECT_ROOT
    / "thesis"
    / "datasets_tests"
    / "linearly_separable"
    / "linearly_separable_2d_test.csv"
)


def find_hyperparams_file(project_root: Path, clf_name: str) -> Path:
    """
    Find the correct best-hyperparameters csv for the 2d linearly separable dataset.
    Important: the correct files are named with 'best-hyperparameters', not
    'best-hyperparams-results'.
    """
    folder = (
        project_root
        / "thesis"
        / "my_results"
        / "linearly_separable"
        / clf_name
    )

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    # First try exact expected names
    exact_candidates = [
        folder / f"{clf_name}_linearly_separable_2d_GridSearchCV-best-hyperparameters.csv",
        folder / f"{clf_name}_linearly_separable_2d_train_GridSearchCV-best-hyperparameters.csv",
    ]

    for candidate in exact_candidates:
        if candidate.exists():
            print(f"[FOUND hyperparams] {candidate}")
            return candidate

    # Fallback: match any 2d best-hyperparameters file
    matches = sorted(
        folder.glob(f"{clf_name}*linearly_separable*2d*best-hyperparameters.csv")
    )
    if matches:
        print(f"[FOUND hyperparams] {matches[0]}")
        return matches[0]

    print(f"\nNo matching best-hyperparameters file found in: {folder}")
    print("Files in folder:")
    for p in sorted(folder.iterdir()):
        print("  ", p.name)

    raise FileNotFoundError(
        f"No best-hyperparameters file found for {clf_name} in {folder}"
    )


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------
sns.set(rc={"figure.figsize": (6, 6)})
sns.set(font_scale=1.3)
sns.set_style("white")

palette = sns.color_palette("deep")
point_cmap = ListedColormap(palette)
boundary_cmap = sns.diverging_palette(30, 255, l=60, as_cmap=True)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
print("Reading training data from:", TRAIN_PATH)
print("Reading test data from:", TEST_PATH)

X_train, y_train = read_data(str(TRAIN_PATH))
X_test, y_test = read_data(str(TEST_PATH))


# -------------------------------------------------------------------
# Models to plot
# -------------------------------------------------------------------
models_to_plot = [
    "CircuitCentricClassifier",
    "CircuitCentricClassifierHalfSeparable",
    "CircuitCentricClassifierSeparable",
]


# -------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes = axes.flatten()

for ax, clf_name in zip(axes, models_to_plot):
    hyperparams_path = find_hyperparams_file(PROJECT_ROOT, clf_name)
    best_hyperparams = csv_to_dict(str(hyperparams_path))

    print(f"\nUsing hyperparameters for {clf_name}:")
    print(best_hyperparams)

    clf_class = getattr(models, clf_name)
    clf = clf_class(**best_hyperparams)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        cmap=boundary_cmap,
        alpha=0.8,
        ax=ax,
        eps=0.5,
    )

    # Training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=point_cmap,
        marker="o",
        edgecolors="k",
        label="train",
    )

    # Test points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=point_cmap,
        edgecolors="k",
        marker="^",
        alpha=1.0,
        label="test",
    )

    ax.set_title(f"{clf_name}\nacc = {score:.3f}")

# Hide unused 4th subplot
for ax in axes[len(models_to_plot):]:
    ax.axis("off")

plt.tight_layout()
out_file = FIGURES_DIR / "2d-linsep-decisionboundaries.png"
plt.savefig(out_file, bbox_inches="tight")
print("\nSaved figure to:", out_file)
plt.show()