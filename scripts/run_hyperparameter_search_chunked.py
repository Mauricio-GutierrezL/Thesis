#!/usr/bin/env python3
"""Run one hyperparameter-search combination and merge chunked results."""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid

from qml_benchmarks.hyperparam_search_utils import (
    construct_hyperparameter_grid,
    read_data,
)
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings
from qml_benchmarks.models.base import BaseGenerator

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)


def canonical_dataset_stem(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    if stem.endswith("_train"):
        return stem[:-6]
    if stem.endswith("_test"):
        return stem[:-5]
    return stem


def custom_scorer(estimator, X, y=None):
    return estimator.score(X, y)


def bool_arg(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes"}


def recompute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if not column.startswith("mean_test_"):
            continue
        metric = column[len("mean_test_") :]
        rank_column = f"rank_test_{metric}"
        df[rank_column] = (
            df[column].rank(method="min", ascending=False, na_option="bottom").astype(int)
        )
    return df


def write_best_hyperparams(df: pd.DataFrame, refit_metric: str, path: Path) -> None:
    rank_column = f"rank_test_{refit_metric}"
    mean_column = f"mean_test_{refit_metric}"
    if rank_column in df.columns:
        best_row = df.sort_values([rank_column, mean_column], ascending=[True, False]).iloc[0]
    else:
        best_row = df.sort_values(mean_column, ascending=False).iloc[0]

    rows = []
    for column in df.columns:
        if not column.startswith("param_"):
            continue
        rows.append(
            {
                "hyperparameter": column[len("param_") :],
                "best_value": best_row[column],
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one hyperparameter-search combination and merge chunked results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--results-path", default=".")
    parser.add_argument("--combo-index", type=int, required=True)
    parser.add_argument(
        "--hyperparameter-scoring",
        type=str,
        nargs="+",
        default=["accuracy", "roc_auc"],
    )
    parser.add_argument("--hyperparameter-refit", type=str, default="accuracy")
    parser.add_argument("--plot-loss", type=bool_arg, default=False)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--pre-dispatch", type=str, default=None)
    parser.add_argument("--clean", type=bool_arg, default=False)
    args = parser.parse_args()

    hyperparam_grid = construct_hyperparameter_grid(hyper_parameter_settings, args.model)
    combos = list(ParameterGrid(hyperparam_grid))
    if args.combo_index < 0 or args.combo_index >= len(combos):
        raise ValueError(
            f"combo index {args.combo_index} out of range for {args.model} "
            f"(0..{len(combos) - 1})"
        )

    Model = getattr(import_module("qml_benchmarks.models"), args.model)
    model_name = Model.__name__
    is_generative = isinstance(Model(), BaseGenerator)
    use_labels = not is_generative

    train_dataset_filename = os.path.join(args.dataset_path)
    X, y = read_data(train_dataset_filename, labels=use_labels)

    dataset_stem = canonical_dataset_stem(args.dataset_path)
    results_filename_stem = f"{model_name}_{dataset_stem}_GridSearchCV"
    results_root = Path(args.results_path) / "results"
    chunk_root = results_root / "_chunked_hps" / results_filename_stem
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_root / f"combo_{args.combo_index:03d}.csv"

    if chunk_path.exists() and not args.clean:
        logging.info("Chunk already exists: %s", chunk_path)
    else:
        model = Model()
        default_score = None
        if y is not None:
            model.fit(X, y)
            default_score = model.score(X, y)
        else:
            model.fit(X)
            default_score = model.score(X)

        logging.info("Chunk %s/%s", args.combo_index + 1, len(combos))
        logging.info("Default score: %s", default_score)
        if hasattr(model, "loss_history_") and args.plot_loss:
            plt.plot(model.loss_history_)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()

        if hasattr(model, "n_qubits_"):
            logging.info("Num qubits %s", model.n_qubits_)

        scorer = args.hyperparameter_scoring if not is_generative else custom_scorer
        refit = args.hyperparameter_refit if not is_generative else False
        pre_dispatch = args.pre_dispatch
        if pre_dispatch is None and args.n_jobs > 0:
            pre_dispatch = str(args.n_jobs)

        selected_combo = {key: [value] for key, value in combos[args.combo_index].items()}
        gs = GridSearchCV(
            estimator=Model(),
            param_grid=selected_combo,
            scoring=scorer,
            refit=refit,
            verbose=3,
            n_jobs=args.n_jobs,
            pre_dispatch=pre_dispatch,
        ).fit(X, y)

        df = pd.DataFrame.from_dict(gs.cv_results_)
        df.to_csv(chunk_path, index=False)
        logging.info("Wrote chunk result to %s", chunk_path)

    chunk_paths = [chunk_root / f"combo_{index:03d}.csv" for index in range(len(combos))]
    if not all(path.exists() for path in chunk_paths):
        missing = sum(not path.exists() for path in chunk_paths)
        logging.info("Chunks still missing for %s: %s", results_filename_stem, missing)
        return

    merged = pd.concat((pd.read_csv(path) for path in chunk_paths), ignore_index=True)
    merged = recompute_ranks(merged)
    final_results_path = results_root / f"{results_filename_stem}.csv"
    final_hps_path = results_root / f"{results_filename_stem}-best-hyperparams.csv"
    merged.to_csv(final_results_path, index=False)
    write_best_hyperparams(merged, args.hyperparameter_refit, final_hps_path)
    logging.info("Merged %s chunk files into %s", len(chunk_paths), final_results_path)


if __name__ == "__main__":
    main()
