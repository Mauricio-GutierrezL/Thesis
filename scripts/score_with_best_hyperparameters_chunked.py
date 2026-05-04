#!/usr/bin/env python3
"""Score one random seed and merge chunked score results."""

import argparse
import inspect
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from importlib import import_module

from qml_benchmarks.hyperparam_search_utils import csv_to_dict, read_data

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)


def canonical_dataset_stem(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    if stem.endswith("_train"):
        return stem[:-6]
    if stem.endswith("_test"):
        return stem[:-5]
    return stem


def bool_arg(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes"}


def coerce_hyperparams_for_classifier(classifier_cls, hyperparams):
    coerced = dict(hyperparams)
    signature = inspect.signature(classifier_cls.__init__)

    for name, value in list(coerced.items()):
        param = signature.parameters.get(name)
        if param is None or param.default is inspect._empty:
            continue

        default = param.default
        if isinstance(default, bool):
            if isinstance(value, str):
                coerced[name] = value.lower() in {"1", "true", "yes"}
            else:
                coerced[name] = bool(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            coerced[name] = int(value)
        elif isinstance(default, float):
            coerced[name] = float(value)

    return coerced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one random seed and merge chunked score results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--classifier-name", required=True)
    parser.add_argument("--trainset-path", required=True)
    parser.add_argument("--testset-path", required=True)
    parser.add_argument("--hyperparams-path", required=True)
    parser.add_argument("--results-path", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--clean", type=bool_arg, default=False)
    args = parser.parse_args()

    results_path = Path(args.results_path) / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    Classifier = getattr(import_module("qml_benchmarks.models"), args.classifier_name)
    classifier_name = Classifier.__name__

    X_train, y_train = read_data(os.path.join(args.trainset_path))
    X_test, y_test = read_data(os.path.join(args.testset_path))

    dataset_stem = canonical_dataset_stem(args.trainset_path)
    results_filename_stem = f"{classifier_name}_{dataset_stem}_GridSearchCV"
    final_results_path = results_path / f"{results_filename_stem}-best-hyperparams-results.csv"
    chunk_root = results_path / "_chunked_scores" / results_filename_stem
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_root / f"seed_{args.seed}.csv"

    if chunk_path.exists() and not args.clean:
        logging.info("Seed result already exists: %s", chunk_path)
    else:
        best_hyperparams = coerce_hyperparams_for_classifier(
            Classifier,
            csv_to_dict(args.hyperparams_path),
        )
        classifier = Classifier(**best_hyperparams, random_state=args.seed)
        classifier.fit(X_train, y_train)
        result = pd.DataFrame.from_dict(
            {
                "seed": [args.seed],
                "train_acc": [classifier.score(X_train, y_train)],
                "test_acc": [classifier.score(X_test, y_test)],
            }
        )
        result.to_csv(chunk_path, index=False)
        logging.info("Wrote chunk score to %s", chunk_path)

    chunk_paths = [chunk_root / f"seed_{seed}.csv" for seed in range(args.num_seeds)]
    if not all(path.exists() for path in chunk_paths):
        missing = sum(not path.exists() for path in chunk_paths)
        logging.info("Seed chunks still missing for %s: %s", results_filename_stem, missing)
        return

    merged = pd.concat((pd.read_csv(path) for path in chunk_paths), ignore_index=True)
    merged = merged.sort_values("seed").reset_index(drop=True)
    merged.to_csv(final_results_path, index=False)
    logging.info("Merged %s seed chunks into %s", len(chunk_paths), final_results_path)


if __name__ == "__main__":
    main()
