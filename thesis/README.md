## Thesis Workspace Layout

This folder is the working area for the thesis-specific experiments built on top of the upstream benchmark code.

### Folder roles

- `datasets_tests/`
  Thesis datasets used for training and evaluation.
  Current datasets include `linearly_separable/`, `hidden_manifold/`, and `hidden_manifold_diff/`.

- `my_results/`
  Thesis experiment outputs.
  The intended structure is:
  `thesis/my_results/<dataset>/<model>/<csv files>`

- `plot/`
  Thesis plotting scripts and generated figures.
  These scripts should read from `thesis/my_results/` and write to `thesis/plot/figures/`.

- `runs/`
  Collected run artifacts for thesis work.
  This includes both batch scripts and logs from different rounds of experimentation.

### Source Of Truth

- Upstream reference material lives in `paper/`.
- Thesis-specific code changes currently live in `src/qml_benchmarks/models/circuit_centric.py`.
- Thesis-specific datasets, results, logs, and figures live under `thesis/`.
- Thesis job and log history live under `thesis/runs/`.

### Current CCC Variants

- `CircuitCentricClassifier`
  Original circuit-centric classifier with the default strongly entangling ansatz.

- `CircuitCentricClassifierHalfSeparable`
  Controlled ansatz with partial entanglement.

- `CircuitCentricClassifierSeparable`
  Controlled ansatz with no entangling gates in the variational part.

### Naming Conventions

- Training datasets:
  `.../<dataset_name>_train.csv`

- Test datasets:
  `.../<dataset_name>_test.csv`

- Hyperparameter search outputs:
  `<Model>_<dataset_name>_GridSearchCV.csv`

- Best hyperparameters:
  `<Model>_<dataset_name>_GridSearchCV-best-hyperparams.csv`

- Evaluation with best hyperparameters:
  `<Model>_<dataset_name>_GridSearchCV-best-hyperparams-results.csv`

### Practical Rule

If a file belongs to the original paper reproduction, keep it under `paper/`.
If a file belongs to the thesis comparison between CCC variants, keep it under `thesis/`.
