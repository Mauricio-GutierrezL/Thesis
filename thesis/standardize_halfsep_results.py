from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


MODEL = "CircuitCentricClassifierHalfSeparable"
GROUP_FOLDERS = {
    "first50": f"{MODEL}First50",
    "random50": f"{MODEL}Random50",
}
DATASETS = ["linearly_separable", "hidden_manifold_diff"]
SOURCE_SUFFIXES = [
    "_GridSearchCV-best-hyperparameters.csv",
    "_GridSearchCV-best-hyperparams.csv",
    "_GridSearchCV-best-hyperparams-results.csv",
    "_GridSearchCV.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def classify_file(name: str) -> str | None:
    token = f"{MODEL}_"
    if name.startswith(token):
        return "first50"
    if token in name:
        return "random50"
    return None


def split_name(name: str) -> tuple[str, str] | None:
    token = f"{MODEL}_"
    if token not in name:
        return None

    tail = name[name.index(token) + len(token):]
    for suffix in SOURCE_SUFFIXES:
        if tail.endswith(suffix):
            dataset_stem = tail[: -len(suffix)]
            canonical_suffix = suffix.replace("best-hyperparameters", "best-hyperparams")
            return dataset_stem, canonical_suffix
    return None


def move_file(src: Path, dst: Path, execute: bool) -> str:
    if dst.exists():
        if sha256(src) == sha256(dst):
            if execute:
                src.unlink()
            return "duplicate_removed"
        return "conflict"

    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return "moved"


def clean_source_dir(source_dir: Path, execute: bool) -> None:
    desktop_ini = source_dir / "desktop.ini"
    if desktop_ini.exists() and execute:
        desktop_ini.unlink()

    if execute:
        try:
            source_dir.rmdir()
        except OSError:
            pass


def standardize(root: Path, execute: bool) -> None:
    moved = 0
    duplicates = 0
    conflicts: list[tuple[Path, Path]] = []

    for dataset in DATASETS:
        source_dir = root / dataset / MODEL
        if not source_dir.exists():
            continue

        for src in sorted(source_dir.iterdir()):
            if not src.is_file() or src.name == "desktop.ini":
                continue

            group = classify_file(src.name)
            parsed = split_name(src.name)
            if group is None or parsed is None:
                continue

            dataset_stem, suffix = parsed
            dst_dir = root / dataset / GROUP_FOLDERS[group]
            dst_name = f"{GROUP_FOLDERS[group]}_{dataset_stem}{suffix}"
            dst = dst_dir / dst_name

            result = move_file(src, dst, execute)
            if result == "moved":
                moved += 1
                print(f"MOVE {src} -> {dst}")
            elif result == "duplicate_removed":
                duplicates += 1
                print(f"DROP duplicate {src} (kept {dst})")
            else:
                conflicts.append((src, dst))
                print(f"CONFLICT {src} -> {dst}")

        clean_source_dir(source_dir, execute)

    print(f"\nSummary: moved={moved}, duplicates_removed={duplicates}, conflicts={len(conflicts)}")
    if conflicts:
        print("Conflicts:")
        for src, dst in conflicts:
            print(f"  {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split HalfSeparable thesis results into clean first50/random50 folders.")
    parser.add_argument("--root", default="thesis/my_results", help="Root results directory")
    parser.add_argument("--execute", action="store_true", help="Apply the changes")
    args = parser.parse_args()

    standardize(Path(args.root), execute=args.execute)


if __name__ == "__main__":
    main()
