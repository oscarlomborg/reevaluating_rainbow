
"""
check_experiment_nans.py

Scan experiment .npz files for NaN or infinite numeric values.

By default it scans:
    ./experiments

It reports:
- file path
- array/key inside the .npz file
- shape and dtype
- number of NaN / +inf / -inf values
- indices of problematic values (limited for readability)

Run:
    python check_experiment_nans.py

Optional:
    python check_experiment_nans.py --root experiments
    python check_experiment_nans.py --root experiments --max-indices 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def check_numeric_array(array: np.ndarray):
    """Return counts and indices of NaN/inf values in a numeric array."""
    if not np.issubdtype(array.dtype, np.number):
        return None

    # Integer/bool arrays cannot contain NaN or inf.
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        return {
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "bad_indices": [],
        }

    nan_mask = np.isnan(array)
    posinf_mask = np.isposinf(array)
    neginf_mask = np.isneginf(array)

    bad_mask = nan_mask | posinf_mask | neginf_mask
    bad_indices = np.argwhere(bad_mask)

    return {
        "nan_count": int(nan_mask.sum()),
        "posinf_count": int(posinf_mask.sum()),
        "neginf_count": int(neginf_mask.sum()),
        "bad_indices": bad_indices,
    }


def inspect_object_array(array: np.ndarray, max_indices: int):
    """
    Inspect object arrays recursively when elements contain numeric arrays.

    Useful for fields such as raw_returns, which may be stored as dtype=object.
    """
    issues = []

    for index, value in np.ndenumerate(array):
        value_array = np.asarray(value)

        result = check_numeric_array(value_array)

        if result is None:
            continue

        total_bad = (
            result["nan_count"]
            + result["posinf_count"]
            + result["neginf_count"]
        )

        if total_bad == 0:
            continue

        shown = result["bad_indices"][:max_indices]

        issues.append({
            "object_index": index,
            "shape": value_array.shape,
            "dtype": str(value_array.dtype),
            "nan_count": result["nan_count"],
            "posinf_count": result["posinf_count"],
            "neginf_count": result["neginf_count"],
            "bad_indices": shown.tolist(),
        })

    return issues


def inspect_npz(path: Path, max_indices: int):
    """Inspect one .npz file and return all detected issues."""
    issues = []

    try:
        # allow_pickle=True is needed for arrays such as raw_returns.
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                array = np.asarray(data[key])

                # Regular numeric array.
                result = check_numeric_array(array)

                if result is not None:
                    total_bad = (
                        result["nan_count"]
                        + result["posinf_count"]
                        + result["neginf_count"]
                    )

                    if total_bad > 0:
                        shown = result["bad_indices"][:max_indices]

                        issues.append({
                            "key": key,
                            "shape": array.shape,
                            "dtype": str(array.dtype),
                            "nan_count": result["nan_count"],
                            "posinf_count": result["posinf_count"],
                            "neginf_count": result["neginf_count"],
                            "bad_indices": shown.tolist(),
                            "object_details": None,
                        })

                    continue

                # Object array, e.g. arrays of per-evaluation episode returns.
                if array.dtype == object:
                    object_issues = inspect_object_array(
                        array,
                        max_indices=max_indices,
                    )

                    if object_issues:
                        issues.append({
                            "key": key,
                            "shape": array.shape,
                            "dtype": str(array.dtype),
                            "nan_count": None,
                            "posinf_count": None,
                            "neginf_count": None,
                            "bad_indices": None,
                            "object_details": object_issues,
                        })

    except Exception as exc:
        return [{
            "load_error": f"{type(exc).__name__}: {exc}"
        }]

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./experiments"),
        help="Experiment directory to scan (default: ./experiments)",
    )
    parser.add_argument(
        "--max-indices",
        type=int,
        default=10,
        help="Maximum bad indices to print per array (default: 10)",
    )
    args = parser.parse_args()

    root = args.root

    if not root.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")

    files = sorted(root.rglob("*.npz"))

    if not files:
        print(f"No .npz files found under: {root.resolve()}")
        return

    print(f"Scanning {len(files)} .npz files under:")
    print(root.resolve())
    print()

    files_with_issues = 0
    arrays_with_issues = 0
    total_nan = 0
    total_posinf = 0
    total_neginf = 0
    load_errors = 0

    for path in files:
        issues = inspect_npz(
            path,
            max_indices=args.max_indices,
        )

        if not issues:
            continue

        files_with_issues += 1

        print("=" * 80)
        print(f"FILE: {path}")

        for issue in issues:
            if "load_error" in issue:
                load_errors += 1
                print(f"  [LOAD ERROR] {issue['load_error']}")
                continue

            arrays_with_issues += 1

            print(f"  KEY:   {issue['key']}")
            print(f"  SHAPE: {issue['shape']}")
            print(f"  DTYPE: {issue['dtype']}")

            if issue["object_details"] is None:
                nan_count = issue["nan_count"]
                posinf_count = issue["posinf_count"]
                neginf_count = issue["neginf_count"]

                total_nan += nan_count
                total_posinf += posinf_count
                total_neginf += neginf_count

                print(f"  NaN:   {nan_count}")
                print(f"  +inf:  {posinf_count}")
                print(f"  -inf:  {neginf_count}")
                print(f"  bad indices: {issue['bad_indices']}")

            else:
                print("  Object-array entries containing NaN/inf:")

                for detail in issue["object_details"]:
                    total_nan += detail["nan_count"]
                    total_posinf += detail["posinf_count"]
                    total_neginf += detail["neginf_count"]

                    print(
                        f"    object index {detail['object_index']}: "
                        f"shape={detail['shape']}, dtype={detail['dtype']}, "
                        f"NaN={detail['nan_count']}, "
                        f"+inf={detail['posinf_count']}, "
                        f"-inf={detail['neginf_count']}"
                    )
                    print(
                        f"      bad indices: {detail['bad_indices']}"
                    )

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files scanned:          {len(files)}")
    print(f"Files with issues:      {files_with_issues}")
    print(f"Arrays with issues:     {arrays_with_issues}")
    print(f"Total NaN values:       {total_nan}")
    print(f"Total +inf values:      {total_posinf}")
    print(f"Total -inf values:      {total_neginf}")
    print(f"Files with load errors: {load_errors}")

    if files_with_issues == 0:
        print("\n[PASS] No NaN or infinite values found.")
    else:
        print("\n[WARNING] Non-finite values were found. See details above.")


if __name__ == "__main__":
    main()
