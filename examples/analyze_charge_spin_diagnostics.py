from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DIAGNOSTIC_KEYS = (
    "force_bias_norm_charge_pivot_mean",
    "force_bias_norm_charge_pivot_max",
    "force_bias_norm_spin_pivot_mean",
    "force_bias_norm_spin_pivot_max",
    "force_bias_norm_per_pivot_charge_pivot_mean",
    "force_bias_norm_per_pivot_charge_pivot_max",
    "force_bias_norm_per_pivot_spin_pivot_mean",
    "force_bias_norm_per_pivot_spin_pivot_max",
    "field_shift_norm_charge_pivot_mean",
    "field_shift_norm_charge_pivot_max",
    "field_shift_norm_spin_pivot_mean",
    "field_shift_norm_spin_pivot_max",
    "field_shift_norm_per_pivot_charge_pivot_mean",
    "field_shift_norm_per_pivot_charge_pivot_max",
    "field_shift_norm_per_pivot_spin_pivot_mean",
    "field_shift_norm_per_pivot_spin_pivot_max",
    "shifted_field_norm_charge_pivot_mean",
    "shifted_field_norm_charge_pivot_max",
    "shifted_field_norm_spin_pivot_mean",
    "shifted_field_norm_spin_pivot_max",
    "shifted_field_norm_per_pivot_charge_pivot_mean",
    "shifted_field_norm_per_pivot_charge_pivot_max",
    "shifted_field_norm_per_pivot_spin_pivot_mean",
    "shifted_field_norm_per_pivot_spin_pivot_max",
    "field_phase_abs_charge_pivot_mean",
    "field_phase_abs_charge_pivot_max",
    "field_phase_abs_spin_pivot_mean",
    "field_phase_abs_spin_pivot_max",
    "spin_pivot_field_shift_cap",
    "spin_pivot_field_shift_cap_n_applied",
    "spin_pivot_field_shift_cap_fraction",
    "spin_pivot_field_shift_cap_scale_min",
    "spin_pivot_field_shift_cap_excess_mean",
    "spin_pivot_field_shift_cap_excess_max",
    "field_shift_uncapped_norm_spin_pivot_mean",
    "field_shift_uncapped_norm_spin_pivot_max",
    "field_shift_uncapped_norm_per_pivot_spin_pivot_mean",
    "field_shift_uncapped_norm_per_pivot_spin_pivot_max",
    "spin_pivot_field_shift_scale",
    "spin_pivot_reference_drift",
    "spin_pivot_reference_drift_norm",
    "spin_pivot_reference_drift_norm_per_pivot",
    "n_floor",
    "n_nonfinite",
    "n_imp_cap",
    "n_weight_cap",
    "n_node_encounters",
    "abs_ratio_mean",
    "abs_ratio_max",
    "imp_raw_mean",
    "imp_raw_min",
    "imp_raw_max",
)

PAIR_PREFIXES = (
    "force_bias_norm",
    "force_bias_norm_per_pivot",
    "field_shift_norm",
    "field_shift_norm_per_pivot",
    "shifted_field_norm",
    "shifted_field_norm_per_pivot",
    "field_phase_abs",
)

EVENT_KEYS = (
    "n_floor",
    "n_nonfinite",
    "n_imp_cap",
    "n_weight_cap",
    "n_node_encounters",
)

CONTEXT_KEYS = EVENT_KEYS + (
    "spin_pivot_field_shift_cap_fraction",
    "spin_pivot_field_shift_cap_scale_min",
    "spin_pivot_field_shift_cap_excess_mean",
    "spin_pivot_field_shift_cap_excess_max",
    "field_shift_uncapped_norm_spin_pivot_mean",
    "field_shift_uncapped_norm_spin_pivot_max",
    "spin_pivot_field_shift_scale",
    "spin_pivot_reference_drift",
    "spin_pivot_reference_drift_norm",
    "spin_pivot_reference_drift_norm_per_pivot",
    "abs_ratio_mean",
    "abs_ratio_max",
    "imp_raw_mean",
    "imp_raw_min",
    "imp_raw_max",
)

OUTLIER_KEYS = (
    "force_bias_norm_spin_pivot_max",
    "field_shift_norm_spin_pivot_max",
)


def _load_diagnostics(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    files = sorted(directory.expanduser().glob(f"{prefix}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No diagnostic files matching {prefix}_*.npz in {directory}")

    chunks: dict[str, list[np.ndarray]] = {}
    for file in files:
        with np.load(file) as data:
            for key in DIAGNOSTIC_KEYS:
                if key in data:
                    chunks.setdefault(key, []).append(
                        np.asarray(data[key], dtype=float).reshape(-1)
                    )

    if not chunks:
        raise ValueError(
            "No charge/spin pivot diagnostic keys were found. Run with a charge_spin "
            "Hamiltonian and write states with make_block_state_logger(...)."
        )

    return {key: np.concatenate(values) for key, values in chunks.items()}


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0.0)


def _write_csv(data: dict[str, np.ndarray], out_path: Path, *, top: int) -> None:
    rows = []
    for key in sorted(data):
        row = {"metric": key, **_stats(data[key])}
        rows.append(row)

    for prefix in PAIR_PREFIXES:
        for suffix in ("mean", "max"):
            c_key = f"{prefix}_charge_pivot_{suffix}"
            s_key = f"{prefix}_spin_pivot_{suffix}"
            if c_key in data and s_key in data:
                ratio_key = f"{prefix}_spin_over_charge_{suffix}"
                rows.append({"metric": ratio_key, **_stats(_safe_ratio(data[s_key], data[c_key]))})

    for outlier_key in OUTLIER_KEYS:
        if outlier_key not in data:
            continue
        idx = _finite_top_indices(np.asarray(data[outlier_key], dtype=float), top)
        if idx.size == 0:
            continue
        for key in CONTEXT_KEYS:
            if key in data:
                rows.append(
                    {
                        "metric": f"outlier_context/{outlier_key}/{key}",
                        **_stats(np.asarray(data[key], dtype=float)[idx]),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("metric", "mean", "median", "p95", "max"))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(data: dict[str, np.ndarray], top: int) -> None:
    n_steps = max(values.shape[0] for values in data.values())
    print(f"Loaded {n_steps} diagnostic steps.")

    for prefix in PAIR_PREFIXES:
        c_key = f"{prefix}_charge_pivot_mean"
        s_key = f"{prefix}_spin_pivot_mean"
        if c_key not in data or s_key not in data:
            continue

        c_stats = _stats(data[c_key])
        s_stats = _stats(data[s_key])
        ratio = _safe_ratio(data[s_key], data[c_key])
        r_stats = _stats(ratio)
        print(
            f"{prefix}: charge_mean={c_stats['mean']:.6g} "
            f"spin_mean={s_stats['mean']:.6g} "
            f"spin/charge_mean={r_stats['mean']:.6g} "
            f"spin/charge_p95={r_stats['p95']:.6g}"
        )

    _print_outlier_context(data, top=top)

    score_key = "field_phase_abs_spin_pivot_max"
    if score_key in data:
        scores = np.asarray(data[score_key])
        finite_idx = np.flatnonzero(np.isfinite(scores))
        if finite_idx.size:
            order = finite_idx[np.argsort(scores[finite_idx])[-top:]][::-1]
            print(f"Top {min(top, order.size)} spin-pivot phase steps:")
            for idx in order:
                print(f"  step={idx} {score_key}={scores[idx]:.6g}")


def _finite_top_indices(values: np.ndarray, top: int) -> np.ndarray:
    if top <= 0:
        return np.zeros((0,), dtype=int)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return np.zeros((0,), dtype=int)
    return finite_idx[np.argsort(values[finite_idx])[-top:]][::-1]


def _print_outlier_context(data: dict[str, np.ndarray], top: int) -> None:
    for outlier_key in OUTLIER_KEYS:
        if outlier_key not in data:
            continue
        scores = np.asarray(data[outlier_key], dtype=float)
        idx = _finite_top_indices(scores, top)
        if idx.size == 0:
            continue

        print(f"Top {idx.size} {outlier_key} outlier context:")
        print(f"  outlier steps: {', '.join(str(int(i)) for i in idx)}")
        for key in CONTEXT_KEYS:
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=float)
            out_values = values[idx]
            all_stats = _stats(values)
            out_stats = _stats(out_values)
            if key in EVENT_KEYS:
                print(
                    f"  {key}: outlier_sum={np.nansum(out_values):.6g} "
                    f"outlier_mean={out_stats['mean']:.6g} all_mean={all_stats['mean']:.6g}"
                )
            else:
                print(
                    f"  {key}: outlier_mean={out_stats['mean']:.6g} "
                    f"outlier_max={out_stats['max']:.6g} all_mean={all_stats['mean']:.6g}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize charge/spin pivot diagnostics from Trot block-state .npz files."
    )
    parser.add_argument(
        "directory", type=Path, help="Directory containing block_state_*.npz files."
    )
    parser.add_argument("--prefix", default="block_state")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--top", type=int, default=10, help="Number of worst phase steps to print.")
    args = parser.parse_args()

    data = _load_diagnostics(args.directory, args.prefix)
    _print_report(data, top=max(args.top, 0))

    if args.out is not None:
        _write_csv(data, args.out, top=max(args.top, 0))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
