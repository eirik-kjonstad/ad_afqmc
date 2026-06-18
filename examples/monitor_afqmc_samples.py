#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


COLUMNS = (
    "phase",
    "block",
    "total_blocks",
    "energy",
    "weight",
    "nodes",
    "elapsed_s",
    "dt_s_per_block",
    "mean_energy",
    "stderr_energy",
)

DEFAULT_REFERENCE_ENERGY = -116.5718


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a running trot AFQMC samples file.")
    parser.add_argument("samples", type=Path, help="Path written by --samples-raw.")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep polling the file and update the plot while the AFQMC run is active.",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Show only the last N production blocks. Default 0 shows all blocks.",
    )
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=DEFAULT_REFERENCE_ENERGY,
        help="Draw this reference energy as a horizontal dashed line. Default: %(default)s Ha.",
    )
    parser.add_argument(
        "--no-reference-energy",
        action="store_true",
        help="Do not draw a reference-energy line.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Save the current plot to this file.")
    return parser.parse_args()


def read_samples(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    rows = []
    if not path.exists():
        return np.zeros((0,), dtype=[("phase", "U3"), *[(name, float) for name in COLUMNS[1:]]])

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(COLUMNS):
                continue
            try:
                rows.append(
                    (
                        parts[0],
                        int(parts[1]),
                        int(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        int(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                        float(parts[8]),
                        float(parts[9]),
                    )
                )
            except ValueError:
                continue

    dtype = [
        ("phase", "U3"),
        ("block", int),
        ("total_blocks", int),
        ("energy", float),
        ("weight", float),
        ("nodes", int),
        ("elapsed_s", float),
        ("dt_s_per_block", float),
        ("mean_energy", float),
        ("stderr_energy", float),
    ]
    return np.asarray(rows, dtype=dtype)


def weighted_running_mean(energy: np.ndarray, weight: np.ndarray) -> np.ndarray:
    denom = np.cumsum(weight)
    numer = np.cumsum(energy * weight)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom != 0.0)


def update_plot(ax, data: np.ndarray, path: Path, *, window: int, reference_energy: float | None) -> None:
    ax.clear()
    ax.set_title(str(path))
    ax.set_xlabel("block")
    ax.set_ylabel("energy / Ha")
    ax.grid(True, alpha=0.25)
    if reference_energy is not None:
        ax.axhline(
            reference_energy,
            color="0.2",
            linestyle="--",
            linewidth=1.4,
            label=f"reference E = {reference_energy:.4f} Ha",
        )

    if data.size == 0:
        ax.text(0.5, 0.5, "waiting for samples...", ha="center", va="center", transform=ax.transAxes)
        return

    eql = data[data["phase"] == "eql"]
    blk = data[data["phase"] == "blk"]

    if eql.size:
        ax.plot(eql["block"], eql["energy"], "o-", ms=3, alpha=0.55, label="equilibration")

    if blk.size:
        if window > 0:
            blk = blk[-window:]
        ax.plot(blk["block"], blk["energy"], ".", ms=4, alpha=0.45, label="production block")
        finite_mean = np.isfinite(blk["mean_energy"])
        if np.any(finite_mean):
            ax.plot(blk["block"][finite_mean], blk["mean_energy"][finite_mean], "-", lw=2, label="running mean")
            finite_err = finite_mean & np.isfinite(blk["stderr_energy"])
            if np.any(finite_err):
                x = blk["block"][finite_err]
                y = blk["mean_energy"][finite_err]
                err = blk["stderr_energy"][finite_err]
                ax.fill_between(x, y - err, y + err, alpha=0.18, label="running stderr")
        else:
            ax.plot(
                blk["block"],
                weighted_running_mean(blk["energy"], blk["weight"]),
                "-",
                lw=2,
                label="weighted running mean",
            )

    last = data[-1]
    ax.text(
        0.01,
        0.99,
        f"last: {last['phase']} {int(last['block'])}/{int(last['total_blocks'])}, "
        f"E={last['energy']:.10f}, W={last['weight']:.3e}, nodes={int(last['nodes'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.legend(loc="best")


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    reference_energy = None if args.no_reference_energy else args.reference_energy
    fig, ax = plt.subplots(figsize=(9, 5))
    while True:
        data = read_samples(args.samples.expanduser())
        update_plot(ax, data, args.samples.expanduser(), window=args.window, reference_energy=reference_energy)
        fig.tight_layout()
        if args.save is not None:
            fig.savefig(args.save.expanduser(), dpi=160)
        if not args.follow:
            if args.save is not None:
                return
            plt.show()
            return
        plt.pause(max(args.interval, 0.1))
        time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    main()
