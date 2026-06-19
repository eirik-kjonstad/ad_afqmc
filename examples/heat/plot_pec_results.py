from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys


@dataclass(frozen=True)
class PecResult:
    source: Path
    atoms: str
    bond_length: float
    hamiltonian: str
    trial: str
    energy: float
    error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot diatomic PEC_RESULT records emitted by diatomic_pec.py. "
            "Inputs may be compact result files or full logs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("n2_pec_results.txt")],
        help="Files containing PEC_RESULT lines. Default: n2_pec_results.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("n2_pec_results.png"),
        help="Output image path. Default: n2_pec_results.png",
    )
    parser.add_argument("--title", default=None, help="Plot title. Defaults to a data-derived title.")
    parser.add_argument("--atoms", default=None, help="Only plot this atom label, e.g. N-N.")
    parser.add_argument("--trial", default=None, help="Only plot this trial kind, e.g. uhf or ucisd.")
    parser.add_argument(
        "--hamiltonian",
        default=None,
        help="Only plot this Hamiltonian route, e.g. charge-spin or standard.",
    )
    parser.add_argument(
        "--no-errorbars",
        action="store_true",
        help="Plot points without AFQMC statistical error bars.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    return parser.parse_args()


def parse_key_values(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_result_line(path: Path, line: str) -> PecResult | None:
    line = line.strip()
    if not line.startswith("PEC_RESULT "):
        return None

    fields = parse_key_values(line[len("PEC_RESULT ") :])
    required = ("atoms", "bond_length", "hamiltonian", "trial", "energy", "error")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"{path}: malformed PEC_RESULT line missing {', '.join(missing)}: {line}")

    return PecResult(
        source=path,
        atoms=fields["atoms"],
        bond_length=float(fields["bond_length"]),
        hamiltonian=fields["hamiltonian"],
        trial=fields["trial"],
        energy=float(fields["energy"]),
        error=float(fields["error"]),
    )


def load_results(paths: list[Path]) -> list[PecResult]:
    results: list[PecResult] = []
    for path in paths:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            for line in handle:
                result = parse_result_line(path, line)
                if result is not None:
                    results.append(result)
    return results


def filter_results(results: list[PecResult], args: argparse.Namespace) -> list[PecResult]:
    filtered = results
    if args.atoms is not None:
        filtered = [result for result in filtered if result.atoms == args.atoms]
    if args.trial is not None:
        filtered = [result for result in filtered if result.trial == args.trial]
    if args.hamiltonian is not None:
        filtered = [result for result in filtered if result.hamiltonian == args.hamiltonian]
    return filtered


def default_title(results: list[PecResult]) -> str:
    atoms = sorted({result.atoms for result in results})
    if len(atoms) == 1:
        return f"{atoms[0]} potential energy curve"
    return "Diatomic potential energy curves"


def plot_results(results: list[PecResult], args: argparse.Namespace) -> None:
    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot PEC results. Install matplotlib first.") from exc

    grouped: dict[tuple[str, str, str], list[PecResult]] = defaultdict(list)
    for result in results:
        grouped[(result.atoms, result.trial, result.hamiltonian)].append(result)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X")

    for index, (key, group) in enumerate(sorted(grouped.items())):
        atoms, trial, hamiltonian = key
        group = sorted(group, key=lambda result: result.bond_length)
        x = [result.bond_length for result in group]
        y = [result.energy for result in group]
        yerr = None if args.no_errorbars else [result.error for result in group]
        label = f"{atoms} {trial} {hamiltonian}"
        marker = markers[index % len(markers)]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            linewidth=1.6,
            markersize=4.5,
            capsize=3 if yerr is not None else 0,
            label=label,
        )

    ax.set_title(args.title or default_title(results))
    ax.set_xlabel("Bond length (Angstrom)")
    ax.set_ylabel("Energy (Ha)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Wrote {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = filter_results(load_results(args.inputs), args)
    if not results:
        inputs = ", ".join(str(path) for path in args.inputs)
        raise SystemExit(f"No PEC_RESULT records found for the selected filters in: {inputs}")

    plot_results(results, args)


if __name__ == "__main__":
    main()
