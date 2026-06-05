from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trot_fe2_real_fields_fit_hk as fe_common

from trot.staging import load as load_staged
from trot.staging import _normalize_real_field_centers


DEFAULT_FE_SPIN_PATTERN = "aabb"


def read_fe4_occupation_inits(path: Path) -> tuple[str, ...]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fe4 UHF init file not found: {path}.")

    occupation_strings: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = re.split(r"#|!|//", raw_line, maxsplit=1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                f"Expected an Fe4 init line like 'INIT 12 <occupation> <energy>', got: {raw_line!r}"
            )
        occupation_strings.append(parts[2])

    if not occupation_strings:
        raise ValueError(f"No Fe4 occupation strings found in {path}.")
    return tuple(occupation_strings)


def read_fe4_occupation_init(path: Path, uhf_init: int) -> str:
    occupation_strings = read_fe4_occupation_inits(path)
    try:
        return occupation_strings[uhf_init]
    except IndexError as exc:
        raise ValueError(
            f"UHF init {uhf_init} is not available in {path}; "
            f"valid values are 0 through {len(occupation_strings) - 1}."
        ) from exc


def infer_fe_centers_from_occupation_inits(
    path: Path, *, band_model: int
) -> tuple[tuple[int, ...], ...]:
    occupation_strings = read_fe4_occupation_inits(path)
    active_sites = sorted(
        {idx for occ in occupation_strings for idx, site in enumerate(occ) if site in {"a", "b"}}
    )

    active_groups: list[list[int]] = []
    for site in active_sites:
        if not active_groups or site != active_groups[-1][-1] + 1:
            active_groups.append([site])
        else:
            active_groups[-1].append(site)

    centers: list[tuple[int, ...]] = []
    for group in active_groups:
        if len(group) % 5 != 0:
            raise ValueError(
                "Cannot infer five-orbital Fe centers from UHF init occupations: "
                f"active group {group[0]}:{group[-1] + 1} has length {len(group)}."
            )
        for offset in range(0, len(group), 5):
            centers.append(_truncate_fe_center(tuple(group[offset : offset + 5]), band_model=band_model))

    if len(centers) != 4:
        raise ValueError(
            f"Expected four Fe centers inferred from {path}, found {len(centers)} centers."
        )
    return tuple(centers)


def _parse_range_token(token: str) -> tuple[int, ...]:
    token = token.replace(" ", "")
    if ":" in token:
        start_s, stop_s = token.split(":", 1)
        start, stop = int(start_s), int(stop_s)
        return tuple(range(start, stop))
    if "-" in token:
        start_s, stop_s = token.split("-", 1)
        start, stop = int(start_s), int(stop_s)
        return tuple(range(start, stop + 1))
    raise ValueError(f"Unsupported Fe range token: {token!r}")


def _truncate_fe_center(orbitals: tuple[int, ...], *, band_model: int) -> tuple[int, ...]:
    if band_model == 2:
        if len(orbitals) < 2:
            raise ValueError(f"Cannot make 2-band Fe center from range {orbitals!r}.")
        return orbitals[:2]
    if band_model == 5 and len(orbitals) > 5:
        return orbitals[:5]
    return orbitals


def _strip_init_record_label(line: str, range_pattern: re.Pattern[str]) -> str:
    keyed_label = re.match(r"^\s*(?:init|uhf|state|record|guess)\s+\d+\s*[:=]\s*(.*)$", line, re.I)
    if keyed_label:
        return keyed_label.group(1).strip()

    numeric_label = re.match(r"^\s*\d+\s*[:=]\s*(.*)$", line)
    if numeric_label and len(range_pattern.findall(numeric_label.group(1))) >= 4:
        return numeric_label.group(1).strip()

    return line


def read_fe_init_records(path: Path, *, band_model: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Fe4 init-range file not found: {path}. Pass --real-field-centers or --inits."
        )

    centers: list[tuple[int, ...]] = []
    range_pattern = re.compile(r"(?<![\w.])\d+\s*[:\-]\s*\d+(?![\w.])")
    for raw_line in path.read_text().splitlines():
        line = re.split(r"#|!|//", raw_line, maxsplit=1)[0].strip()
        if not line:
            continue
        line = _strip_init_record_label(line, range_pattern)
        for token in range_pattern.findall(line):
            centers.append(_truncate_fe_center(_parse_range_token(token), band_model=band_model))

    if not centers:
        raise ValueError(
            f"No Fe ranges found in {path}. Use Python-style ranges like '2:7' "
            "or inclusive ranges like '2-6'."
        )
    if len(centers) % 4 != 0:
        raise ValueError(
            f"Expected Fe4 init ranges in groups of four in {path}, found {len(centers)} ranges. "
            "Use Python-style ranges like '2:7' or inclusive ranges like '2-6'."
        )
    return tuple(tuple(centers[i : i + 4]) for i in range(0, len(centers), 4))


def read_fe_centers_from_inits(
    path: Path, *, band_model: int, uhf_init: int | None
) -> tuple[tuple[int, ...], ...]:
    records = read_fe_init_records(path, band_model=band_model)

    if uhf_init is None:
        if len(records) != 1:
            raise ValueError(
                f"Found {len(records)} Fe4 init records in {path}. "
                "Select one with --uhf-init N, using one-based indexing."
            )
        return records[0]

    if uhf_init < 1 or uhf_init > len(records):
        raise ValueError(
            f"--uhf-init {uhf_init} is out of range for {path}; "
            f"valid values are 1..{len(records)}."
        )
    return records[uhf_init - 1]


def _center_spec_for_cache(real_field_centers: Any) -> str:
    centers = real_field_centers if isinstance(real_field_centers, tuple) else (real_field_centers,)
    return "custom" if centers else "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage and run Fe4 with a center-targeted real-field route. "
            "By default, the UHF guess is selected from fe4-inits.inp."
        )
    )
    parser.add_argument(
        "fcidump",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("Fe4S4.FCIDUMP"),
        help="Path to the Fe4 FCIDUMP file.",
    )
    parser.add_argument(
        "--inits",
        "--uhf-inits",
        dest="inits",
        type=Path,
        default=Path(__file__).with_name("fe4-inits.inp"),
        help=(
            "File containing Fe4 UHF occupation-string initial guesses. "
            "--uhf-inits is accepted for compatibility with the original Fe4 scripts."
        ),
    )
    parser.add_argument(
        "--uhf-init",
        type=int,
        default=0,
        help=(
            "Zero-based row in the UHF init file to use, matching the original Fe4 scripts."
        ),
    )
    parser.add_argument("--cache", type=Path, default=None, help="Real-field staged HDF5 cache path.")
    parser.add_argument(
        "--standard-cache",
        type=Path,
        default=None,
        help="Ordinary Cholesky staged HDF5 cache path used with --compare-standard.",
    )
    parser.add_argument(
        "--method",
        choices=("uhf", "ccsd", "ccsdt", "ccsdtq"),
        default="ccsd",
        help=(
            "Trial/preparation method. Use 'uhf' for a UHF Slater trial without ccpy; "
            "CC methods run ccpy and stage a CI expansion from the requested --order."
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="CI excitation order staged from ccpy amplitudes. Ignored for --method uhf.",
    )
    parser.add_argument(
        "--fe-band-model",
        type=int,
        choices=(2, 5),
        default=5,
        help="Number of orbitals to take from each inferred Fe center.",
    )
    parser.add_argument(
        "--fe-spin-pattern",
        default=DEFAULT_FE_SPIN_PATTERN,
        help="Spin pattern for the four Fe centers in the UHF guess, e.g. aabb or abab.",
    )
    parser.add_argument(
        "--occupation-string",
        default=None,
        help=(
            "Optional explicit site occupation string for the UHF guess using 2/a/b/0. "
            "Overrides --fe-spin-pattern."
        ),
    )
    parser.add_argument(
        "--real-field-centers",
        default=None,
        help=(
            "Localized orbital ranges for the Fe centers. Overrides --inits. "
            "Use Python slice syntax per center, comma separated; e.g. '2:7,13:18,...'."
        ),
    )
    parser.add_argument(
        "--real-field-method",
        choices=(
            "hk_density",
            "local_exact",
            "kanamori_sign",
            "kanamori_uj",
            "kanamori_sign_full",
        ),
        default="local_exact",
        help="Real-field staging route for the specified centers.",
    )
    parser.add_argument(
        "--chkfile",
        type=Path,
        default=Path(__file__).with_name("fe4_uhf.chk"),
        help="PySCF checkpoint file for the UHF calculation.",
    )
    parser.add_argument("--chol-cut", type=float, default=1.0e-5)
    parser.add_argument("--amp-convergence", type=float, default=1.0e-10)
    parser.add_argument("--energy-convergence", type=float, default=1.0e-10)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--compare-standard",
        action="store_true",
        help="Also stage and optionally run the ordinary Cholesky decomposition.",
    )
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--verbose-stage", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--n-walkers", type=int, default=200)
    parser.add_argument("--n-eql-blocks", type=int, default=20)
    parser.add_argument("--n-blocks", type=int, default=200)
    parser.add_argument("--n-prop-steps", type=int, default=50)
    parser.add_argument("--n-exp-terms", type=int, default=6)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    custom_centers = any(
        arg == "--real-field-centers" or arg.startswith("--real-field-centers=")
        for arg in sys.argv[1:]
    )
    if args.real_field_centers is None:
        args.real_field_centers = infer_fe_centers_from_occupation_inits(
            args.inits, band_model=args.fe_band_model
        )

    init_tag = "customocc" if args.occupation_string is not None else f"init{args.uhf_init}"
    trial_tag = (
        f"uhf_{init_tag}"
        if args.method == "uhf"
        else f"{args.method}_order{args.order}_{init_tag}"
    )
    if custom_centers:
        center_tag = _center_spec_for_cache(args.real_field_centers)
    else:
        center_tag = f"{args.fe_band_model}band"
    if args.cache is None:
        args.cache = Path(__file__).with_name(
            f"fe4_real_fields_{args.real_field_method}_{center_tag}_{trial_tag}_staged.h5"
        )
    if args.standard_cache is None:
        args.standard_cache = Path(__file__).with_name(f"fe4_standard_{trial_tag}_staged.h5")
    return args


def fe4_broken_symmetry_guess(umf: Any, args: argparse.Namespace) -> np.ndarray:
    n_sites = int(umf.mol.nao)
    if args.occupation_string is not None:
        occupation_string = args.occupation_string.strip()
    elif args.uhf_init is not None:
        occupation_string = read_fe4_occupation_init(args.inits, args.uhf_init)
        print(f"Running UHF init {args.uhf_init} with configuration {occupation_string}")
    else:
        centers = _normalize_real_field_centers(args.real_field_centers, norb=n_sites)
        pattern = args.fe_spin_pattern.strip().lower()
        if len(pattern) != len(centers):
            raise ValueError(
                f"--fe-spin-pattern has length {len(pattern)}, but there are {len(centers)} centers."
            )
        if any(ch not in {"a", "b"} for ch in pattern):
            raise ValueError("--fe-spin-pattern may contain only 'a' and 'b'.")

        occupations = np.zeros((n_sites, 2), dtype=np.float64)
        fe_orbitals: set[int] = set()
        for center, spin in zip(centers, pattern):
            for orb in center:
                if orb in fe_orbitals:
                    raise ValueError(f"Fe centers overlap at orbital {orb}.")
                fe_orbitals.add(int(orb))
                occupations[int(orb)] = [1.0, 0.0] if spin == "a" else [0.0, 1.0]

        n_fe_electrons = len(fe_orbitals)
        n_remaining = int(umf.mol.nelectron) - n_fe_electrons
        if n_remaining < 0 or n_remaining % 2 != 0:
            raise ValueError(
                "Cannot build paired ligand/core guess: "
                f"nelec={umf.mol.nelectron}, singly occupied Fe orbitals={n_fe_electrons}."
            )
        n_pairs = n_remaining // 2
        for orb in range(n_sites):
            if orb in fe_orbitals:
                continue
            if n_pairs == 0:
                break
            occupations[orb] = [1.0, 1.0]
            n_pairs -= 1
        if n_pairs:
            raise ValueError(f"Not enough non-Fe orbitals to place {n_pairs} electron pairs.")

        dm0 = np.zeros((2, n_sites, n_sites))
        idx = np.mgrid[:n_sites]
        dm0[0, idx, idx] = occupations[:, 0]
        dm0[1, idx, idx] = occupations[:, 1]
        return dm0

    if len(occupation_string) != n_sites:
        raise ValueError(
            "Fe4 occupation string has length "
            f"{len(occupation_string)}, but the FCIDUMP has {n_sites} orbitals."
        )

    try:
        occupations = np.array(
            [
                {
                    "2": [1.0, 1.0],
                    "a": [1.0, 0.0],
                    "b": [0.0, 1.0],
                    "0": [0.0, 0.0],
                }[site]
                for site in occupation_string
            ]
        )
    except KeyError as exc:
        raise ValueError(
            "Fe4 occupation strings may contain only '2', 'a', 'b', and '0'."
        ) from exc

    dm0 = np.zeros((2, n_sites, n_sites))
    idx = np.mgrid[:n_sites]
    dm0[0, idx, idx] = occupations[:, 0]
    dm0[1, idx, idx] = occupations[:, 1]
    return dm0


def build_uhf_mf(args: argparse.Namespace):
    start = time.perf_counter()
    mf = fe_common.prepare_fcidump_mf(args.fcidump)
    umf = mf.to_uhf().newton()
    umf.chkfile = str(args.chkfile)
    umf.mol.verbose = 4
    umf.kernel(dm0=fe4_broken_symmetry_guess(umf, args))
    seconds = time.perf_counter() - start
    print(f"Prepared Fe4 UHF inputs in {seconds:.2f}s")
    return umf


def build_cc_driver(args: argparse.Namespace):
    try:
        from ccpy.drivers.driver import Driver
    except ImportError as exc:
        raise RuntimeError("This example requires ccpy. Please install ccpy to run it.") from exc

    start = time.perf_counter()
    umf = build_uhf_mf(args)
    cc_driver = Driver.from_pyscf(umf, nfrozen=0, uhf=True)
    cc_driver.options["amp_convergence"] = args.amp_convergence
    cc_driver.options["energy_convergence"] = args.energy_convergence
    cc_driver.options["RHF_symmetry"] = False
    cc_driver.run_cc(method=args.method)

    seconds = time.perf_counter() - start
    print(f"Prepared Fe4 UHF/{args.method.upper()} inputs in {seconds:.2f}s")
    return cc_driver, umf


def _load_cache(path: Path):
    start = time.perf_counter()
    staged = load_staged(path)
    seconds = time.perf_counter() - start
    print(f"Loaded staged inputs from {path} in {seconds:.2f}s")
    return staged, seconds


def build_or_load_staged(args: argparse.Namespace) -> dict[str, tuple[Any, float]]:
    args.cache = args.cache.expanduser().resolve()
    args.standard_cache = args.standard_cache.expanduser().resolve()
    real_staged: tuple[Any, float] | None = None
    standard_staged: tuple[Any, float] | None = None

    need_real = args.overwrite_cache or not args.cache.exists()
    need_standard = args.compare_standard and (
        args.overwrite_cache or not args.standard_cache.exists()
    )

    if not need_real:
        staged_inputs, seconds = _load_cache(args.cache)
        real_staged = (fe_common._annotate_staged(staged_inputs, args), seconds)
    if args.compare_standard and not need_standard:
        staged_inputs, seconds = _load_cache(args.standard_cache)
        standard_staged = (fe_common._annotate_staged(staged_inputs, args), seconds)

    if need_real or need_standard:
        if args.method == "uhf":
            umf = build_uhf_mf(args)
            if need_real:
                real_staged = fe_common.stage_uhf_one(
                    label=f"real-field {args.real_field_method} UHF",
                    cache=args.cache,
                    args=args,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                )
            if need_standard:
                standard_staged = fe_common.stage_uhf_one(
                    label="standard Cholesky UHF",
                    cache=args.standard_cache,
                    args=args,
                    umf=umf,
                    real_field_centers=None,
                )
        else:
            cc_driver, umf = build_cc_driver(args)
            if need_real:
                real_staged = fe_common.stage_ccpy_one(
                    label=f"real-field {args.real_field_method} {args.method} order {args.order}",
                    cache=args.cache,
                    args=args,
                    cc_driver=cc_driver,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                )
            if need_standard:
                standard_staged = fe_common.stage_ccpy_one(
                    label=f"standard Cholesky {args.method} order {args.order}",
                    cache=args.standard_cache,
                    args=args,
                    cc_driver=cc_driver,
                    umf=umf,
                    real_field_centers=None,
                )

    staged: dict[str, tuple[Any, float]] = {}
    if real_staged is not None:
        staged["real-field"] = real_staged
    if standard_staged is not None:
        staged["standard"] = standard_staged
    return staged


def main() -> None:
    args = parse_args()
    staged = build_or_load_staged(args)

    for label, (staged_inputs, stage_seconds) in staged.items():
        fe_common._print_field_metadata(label, staged_inputs)
        print(f"{label} staging/load timing: {stage_seconds:.2f}s")

    if args.stage_only:
        return

    for label, (staged_inputs, _) in staged.items():
        fe_common.run_afqmc(label, staged_inputs, args)


if __name__ == "__main__":
    main()
