from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from pyscf.tools.fcidump import to_scf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from trot.afqmc import Afqmc
    from trot.prop.types import QmcParams
    from trot.staging import load as load_staged
    from trot.staging import stage
    from trot.staging import stage_from_ccpy
    from trot.staging import StagedMfOrCc
    from trot.staging import _stage_ham_input_from_fcidump
except ImportError as exc:
    raise RuntimeError(
        "This example requires trot on PYTHONPATH. For a local checkout, run with "
        f"PYTHONPATH={ROOT}:$PYTHONPATH."
    ) from exc


DEFAULT_FE_CENTERS_5_BAND = "2:7,13:18"
DEFAULT_FE_CENTERS_2_BAND = "2:4,13:15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage and run Fe2S2 with a center-targeted real-field route. "
            "The default Fe-center ranges are the five-orbital localized blocks "
            f"{DEFAULT_FE_CENTERS_5_BAND}."
        )
    )
    parser.add_argument(
        "fcidump",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("Fe2S2.FCIDUMP"),
        help="Path to the Fe2S2 FCIDUMP file.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Real-field staged HDF5 cache path.",
    )
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
        help=(
            "Preset Fe-center orbital count. The 5-band preset uses "
            f"{DEFAULT_FE_CENTERS_5_BAND}; the 2-band preset uses "
            f"{DEFAULT_FE_CENTERS_2_BAND}. Ignored when --real-field-centers is set."
        ),
    )
    parser.add_argument(
        "--real-field-centers",
        default=None,
        help=(
            "Localized orbital ranges for the Fe centers. Use Python slice syntax per center, "
            "comma separated; e.g. '2:7,13:18'. Overrides --fe-band-model."
        ),
    )
    parser.add_argument(
        "--real-field-method",
        choices=("hk_density", "local_exact", "kanamori_sign", "kanamori_sign_full"),
        default="local_exact",
        help="Real-field staging route for the specified centers.",
    )
    parser.add_argument(
        "--chkfile",
        type=Path,
        default=Path(__file__).with_name("uhf.chk"),
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
        args.real_field_centers = (
            DEFAULT_FE_CENTERS_2_BAND
            if args.fe_band_model == 2
            else DEFAULT_FE_CENTERS_5_BAND
        )
    if args.cache is None:
        trial_tag = "uhf" if args.method == "uhf" else f"{args.method}_order{args.order}"
        cache_name = (
            f"fe2_real_fields_{args.real_field_method}_custom_{trial_tag}_staged.h5"
            if custom_centers
            else (
                f"fe2_real_fields_{args.real_field_method}_{args.fe_band_model}"
                f"band_{trial_tag}_staged.h5"
            )
        )
        args.cache = Path(__file__).with_name(cache_name)
    if args.standard_cache is None:
        trial_tag = "uhf" if args.method == "uhf" else f"{args.method}_order{args.order}"
        args.standard_cache = Path(__file__).with_name(f"fe2_standard_{trial_tag}_staged.h5")
    return args


def prepare_fcidump_mf(fcidump_path: Path):
    mf = to_scf(str(fcidump_path), molpro_orbsym=True, mf=None)

    intor_symmetric = mf.mol.intor_symmetric
    mf.mol.intor_symmetric = lambda intor, **kwargs: (
        np.eye(mf.mol.nao) if intor == "int1e_ovlp" else intor_symmetric(intor, **kwargs)
    )

    mf.mol.verbose = 4
    mf.mol._symm_orig = np.zeros((3,))
    mf.mol._symm_axes = np.eye(3)
    mf.mol.irrep_name = ["A"]
    mf.mol.groupname = "C1"
    mf.mol.symmetry = "C1"

    return mf


def fe2_broken_symmetry_guess(umf) -> np.ndarray:
    n_sites = int(umf.mol.nao)
    occupation_string = "22aaaaa222222bbbbb22"
    if len(occupation_string) != n_sites:
        raise ValueError(
            "Fe2 occupation string has length "
            f"{len(occupation_string)}, but the FCIDUMP has {n_sites} orbitals."
        )

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
    dm0 = np.zeros((2, n_sites, n_sites))
    idx = np.mgrid[:n_sites]
    dm0[0, idx, idx] = occupations[:, 0]
    dm0[1, idx, idx] = occupations[:, 1]
    return dm0


def build_uhf_mf(args: argparse.Namespace):
    start = time.perf_counter()
    mf = prepare_fcidump_mf(args.fcidump)
    umf = mf.to_uhf().newton()
    umf.chkfile = str(args.chkfile)
    umf.mol.verbose = 4
    umf.kernel(dm0=fe2_broken_symmetry_guess(umf))
    seconds = time.perf_counter() - start
    print(f"Prepared Fe2 UHF inputs in {seconds:.2f}s")
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
    print(f"Prepared Fe2 UHF/{args.method.upper()} inputs in {seconds:.2f}s")
    return cc_driver, umf


def _load_cache(path: Path):
    start = time.perf_counter()
    staged = load_staged(path)
    seconds = time.perf_counter() - start
    print(f"Loaded staged inputs from {path} in {seconds:.2f}s")
    return staged, seconds


def _print_local_parameters(meta: dict[str, Any]) -> None:
    center_reports = meta.get("center_reports", [])
    if not center_reports:
        return

    print("local parameters:")
    for report in center_reports:
        center = int(report["center"])
        orbitals = report.get("orbitals", [])
        params = report.get("parameters", {})
        kanamori_terms = report.get("kanamori_terms", [])
        onsite_dec = {
            int(term["orbital"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") == "onsite_U"
        }
        onsite_pref = {
            int(term["orbital"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") == "onsite_U"
        }
        density_dec = {
            tuple(term["orbitals"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") == "interorbital_Uprime"
        }
        density_pref = {
            tuple(term["orbitals"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") == "interorbital_Uprime"
        }
        hund_dec = {
            tuple(term["orbitals"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") == "hund_J_bond"
        }
        hund_pref = {
            tuple(term["orbitals"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") == "hund_J_bond"
        }
        print(f"  center={center} orbitals={orbitals}")
        for term in params.get("onsite_U", []):
            orb = int(term["orbital"])
            dec = onsite_dec.get(orb)
            suffix = f" decomposition={dec}" if dec is not None else ""
            if onsite_pref.get(orb) is not None and onsite_pref[orb] != dec:
                suffix += f" preferred={onsite_pref[orb]}"
            print(f"    U      orb={orb} value={float(term['U']): .10f}{suffix}")

        uprime = {
            tuple(term["orbitals"]): float(term["Uprime"])
            for term in params.get("interorbital_Uprime", [])
        }
        exchange = {tuple(term["orbitals"]): float(term["J"]) for term in params.get("exchange_J", [])}
        pair_hopping = {
            tuple(term["orbitals"]): float(term["P"])
            for term in params.get("pair_hopping_like", [])
        }
        for pair in sorted(set(uprime) | set(exchange) | set(pair_hopping)):
            dec_parts = []
            if pair in density_dec:
                dec_parts.append(f"Uprime_dec={density_dec[pair]}")
                if density_pref.get(pair) is not None and density_pref[pair] != density_dec[pair]:
                    dec_parts.append(f"Uprime_pref={density_pref[pair]}")
            if pair in hund_dec:
                dec_parts.append(f"J_dec={hund_dec[pair]}")
                if hund_pref.get(pair) is not None and hund_pref[pair] != hund_dec[pair]:
                    dec_parts.append(f"J_pref={hund_pref[pair]}")
            suffix = f" {' '.join(dec_parts)}" if dec_parts else ""
            print(
                f"    pair   orbs={list(pair)} "
                f"Uprime={uprime.get(pair, 0.0): .10f} "
                f"J={exchange.get(pair, 0.0): .10f} "
                f"P={pair_hopping.get(pair, 0.0): .10f}"
                f"{suffix}"
            )


def _print_field_metadata(label: str, staged: Any) -> None:
    meta = staged.meta.get("field_metadata")
    print(f"\n[{label}]")
    print(f"n_orbitals = {staged.ham.norb}")
    print(f"n_fields   = {staged.ham.chol.shape[0]}")
    if hasattr(staged, "meta") and staged.meta.get("fe_band_model") is not None:
        print(f"Fe preset  = {staged.meta.get('fe_band_model')}-band")
    if hasattr(staged, "meta") and staged.meta.get("trial_method") is not None:
        method_line = f"trial method = {staged.meta.get('trial_method')}"
        if staged.meta.get("trial_order") is not None:
            method_line += f" order {staged.meta.get('trial_order')}"
        method_line += f" ({staged.trial.kind})"
        print(method_line)
    if not meta:
        print("field route = ordinary Cholesky")
        return

    print(f"field route             = {meta.get('real_field_fit')}")
    print(f"centers                 = {meta.get('centers')}")
    if meta.get("real_field_fit") == "hk_density":
        print(f"HK real fields          = {meta.get('n_hk_real_fields')}")
    else:
        print(f"local real fields       = {meta.get('n_local_real_fields')}")
        print(f"local complex fields    = {meta.get('n_local_complex_fields')}")
    print(f"residual real fields    = {meta.get('n_residual_real_fields')}")
    print(f"residual complex fields = {meta.get('n_residual_complex_fields')}")
    frob = meta.get("frobenius", {})
    if frob:
        print("Frobenius diagnostics:")
        print(f"  ||V_full||                 = {float(frob['full_norm']):.10f}")
        if "hk_onsite_norm" in frob:
            print(f"  ||V_HK onsite||            = {float(frob['hk_onsite_norm']):.10f}")
        if "local_block_norm" in frob:
            print(f"  ||V_local block||          = {float(frob['local_block_norm']):.10f}")
        if "extracted_local_block_norm" in frob:
            print(
                "  ||V_local extracted||      = "
                f"{float(frob['extracted_local_block_norm']):.10f}"
            )
        print(f"  ||V_residual||             = {float(frob['residual_norm']):.10f}")
        print(f"  ||V_center block||         = {float(frob['center_block_norm']):.10f}")
        print(f"  ||V_full pair||            = {float(frob['full_pair_norm']):.10f}")
        print(
            "  residual pair rel. error   = "
            f"{float(frob['residual_pair_reconstruction_relative_error']):.3e}"
        )
        if "hk_fraction_full_weight" in frob:
            print("  HK/full weight fraction    = " f"{float(frob['hk_fraction_full_weight']):.6f}")
            print(
                "  HK/full pair weight frac.  = "
                f"{float(frob['hk_fraction_full_pair_weight']):.6f}"
            )
            print(
                "  HK/center weight fraction  = "
                f"{float(frob['hk_fraction_center_block_weight']):.6f}"
            )
        if "local_block_fraction_full_weight" in frob:
            print(
                "  local/full weight fraction = "
                f"{float(frob['local_block_fraction_full_weight']):.6f}"
            )
            print(
                "  extracted/full weight frac.= "
                f"{float(frob['extracted_local_block_fraction_full_weight']):.6f}"
            )
    extracted = meta.get("extracted_terms", [])
    if extracted:
        print("onsite U terms:")
        for term in extracted[:20]:
            print(
                f"  center={term['center']} orbital={term['orbital']} " f"U={float(term['U']):.10f}"
            )
        if len(extracted) > 20:
            print(f"  ... {len(extracted) - 20} more")
    _print_local_parameters(meta)


def _annotate_staged(staged: Any, args: argparse.Namespace) -> Any:
    staged.meta["fe_band_model"] = args.fe_band_model
    staged.meta["trial_method"] = args.method
    if args.method != "uhf":
        staged.meta["trial_order"] = int(args.order)
    return staged


def stage_ccpy_one(
    *,
    label: str,
    cache: Path,
    args: argparse.Namespace,
    cc_driver: Any,
    umf: Any,
    real_field_centers: str | None,
):
    start = time.perf_counter()
    staged = stage_from_ccpy(
        cc_driver,
        umf,
        order=args.order,
        chol_cut=args.chol_cut,
        fcidump=args.fcidump,
        cache=cache,
        overwrite=True,
        verbose=args.verbose_stage,
        real_field_centers=real_field_centers,
        real_field_method=args.real_field_method if real_field_centers is not None else "hk_density",
    )
    staged = _annotate_staged(staged, args)
    seconds = time.perf_counter() - start
    print(f"Wrote {label} staged inputs to {cache} in {seconds:.2f}s")
    return staged, seconds


def stage_uhf_one(
    *,
    label: str,
    cache: Path,
    args: argparse.Namespace,
    umf: Any,
    real_field_centers: str | None,
):
    start = time.perf_counter()
    staged_obj = StagedMfOrCc(umf, 0)
    ham = _stage_ham_input_from_fcidump(
        staged_obj,
        fcidump=args.fcidump,
        chol_cut=args.chol_cut,
        verbose=args.verbose_stage,
        real_field_centers=real_field_centers,
        real_field_method=args.real_field_method if real_field_centers is not None else "hk_density",
    )
    staged = stage(
        umf,
        chol_cut=args.chol_cut,
        cache=cache,
        overwrite=True,
        verbose=args.verbose_stage,
        ham=ham,
    )
    staged = _annotate_staged(staged, args)
    seconds = time.perf_counter() - start
    print(f"Wrote {label} staged inputs to {cache} in {seconds:.2f}s")
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
        real_staged = (_annotate_staged(staged_inputs, args), seconds)
    if args.compare_standard and not need_standard:
        staged_inputs, seconds = _load_cache(args.standard_cache)
        standard_staged = (_annotate_staged(staged_inputs, args), seconds)

    if need_real or need_standard:
        if args.method == "uhf":
            umf = build_uhf_mf(args)
            if need_real:
                real_staged = stage_uhf_one(
                    label=f"real-field {args.real_field_method} UHF",
                    cache=args.cache,
                    args=args,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                )
            if need_standard:
                standard_staged = stage_uhf_one(
                    label="standard Cholesky UHF",
                    cache=args.standard_cache,
                    args=args,
                    umf=umf,
                    real_field_centers=None,
                )
        else:
            cc_driver, umf = build_cc_driver(args)
            if need_real:
                real_staged = stage_ccpy_one(
                    label=f"real-field {args.real_field_method} {args.method} order {args.order}",
                    cache=args.cache,
                    args=args,
                    cc_driver=cc_driver,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                )
            if need_standard:
                standard_staged = stage_ccpy_one(
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


def run_afqmc(label: str, staged: Any, args: argparse.Namespace) -> tuple[float, float, float]:
    af = Afqmc(staged)
    af.mixed_precision = args.mixed_precision
    af.params = QmcParams(
        dt=args.dt,
        n_chunks=args.n_chunks,
        n_exp_terms=args.n_exp_terms,
        n_prop_steps=args.n_prop_steps,
        n_blocks=args.n_blocks,
        n_walkers=args.n_walkers,
        seed=args.seed,
        n_eql_blocks=args.n_eql_blocks,
    )
    af.walker_kind = "unrestricted"

    start = time.perf_counter()
    mean, err = af.kernel()
    seconds = time.perf_counter() - start
    print(f"{label} AFQMC/{staged.trial.kind} energy: {mean:.10f} +/- {err:.10f} Ha")
    print(f"{label} AFQMC timing: {seconds:.2f}s")
    return float(mean), float(err), seconds


def main() -> None:
    args = parse_args()
    staged = build_or_load_staged(args)

    for label, (staged_inputs, stage_seconds) in staged.items():
        _print_field_metadata(label, staged_inputs)
        print(f"{label} staging/load timing: {stage_seconds:.2f}s")

    if args.stage_only:
        return

    for label, (staged_inputs, _) in staged.items():
        run_afqmc(label, staged_inputs, args)


if __name__ == "__main__":
    main()
