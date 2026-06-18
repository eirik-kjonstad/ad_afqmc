from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pyscf import gto, scf

from trot.afqmc import Afqmc
from trot.staging import load as load_staged
from trot.staging import StagedMfOrCc, TrialInput, _stage_ham_input_from_fcidump
from trot.staging import stage, stage_from_ccpy


TRIAL_TO_CCPY = {
    "ucisd": ("ccsd", 2),
    "ucisdt": ("ccsdt", 3),
    "ucisdtq": ("ccsdtq", 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AFQMC for open-shell CH with UHF/UCISD/UCISDT/UCISDTQ trials."
    )
    parser.add_argument(
        "--trial",
        choices=("uhf", "ucisd", "ucisdt", "ucisdtq"),
        default="uhf",
        help="Trial wave function to stage and run.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Staged HDF5 cache path. Defaults to ch_<trial>_staged.h5 in this directory.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute staged inputs even if the cache file exists.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build or load staged inputs without running AFQMC.",
    )
    parser.add_argument("--chol-cut", type=float, default=1.0e-14)
    parser.add_argument("--n-walkers", type=int, default=80)
    parser.add_argument("--n-eql-blocks", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--scf-max-cycle", type=int, default=300)
    parser.add_argument("--cc-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--stability-cycles", type=int, default=5)
    parser.add_argument("--pyscf-verbose", type=int, default=3)
    parser.add_argument("--stage-verbose", action="store_true")
    return parser.parse_args()


def default_cache_path(trial: str) -> Path:
    return Path(__file__).with_name(f"ch_{trial}_staged.h5")


def build_mol(verbose: int):
    return gto.M(
        atom="""
        C   0.000000   0.000000   0.000000
        H   1.116670   0.000000   0.000000
        """,
        unit="Angstrom",
        basis="6-31g",
        charge=0,
        spin=1,
        symmetry="c1",
        verbose=verbose,
    )


def build_fcidump_context(mf) -> dict[str, object]:
    mol = mf.mol
    norb = int(mf.mo_coeff[0].shape[1])
    return {
        "NORB": norb,
        "NELEC": int(mol.nelectron),
        "MS2": int(mol.spin),
        "ECORE": float(mol.energy_nuc()),
        "H1": np.asarray(mf.get_hcore()),
        "H2": np.asarray(mol.intor("int2e")),
    }


def uhf_identity_trial(norb: int) -> TrialInput:
    eye = np.eye(norb)
    return TrialInput(
        kind="uhf",
        data={"mo_a": eye, "mo_b": eye},
        frozen=0,
        source_kind="mf",
    )


def validate_charge_spin_staged(staged, cache: Path):
    field_metadata = staged.ham.field_metadata or {}
    if field_metadata.get("real_field_fit") != "uhf_charge_spin_unrham":
        raise RuntimeError(
            f"{cache} was not staged with the UHF charge/spin unrestricted Hamiltonian; "
            "rerun with --overwrite-cache or choose a different --cache path."
        )
    return staged


def print_field_summary(staged) -> None:
    meta = staged.ham.field_metadata or {}
    print("\n[CH field summary]")
    print(f"trial kind              = {staged.trial.kind}")
    print(f"hamiltonian basis       = {staged.ham.basis}")
    print(f"one-body shape          = {tuple(staged.ham.h1.shape)}")
    print(f"field shape             = {tuple(staged.ham.chol.shape)}")
    print(f"n_orbitals              = {staged.ham.norb}")
    print(f"n_fields                = {staged.ham.chol.shape[0]}")
    print(f"field factors present   = {staged.ham.field_factors is not None}")
    print(
        "real-field AFQMC route  = "
        f"{'enabled' if staged.ham.field_factors is not None else 'disabled'}"
    )
    if not meta:
        print("field route             = ordinary Cholesky")
        return

    print(f"field route             = {meta.get('real_field_fit')}")
    print(f"field basis metadata    = {meta.get('basis')}")
    print(f"decomposition scope     = {meta.get('decomposition_scope')}")
    if meta.get("decomposition_scope") == "full":
        print(f"full real fields        = {meta.get('n_full_real_fields')}")
        print(f"full complex fields     = {meta.get('n_full_complex_fields')}")
        print(f"charge-dominant fields  = {meta.get('n_full_charge_dominant_fields')}")
        print(f"spin-dominant fields    = {meta.get('n_full_spin_dominant_fields')}")
        print(f"mixed charge-spin fields= {meta.get('n_full_mixed_charge_spin_fields')}")
    else:
        print(f"local real fields       = {meta.get('n_local_real_fields')}")
        print(f"local complex fields    = {meta.get('n_local_complex_fields')}")
        print(f"residual real fields    = {meta.get('n_residual_real_fields')}")
        print(f"residual complex fields = {meta.get('n_residual_complex_fields')}")

    frob = meta.get("frobenius", {})
    if frob:
        print("Frobenius diagnostics:")
        print(f"  ||V_full||                 = {float(frob['full_norm']):.10f}")
        print(f"  ||V_full pair||            = {float(frob['full_pair_norm']):.10f}")
        if meta.get("decomposition_scope") == "full":
            print(
                "  full pair rel. error       = "
                f"{float(frob['full_pair_reconstruction_relative_error']):.3e}"
            )
        else:
            print(
                "  residual pair rel. error   = "
                f"{float(frob['residual_pair_reconstruction_relative_error']):.3e}"
            )


def run_stable_uhf(mol, args: argparse.Namespace):
    mf = scf.UHF(mol)
    mf.max_cycle = args.scf_max_cycle
    mf.run(conv_tol=args.scf_conv_tol)
    if not mf.converged:
        raise RuntimeError("UHF did not converge.")

    if args.skip_stability:
        return mf

    for _ in range(args.stability_cycles):
        mo_i, _, stable_i, _ = mf.stability(return_status=True)
        if stable_i:
            return mf

        dm1 = mf.make_rdm1(mo_i, mf.mo_occ)
        mf.run(dm1, conv_tol=args.scf_conv_tol)
        if not mf.converged:
            raise RuntimeError("UHF did not converge after stability analysis.")

    _, _, stable_i, _ = mf.stability(return_status=True)
    if stable_i:
        return mf

    raise RuntimeError("UHF did not converge to an internally stable determinant.")


def build_ccpy_driver(mf, args: argparse.Namespace):
    Driver = require_ccpy_driver()
    cc_method, _order = TRIAL_TO_CCPY[args.trial]
    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["amp_convergence"] = args.cc_conv_tol
    driver.options["energy_convergence"] = args.cc_conv_tol
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method=cc_method)
    return driver


def require_ccpy_driver():
    try:
        from ccpy.drivers.driver import Driver
    except Exception as exc:
        raise RuntimeError("CI trial examples require ccpy. Please install ccpy to run them.") from exc

    return Driver


def make_staged(args: argparse.Namespace, cache: Path):
    if cache.exists() and not args.overwrite_cache:
        return validate_charge_spin_staged(load_staged(cache), cache)

    if args.trial != "uhf":
        require_ccpy_driver()

    mol = build_mol(args.pyscf_verbose)
    mf = run_stable_uhf(mol, args)
    fcidump = build_fcidump_context(mf)

    if args.trial == "uhf":
        ham = _stage_ham_input_from_fcidump(
            StagedMfOrCc(mf, 0),
            fcidump=fcidump,
            chol_cut=args.chol_cut,
            verbose=args.stage_verbose,
            real_field_centers=None,
            real_field_method="uhf_charge_spin_unrham",
        )
        staged = stage(
            mf,
            chol_cut=args.chol_cut,
            cache=cache,
            overwrite=args.overwrite_cache,
            verbose=args.stage_verbose,
            ham=ham,
            trial=uhf_identity_trial(int(ham.norb)),
        )
        return validate_charge_spin_staged(staged, cache)

    _cc_method, order = TRIAL_TO_CCPY[args.trial]
    driver = build_ccpy_driver(mf, args)
    staged = stage_from_ccpy(
        driver,
        mf,
        order=order,
        chol_cut=args.chol_cut,
        fcidump=fcidump,
        cache=cache,
        overwrite=args.overwrite_cache,
        verbose=args.stage_verbose,
        real_field_method="uhf_charge_spin_unrham",
    )
    return validate_charge_spin_staged(staged, cache)


def run_afqmc(staged, args: argparse.Namespace) -> tuple[float, float]:
    af = Afqmc(
        staged,
        n_walkers=args.n_walkers,
        n_eql_blocks=args.n_eql_blocks,
        n_blocks=args.n_blocks,
        n_chunks=args.n_chunks,
        seed=args.seed,
        dt=args.dt,
    )
    af.walker_kind = "unrestricted"
    af.mixed_precision = args.mixed_precision
    return af.kernel()


def main() -> None:
    args = parse_args()
    cache = (args.cache or default_cache_path(args.trial)).expanduser()
    staged = make_staged(args, cache)
    fit = staged.ham.field_metadata["real_field_fit"] if staged.ham.field_metadata else "standard"
    print(f"Staged CH/{staged.trial.kind} inputs ({fit} Hamiltonian): {cache}")
    print_field_summary(staged)

    if args.stage_only:
        return

    mean, err = run_afqmc(staged, args)
    print(f"AFQMC/{staged.trial.kind.upper()} energy: {mean:.10f} +/- {err:.10f} Ha")


if __name__ == "__main__":
    main()
