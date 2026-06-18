from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyscf import gto, scf

from trot.afqmc import Afqmc
from trot.staging import load as load_staged
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
        return load_staged(cache)

    if args.trial != "uhf":
        require_ccpy_driver()

    mol = build_mol(args.pyscf_verbose)
    mf = run_stable_uhf(mol, args)

    if args.trial == "uhf":
        return stage(
            mf,
            chol_cut=args.chol_cut,
            cache=cache,
            overwrite=args.overwrite_cache,
            verbose=args.stage_verbose,
        )

    _cc_method, order = TRIAL_TO_CCPY[args.trial]
    driver = build_ccpy_driver(mf, args)
    return stage_from_ccpy(
        driver,
        mf,
        order=order,
        chol_cut=args.chol_cut,
        cache=cache,
        overwrite=args.overwrite_cache,
        verbose=args.stage_verbose,
    )


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
    print(f"Staged CH/{staged.trial.kind} inputs: {cache}")

    if args.stage_only:
        return

    mean, err = run_afqmc(staged, args)
    print(f"AFQMC/{staged.trial.kind.upper()} energy: {mean:.10f} +/- {err:.10f} Ha")


if __name__ == "__main__":
    main()
