"""
Run AFQMC with a ccpy-derived CC trial and the global phaseless projection.

This example builds a small UHF reference, runs ccpy coupled cluster, converts
the ccpy T amplitudes into a TROT UCISD/UCISDT/UCISDTQ trial with
``stage_from_ccpy``, and enables the coefficient-space global phaseless update.

Examples
--------
Fast smoke run with a CCSD -> UCISD trial:

    python examples/ccpy_global_phaseless.py --cc-method ccsd --ci-order 2

Use a CCSDT calculation but keep a UCISDT trial:

    python examples/ccpy_global_phaseless.py --cc-method ccsdt --ci-order 3

Compare to the standard walker-local phaseless projection:

    python examples/ccpy_global_phaseless.py --standard-phaseless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trot.afqmc import Afqmc
from trot.staging import stage_from_ccpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ccpy CC trial AFQMC example with optional global phaseless projection."
    )
    parser.add_argument(
        "--cc-method",
        choices=("ccsd", "ccsdt"),
        default="ccsd",
        help="ccpy coupled-cluster method used to generate T amplitudes.",
    )
    parser.add_argument(
        "--ci-order",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="CI excitation order staged for the AFQMC trial: 2=UCISD, 3=UCISDT, 4=UCISDTQ.",
    )
    parser.add_argument("--n-walkers", type=int, default=80)
    parser.add_argument("--n-eql-blocks", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--n-prop-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Dimensionless r in R = r * dt * ||w_tilde / S||.",
    )
    parser.add_argument(
        "--standard-phaseless",
        action="store_true",
        help="Disable the new global projection and use the standard local cosine projection.",
    )
    parser.add_argument(
        "--no-gauge-fix",
        action="store_true",
        help="Leave projected weights in their raw global complex phase.",
    )
    parser.add_argument(
        "--measure-energy-with-uhf",
        action="store_true",
        help="Use the embedded UHF determinant for reweighted block energy estimates.",
    )
    return parser.parse_args()


def _build_molecule():
    return gto.M(
        atom="""
        N  0.0000000000  0.0000000000  0.0000000000
        H  1.0225900000  0.0000000000  0.0000000000
        H -0.2281193615  0.9968208791  0.0000000000
        """,
        basis="sto-6g",
        spin=1,
        symmetry="c1",
        verbose=3,
    )


def _run_ccpy(mf, method: str):
    try:
        from ccpy.drivers.driver import Driver
    except Exception as exc:
        raise RuntimeError("This example requires ccpy. Please install ccpy to run it.") from exc

    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["amp_convergence"] = 1.0e-10
    driver.options["energy_convergence"] = 1.0e-10
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method=method)
    return driver


def main() -> None:
    args = _parse_args()
    if args.ci_order == 4 and args.cc_method != "ccsdt":
        raise ValueError("A UCISDTQ trial needs at least --cc-method ccsdt for this example.")

    mol = _build_molecule()
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.run(conv_tol=1.0e-12)
    if not mf.converged:
        raise RuntimeError("UHF did not converge.")

    cc_driver = _run_ccpy(mf, args.cc_method)
    staged = stage_from_ccpy(
        cc_driver,
        mf,
        order=args.ci_order,
        chol_cut=1.0e-14,
        verbose=True,
    )

    af = Afqmc(staged)
    af.walker_kind = "unrestricted"
    af.mixed_precision = False
    af.dt = args.dt
    af.n_walkers = args.n_walkers
    af.n_eql_blocks = args.n_eql_blocks
    af.n_blocks = args.n_blocks
    af.n_prop_steps = args.n_prop_steps
    af.seed = args.seed
    af.global_phaseless_projection = not args.standard_phaseless
    af.global_phaseless_budget_scale = args.budget_scale
    af.global_phaseless_gauge_fix = not args.no_gauge_fix
    af.measure_energy_with_uhf = args.measure_energy_with_uhf

    projection = "global coefficient-space" if af.global_phaseless_projection else "standard local"
    print(f"\nccpy method        : {args.cc_method}")
    print(f"staged trial       : {staged.trial.kind}")
    print(f"phaseless update   : {projection}")
    print(f"budget scale       : {af.global_phaseless_budget_scale}")
    print(f"global gauge fix   : {af.global_phaseless_gauge_fix}")
    print(f"measure with UHF   : {af.measure_energy_with_uhf}\n")

    mean, err = af.kernel()
    print(f"\nAFQMC/{staged.trial.kind.upper()} energy: {mean:.10f} +/- {err:.10f} Ha")


if __name__ == "__main__":
    main()
