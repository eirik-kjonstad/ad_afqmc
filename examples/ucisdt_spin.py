from pyscf import gto, scf

from trot.afqmc import Afqmc
from trot.staging import stage_from_ccpy

try:
    from ccpy.drivers.driver import Driver
except Exception as exc:
    raise RuntimeError(
        "This example requires ccpy. Please install ccpy to run it.") from exc

mol = gto.M(
    atom="H 0 0 0; F 1.7 0.0 0.0",
    basis="6-31g",
    symmetry="c1",
    verbose=3,
)


def run_stable_uhf(mf, conv_tol=1.0e-12, max_stability_cycles=5):
    mf.run(conv_tol=conv_tol)
    if not mf.converged:
        raise RuntimeError("UHF did not converge.")

    for _ in range(max_stability_cycles):
        mo_i, _, stable_i, _ = mf.stability(return_status=True)
        if stable_i:
            return mf

        dm1 = mf.make_rdm1(mo_i, mf.mo_occ)
        mf.run(dm1, conv_tol=conv_tol)
        if not mf.converged:
            raise RuntimeError(
                "UHF did not converge after stability analysis.")

    _, _, stable_i, _ = mf.stability(return_status=True)
    if stable_i:
        return mf

    raise RuntimeError(
        "UHF did not converge to an internally stable determinant.")


mf = scf.UHF(mol)
mf.max_cycle = 300
mf = run_stable_uhf(mf)

cc_driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
cc_driver.options["amp_convergence"] = 1.0e-12
cc_driver.options["energy_convergence"] = 1.0e-12
cc_driver.options["RHF_symmetry"] = False
cc_driver.run_cc(method="ccsdt")

staged = stage_from_ccpy(cc_driver, mf, order=3,
                         chol_cut=1.0e-14, verbose=False)

spin_lambda = 0.1
af = Afqmc(
    staged,
    decomposition="spin",
    spin_decomposition_lambda=spin_lambda,
    diagnostics_dir="diag_spin",
)
af.walker_kind = "unrestricted"
af.n_walkers = 80
af.n_eql_blocks = 50
af.n_blocks = 200
af.dt = 0.005
af.mixed_precision = False
af.seed = 7
mean, err = af.kernel()
print(
    f"spin-decomposed AFQMC/UCISDT energy "
    f"(lambda={spin_lambda:g}): {mean:.10f} +/- {err:.10f} Ha"
)
