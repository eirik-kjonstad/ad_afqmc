"""
UCISDT-guided AFQMC example using the charge/Sz Cholesky HS decomposition.

This follows examples/ucisdt.py, but selects the optional two-channel
charge/Sz auxiliary-field decomposition. The decomposition is spin block
diagonal, so this example uses unrestricted walkers.
"""

from pyscf import gto, scf

from trot.afqmc import Afqmc
from trot.staging import stage_from_ccpy

try:
    from ccpy.drivers.driver import Driver
except Exception as exc:
    raise RuntimeError(
        "This example requires ccpy. Please install ccpy to run it.") from exc


def run_uhf(mol, *, dm0=None):
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1.0e-12
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("UHF did not converge.")
    return mf


def rerun_from_stable_uhf_orbitals(mf):
    stability_result = mf.stability(return_status=True)
    if len(stability_result) != 4:
        raise RuntimeError(
            "UHF stability analysis did not return internal stability status.")

    mo_i = stability_result[0]
    dm0 = mf.make_rdm1(mo_i, mf.mo_occ)
    mf = run_uhf(mf.mol, dm0=dm0)

    stability_result = mf.stability(return_status=True)
    if len(stability_result) != 4:
        raise RuntimeError(
            "UHF stability analysis did not return internal stability status.")
    stable_i = stability_result[2]
    if not stable_i:
        raise RuntimeError(
            "UHF stability rerun did not converge to an internally stable solution.")

    return mf


mol = gto.M(
    atom="H 0 0 0; F 1.7 0.0 0.0",
    basis="6-31g",
    symmetry="c1",
    verbose=3,
)

mf = run_uhf(mol)
mf = rerun_from_stable_uhf_orbitals(mf)

cc_driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
cc_driver.options["amp_convergence"] = 1.0e-12
cc_driver.options["energy_convergence"] = 1.0e-12
cc_driver.options["RHF_symmetry"] = False
cc_driver.run_cc(method="ccsdt")

staged = stage_from_ccpy(cc_driver, mf, order=3,
                         chol_cut=1.0e-14, verbose=False)

af = Afqmc(staged)
af.walker_kind = "unrestricted"
af.hs_decomposition = "charge_sz"
af.n_walkers = 80
af.n_eql_blocks = 400
af.n_blocks = 1000
af.dt = 0.0005
af.seed = 7
mean, err = af.kernel()
print(f"Charge/Sz AFQMC/UCISDT energy: {mean:.10f} +/- {err:.10f} Ha")
