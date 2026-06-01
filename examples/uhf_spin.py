from pyscf import gto, scf

from trot.afqmc import Afqmc

try:
    from ccpy.drivers.driver import Driver
except Exception as exc:
    raise RuntimeError(
        "This example requires ccpy. Please install ccpy to run it.") from exc


def build_mf():
    mol = gto.M(
        atom="""
        N  -1.67119571   -1.44021737    0.00000000
        H  -2.12619571   -0.65213425    0.00000000
        H  -0.76119571   -1.44021737    0.00000000
        """,
        spin=1,
        basis="6-31g",
        symmetry="c1",
        verbose=3,
    )

    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.run(conv_tol=1.0e-12)
    if not mf.converged:
        raise RuntimeError("UHF did not converge.")

    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1, conv_tol=1.0e-12)
    if not mf.converged:
        raise RuntimeError("UHF stability rerun did not converge.")
    mf.stability()
    return mf


def _last_scalar(value):
    if isinstance(value, (list, tuple)):
        return _last_scalar(value[-1])
    try:
        return float(value)
    except TypeError:
        return float(value[-1])


def ccpy_total_energy(driver):
    if hasattr(driver, "system") and hasattr(driver.system, "reference_energy"):
        e_ref = _last_scalar(driver.system.reference_energy)
        if hasattr(driver, "correlation_energy"):
            return e_ref + _last_scalar(driver.correlation_energy)
    if hasattr(driver, "energy"):
        return _last_scalar(driver.energy)
    raise AttributeError("Could not find a ccpy total energy on the driver.")


def run_ccsdt_reference(mf):
    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["amp_convergence"] = 1.0e-12
    driver.options["energy_convergence"] = 1.0e-12
    driver.options["RHF_symmetry"] = False

    driver.run_cc(method="ccsdt")

    return ccpy_total_energy(driver)


def run_afqmc(mf, *, hs_decomposition: str):
    af = Afqmc(mf, hs_decomposition=hs_decomposition)
    af.walker_kind = "unrestricted"

    # Keep these identical when comparing charge and spin decompositions.
    af.seed = 7
    af.n_walkers = 100
    af.n_eql_blocks = 100
    af.n_blocks = 4000

#    if hs_decomposition == "spin":
#        af.dt = 0.0005
#        af.n_prop_steps = 500

    return af.kernel()


if __name__ == "__main__":
    mf = build_mf()

    e_cc4 = run_ccsdt_reference(mf)
#    e_charge, err_charge = run_afqmc(mf, hs_decomposition="charge")
    e_spin, err_spin = run_afqmc(mf, hs_decomposition="spin")

    print("\nCCSDT reference:")
    print(f"  E = {e_cc4}")
#    print("\nCharge decomposition:")
#    print(f"  E = {e_charge} +/- {err_charge}")
#    print(f"  Delta vs CCSDT = {e_charge - e_cc4}")
    print("Spin decomposition:")
    print(f"  E = {e_spin} +/- {err_spin}")
    print(f"  Delta vs CCSDT = {e_spin - e_cc4}")
