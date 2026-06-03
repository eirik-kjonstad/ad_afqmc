from pyscf import gto, scf

from trot.afqmc import Afqmc

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

spin_lambda = 1.0
af = Afqmc(
    mf,
    decomposition="spin",
    spin_decomposition_lambda=spin_lambda,
)
af.n_walkers = 80
af.n_eql_blocks = 200
af.n_blocks = 3000
af.seed = 7
af.walker_kind = "unrestricted"
mean, err = af.kernel()
print(
    f"spin-decomposed AFQMC/UHF energy "
    f"(lambda={spin_lambda:g}): {mean:.10f} +/- {err:.10f} Ha"
)
