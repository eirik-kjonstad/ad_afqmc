import numpy as np
from pyscf import cc, ci, gto, scf
from ad_afqmc import cc_2_ci

mol = gto.M(
    atom="""
    B 0.0 0.0 0.0
    B  0.0 0.0 1.6
    """,
    basis="sto-3g",
    spin=0,
    charge=0,
    verbose=3,
    symmetry="C1",
    unit="A",
)

mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
mf.mo_coeff = [mf.mo_coeff[0],mf.mo_coeff[0]]

cc_2_ci.debug_t2c(mf, 0, "CCSD")
cc_2_ci.debug_t2c(mf, 0, "CCSDT")

