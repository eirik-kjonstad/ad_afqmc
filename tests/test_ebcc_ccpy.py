import numpy as np
from pyscf import cc, ci, gto, scf
from ad_afqmc import cc_2_ci

mol1 = gto.M(
    atom="""
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.971161
    H        0.000000   -1.000000   -0.971161
    """,
    basis="sto-3g",
    spin=0,
    charge=0,
    verbose=3,
    symmetry="C1",
    unit="A",
)

mol2 = gto.M(
    atom="""
    N        0.000000    0.000000  0.0 
    N        0.000000    0.000000  1.0
    """,
    basis="sto-3g",
    spin=1,
    charge=-1,
    verbose=3,
    symmetry="C1",
    unit="A",
)

mols = [mol1, mol2]

for mol in mols:
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    
    cc_2_ci.debug_amplitudes(mf, "CCSD", 3, 1e-7)
    cc_2_ci.debug_amplitudes(mf, "CCSDT", 3, 2e-5)
