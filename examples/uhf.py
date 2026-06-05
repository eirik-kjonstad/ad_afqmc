from pyscf import gto, scf

from trot.afqmc import Afqmc

mol = gto.M(
    atom="""
    N        0.0000000000      0.0000000000      0.0000000000
    H        1.0225900000      0.0000000000      0.0000000000
    H       -0.2281193615      0.9968208791      0.0000000000
    N        0.0000000000      0.0000000000      5.0000000000
    H        1.0225900000      0.0000000000      5.0000000000
    H       -0.2281193615      0.9968208791      5.0000000000
    """,
    spin=2,
    basis="6-31g",
    verbose=3,
)

mf = scf.UHF(mol).newton()
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

af = Afqmc(mf)
af.n_walkers = 200
af.mixed_precision = False
af.walker_kind = "unrestricted"
mean, err = af.kernel()
