import numpy as np

def prepare_ucc_amplitudes_from_ccpy(driver, get_amps=False):
    order = driver.operator_params["order"]

    print(f"Preparing amplitudes.npz for ad_afqmc. Order: {order}")

    if order < 2:
        print(f"Amplitude order={order} not yet supported.")
        exit(1)
    elif order > 3:
        print(f"NB: Will only extract amplitudes up to order=3")

    from copy import deepcopy

    R = deepcopy(driver.T)
    system = driver.system

    # ai => ia
    R.a = R.a.transpose(1,0)
    R.b = R.b.transpose(1,0)

    # abij => iajb
    R.aa = R.aa.transpose(2, 0, 3, 1)
    R.bb = R.bb.transpose(2, 0, 3, 1)
    R.ab = R.ab.transpose(2, 0, 3, 1)

    # Define amplitudes without introducing symmetries
    C1a = R.a 
    C1b = R.b 

    C2aa = (
        R.aa + 2.0e0 * np.einsum("ia,jb->iajb", R.a, R.a)
    )

    C2bb = (
        R.bb + 2.0e0 * np.einsum("ia,jb->iajb", R.b, R.b)
    )

    C2ab = (
        R.ab + np.einsum("ia,jb->iajb", R.a, R.b)
    )

    # Introduce symmetries:
    # and a,b/i,j antisymmetries are fulfilled
    # ai,bj symmetries already fulfilled
    C2aa = completely_antisymmetrize_2(C2aa) 
    C2bb = completely_antisymmetrize_2(C2bb) 

    if order == 2:
        np.savez(
            "amplitudes.npz",
            ci1a=C1a,
            ci1b=C1b,
            ci2aa=C2aa,
            ci2ab=C2ab,
            ci2bb=C2bb,
        )

        if get_amps:
            return np.float64(1.0), (C1a, C1b), (C2aa.transpose(0,2,1,3), C2ab.transpose(0,2,1,3), C2bb.transpose(0,2,1,3))

    elif order == 3:
        # abcijk => iajbkc
        R.aaa = R.aaa.transpose(3, 0, 4, 1, 5, 2)
        R.aab = R.aab.transpose(3, 0, 4, 1, 5, 2)
        R.abb = R.abb.transpose(3, 0, 4, 1, 5, 2)
        R.bbb = R.bbb.transpose(3, 0, 4, 1, 5, 2)

        # Define amplitudes without introducing symmetries
        C3aaa = (
            R.aaa + 
            9.0e0 * np.einsum("iajb,kc->iajbkc", R.aa, R.a) + 
            6.0e0 * np.einsum("ia,jb,kc->iajbkc", R.a, R.a, R.a)
        )

        C3bbb = (
            R.bbb +
            9.0e0 * np.einsum("iajb,kc->iajbkc", R.bb, R.b) + 
            6.0e0 * np.einsum("ia,jb,kc->iajbkc", R.b, R.b, R.b)
        )

        C3aab = (
            R.aab + 
            np.einsum("iajb,kc->iajbkc", R.aa, R.b) + 
            4.0e0 * np.einsum("ia,jbkc->iajbkc", R.a, R.ab) + 
            2.0e0 * np.einsum("ia,jb,kc->iajbkc", R.a, R.a, R.b)
        )

        C3abb = (
            R.abb + 
            np.einsum("ia,jbkc->iajbkc", R.a, R.bb) + 
            4.0e0 * np.einsum("iajb,kc->iajbkc", R.ab, R.b) + 
            2.0e0 * np.einsum("ia,jb,kc->iajbkc", R.a, R.b, R.b)
        )

        # Introduce symmetries:
        # Make sure ai,bj,ck symmetries
        # and a,b,c/i,j,k antisymmetries are fulfilled
        C3aaa = completely_symmetrize_pairs_3(C3aaa)
        C3aaa = completely_antisymmetrize_3(C3aaa)

        C3bbb = completely_symmetrize_pairs_3(C3bbb)
        C3bbb = completely_antisymmetrize_3(C3bbb)

        X3aab = 0.5e0 * ( C3aab + C3aab.transpose(2,3,0,1,4,5) ) 
        X3aab = 0.5e0 * ( X3aab - X3aab.transpose(0,3,2,1,4,5) )

        X3abb = 0.5e0 * ( C3abb + C3abb.transpose(0,1,4,5,2,3) )
        X3abb = 0.5e0 * ( X3abb - X3abb.transpose(0,1,2,5,4,3) )

        C3aab = X3aab
        C3abb = X3abb

        np.savez(
            "amplitudes.npz",
            ci1a=C1a,
            ci1b=C1b,
            ci2aa=C2aa,
            ci2ab=C2ab,
            ci2bb=C2bb,
            ci3aaa=C3aaa,
            ci3aab=C3aab,
            ci3abb=C3abb,
            ci3bbb=C3bbb,
        )

        print(f"a: {np.linalg.norm(C1a)}")
        print(f"b: {np.linalg.norm(C1b)}")
        print(f"aa: {np.linalg.norm(C2aa)}")
        print(f"ab: {np.linalg.norm(C2ab)}")
        print(f"bb: {np.linalg.norm(C2bb)}")
        print(f"aaa: {np.linalg.norm(C3aaa)}")
        print(f"aab: {np.linalg.norm(C3aab)}")
        print(f"abb: {np.linalg.norm(C3abb)}")
        print(f"bbb: {np.linalg.norm(C3bbb)}")

        if get_amps:
            return [np.float64(1.0), (C1a, C1b), (C2aa.transpose(0,2,1,3), C2ab.transpose(0,2,1,3), C2bb.transpose(0,2,1,3)), 
            (C3aaa.transpose(0,2,4,1,3,5), C3aab.transpose(0,2,4,1,3,5), C3abb.transpose(0,2,4,1,3,5), C3bbb.transpose(0,2,4,1,3,5))]


def completely_symmetrize_pairs_3(X3):
    Y3 = 1/6 * (
        X3 +
        X3.transpose(0, 1, 4, 5, 2, 3) +
        X3.transpose(2, 3, 4, 5, 0, 1) +
        X3.transpose(2, 3, 0, 1, 4, 5) +
        X3.transpose(4, 5, 0, 1, 2, 3) +
        X3.transpose(4, 5, 2, 3, 0, 1)
    )
    return Y3

def completely_antisymmetrize_3(X3):
    Y3 = 1/6 * (
        X3
        - X3.transpose(0, 1, 2, 5, 4, 3) 
        + X3.transpose(0, 3, 2, 5, 4, 1) 
        - X3.transpose(0, 3, 2, 1, 4, 5) 
        + X3.transpose(0, 5, 2, 1, 4, 3) 
        - X3.transpose(0, 5, 2, 3, 4, 1) 
    )

    return Y3


def completely_symmetrize_pairs_2(X2):
    Y2 = 0.5e0 * (X2 + X2.transpose(2,3,0,1))
    return Y2

def completely_antisymmetrize_2(X2):
    Y2 = 0.5 * (X2 - X2.transpose(0, 3, 2, 1))
    return Y2 
