import numpy as np

def prepare_ucc_amplitudes_from_ccpy(driver, get_amps=False, order=-1):
    if order == -1:
        order = driver.operator_params["order"]

    print(f"Order: {order}")
    print(f"Order of CC calculation: {driver.operator_params["order"]}")

    print(f"Preparing amplitudes.npz for ad_afqmc. Order: {order}")

    if order < 2:
        print(f"Amplitude order={order} not yet supported.")
        exit(1)
    elif order > 4:
        print(f"NB: Will only extract amplitudes up to order=4")

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

    # Set connected contributions to zero if user requests
    # disconnected contributions of higher orders
    no_a = np.shape(R.a)[0] 
    nv_a = np.shape(R.a)[1] 
    no_b = np.shape(R.b)[0] 
    nv_b = np.shape(R.b)[1] 

    if order > 2 and driver.operator_params["order"] < 3:
        R.aaa = np.zeros((nv_a, nv_a, nv_a, no_a, no_a, no_a))
        R.aab = np.zeros((nv_a, nv_a, nv_b, no_a, no_a, no_b))
        R.abb = np.zeros((nv_a, nv_b, nv_b, no_a, no_b, no_b))
        R.bbb = np.zeros((nv_b, nv_b, nv_b, no_b, no_b, no_b))

    if order > 3 and driver.operator_params["order"] < 4:
        R.aaaa = np.zeros((nv_a, nv_a, nv_a, nv_a, no_a, no_a, no_a, no_a))
        R.aaab = np.zeros((nv_a, nv_a, nv_a, nv_b, no_a, no_a, no_a, no_b))
        R.aabb = np.zeros((nv_a, nv_a, nv_b, nv_b, no_a, no_a, no_b, no_b))
        R.abbb = np.zeros((nv_a, nv_b, nv_b, nv_b, no_a, no_b, no_b, no_b))
        R.bbbb = np.zeros((nv_b, nv_b, nv_b, nv_b, no_b, no_b, no_b, no_b))

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

    elif order > 2:
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

        if order == 3:

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

        elif order == 4:
            # start making quadruples
            # c4 = t4 + t1 t3 + 1/2 t2^2 + 1/2 t1^2 t2 + 1/24 t1^4

            # abcdijkl => iajbkcld
            R.aaaa = R.aaaa.transpose(4, 0, 5, 1, 6, 2, 7, 3)
            R.aaab = R.aaab.transpose(4, 0, 5, 1, 6, 2, 7, 3)
            R.aabb = R.aabb.transpose(4, 0, 5, 1, 6, 2, 7, 3)
            R.abbb = R.abbb.transpose(4, 0, 5, 1, 6, 2, 7, 3)
            R.bbbb = R.bbbb.transpose(4, 0, 5, 1, 6, 2, 7, 3)

            C4aaaa = (
                R.aaaa + 
                16.0e0 * np.einsum("ia,jbkcld->iajbkcld", R.a, R.aaa) + 
                18.0e0 * np.einsum("iajb,kcld->iajbkcld", R.aa, R.aa) + 
                72.0e0 * np.einsum("ia,jb,kcld->iajbkcld", R.a, R.a, R.aa) + 
                24.0e0 * np.einsum("ia,jb,kc,ld->iajbkcld", R.a, R.a, R.a, R.a)
            )

            C4bbbb = (
                R.bbbb + 
                16.0e0 * np.einsum("ia,jbkcld->iajbkcld", R.b, R.bbb) + 
                18.0e0 * np.einsum("iajb,kcld->iajbkcld", R.bb, R.bb) + 
                72.0e0 * np.einsum("ia,jb,kcld->iajbkcld", R.b, R.b, R.bb) + 
                24.0e0 * np.einsum("ia,jb,kc,ld->iajbkcld", R.b, R.b, R.b, R.b)
            )

            C4aaab = (
                R.aaab + 
                np.einsum("iajbkc,ld->iajbkcld", R.aaa, R.b) + 
                9.0e0 * np.einsum("ia,jbkcld->iajbkcld", R.a, R.aab) + 
                9.0e0 * np.einsum("iajb,kcld->iajbkcld", R.aa, R.ab) + 
                9.0e0 * np.einsum("iajb,kc,ld->iajbkcld", R.aa, R.a, R.b) + 
                18.0e0 * np.einsum("ia,jb,kcld->iajbkcld", R.a, R.a, R.ab) + 
                6.0e0 * np.einsum("ia,jb,kc,ld->iajbkcld", R.a, R.a, R.a, R.b)
            )  

            C4abbb = (
                R.abbb + 
                9.0e0 * np.einsum("iajbkc,ld->iajbkcld", R.abb, R.b) + 
                np.einsum("ia,jbkcld->iajbkcld", R.a, R.bbb) + 
                9.0e0 * np.einsum("iajb,kcld->iajbkcld", R.ab, R.bb) + 
                18.0e0 * np.einsum("iajb,kc,ld->iajbkcld", R.ab, R.b, R.b) + 
                9.0e0 * np.einsum("ia,jb,kcld->iajbkcld", R.a, R.b, R.bb) + 
                6.0e0 * np.einsum("ia,jb,kc,ld->iajbkcld", R.a, R.b, R.b, R.b)
            )

            C4aabb = (
                R.aabb +
                4.0e0 * np.einsum("iajbkc,ld->iajbkcld", R.aab, R.b) + 
                4.0e0 * np.einsum("ia,jbkcld->iajbkcld", R.a, R.abb) + 
                np.einsum("iajb,kcld->iajbkcld", R.aa, R.bb) + 
                8.0e0 * np.einsum("iakc,jbld->iajbkcld", R.ab, R.ab) + 
                2.0e0 * np.einsum("ia,jb,kcld->iajbkcld", R.a, R.a, R.bb) + 
                2.0e0 * np.einsum("iajb,kc,ld->iajbkcld", R.aa, R.b, R.b) + 
                16.0e0 * np.einsum("ia,kc,jbld->iajbkcld", R.a, R.b, R.ab) + 
                4.0e0 * np.einsum("ia,jb,kc,ld->iajbkcld", R.a, R.a, R.b, R.b)
            )

            C4aaaa = completely_symmetrize_pairs_4(C4aaaa)
            C4aaaa = completely_antisymmetrize_4(C4aaaa)

            C4bbbb = completely_symmetrize_pairs_4(C4bbbb)
            C4bbbb = completely_antisymmetrize_4(C4bbbb)

            C4aaab = 1/6 * (
                C4aaab +
                C4aaab.transpose(0, 1, 4, 5, 2, 3, 6, 7) +
                C4aaab.transpose(2, 3, 4, 5, 0, 1, 6, 7) +
                C4aaab.transpose(2, 3, 0, 1, 4, 5, 6, 7) +
                C4aaab.transpose(4, 5, 0, 1, 2, 3, 6, 7) +
                C4aaab.transpose(4, 5, 2, 3, 0, 1, 6, 7)
            )

            C4aaab = 1/6 * (
                C4aaab
                - C4aaab.transpose(0, 1, 2, 5, 4, 3, 6, 7) 
                + C4aaab.transpose(0, 3, 2, 5, 4, 1, 6, 7) 
                - C4aaab.transpose(0, 3, 2, 1, 4, 5, 6, 7) 
                + C4aaab.transpose(0, 5, 2, 1, 4, 3, 6, 7) 
                - C4aaab.transpose(0, 5, 2, 3, 4, 1, 6, 7) 
            )

            C4abbb = 1/6 * (
                C4abbb +
                C4abbb.transpose(0, 1, 2, 3, 6, 7, 4, 5) +
                C4abbb.transpose(0, 1, 4, 5, 6, 7, 2, 3) +
                C4abbb.transpose(0, 1, 4, 5, 2, 3, 6, 7) +
                C4abbb.transpose(0, 1, 6, 7, 2, 3, 4, 5) +
                C4abbb.transpose(0, 1, 6, 7, 4, 5, 2, 3)
            )

            C4abbb = 1/6 * (
                C4abbb
                - C4abbb.transpose(0, 1, 2, 3, 4, 7, 6, 5) 
                + C4abbb.transpose(0, 1, 2, 5, 4, 7, 6, 3) 
                - C4abbb.transpose(0, 1, 2, 5, 4, 3, 6, 7) 
                + C4abbb.transpose(0, 1, 2, 7, 4, 3, 6, 5) 
                - C4abbb.transpose(0, 1, 2, 7, 4, 5, 6, 3) 
            )

            C4aabb = 1/2 * (C4aabb + C4aabb.transpose(2,3,0,1,4,5,6,7))
            C4aabb = 1/2 * (C4aabb + C4aabb.transpose(0,1,2,3,6,7,4,5))

            C4aabb = 1/2 * (C4aabb - C4aabb.transpose(0,3,2,1,4,5,6,7))
            C4aabb = 1/2 * (C4aabb - C4aabb.transpose(0,1,2,3,4,7,6,5))

            print(f"a: {np.linalg.norm(C1a)}")
            print(f"b: {np.linalg.norm(C1b)}")
            print(f"aa: {np.linalg.norm(C2aa)}")
            print(f"ab: {np.linalg.norm(C2ab)}")
            print(f"bb: {np.linalg.norm(C2bb)}")
            print(f"aaa: {np.linalg.norm(C3aaa)}")
            print(f"aab: {np.linalg.norm(C3aab)}")
            print(f"abb: {np.linalg.norm(C3abb)}")
            print(f"bbb: {np.linalg.norm(C3bbb)}")

            print(f"aaaa: {np.linalg.norm(C4aaaa)}")
            print(f"aaab: {np.linalg.norm(C4aaab)}")
            print(f"aabb: {np.linalg.norm(C4aabb)}")
            print(f"abbb: {np.linalg.norm(C4abbb)}")
            print(f"bbbb: {np.linalg.norm(C4bbbb)}")

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
                ci4aaaa=C4aaaa,
                ci4aaab=C4aaab,
                ci4aabb=C4aabb,
                ci4abbb=C4abbb,
                ci4bbbb=C4bbbb,
            )

            if get_amps:
                return [np.float64(1.0), (C1a, C1b), (C2aa.transpose(0,2,1,3), C2ab.transpose(0,2,1,3), C2bb.transpose(0,2,1,3)), 
                (C3aaa.transpose(0,2,4,1,3,5), C3aab.transpose(0,2,4,1,3,5), C3abb.transpose(0,2,4,1,3,5), C3bbb.transpose(0,2,4,1,3,5)),
                (C4aaaa.transpose(0,2,4,6,1,3,5,7), C4aaab.transpose(0,2,4,6,1,3,5,7), C4aabb.transpose(0,2,4,6,1,3,5,7), 
                    C4abbb.transpose(0,2,4,6,1,3,5,7), C4bbbb.transpose(0,2,4,6,1,3,5,7))]


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

import itertools

def completely_symmetrize_pairs_4(X4):
    """
    Symmetrize a tensor X4 with indices X_ia jb kc ld 
    over the four index pairs: (i,a), (j,b), (k,c), (l,d).
    The result is averaged over all 24 permutations of these pairs.
    """
    perms = itertools.permutations([0, 1, 2, 3])  # Each number represents a pair index
    sym_tensor = 0
    for p in perms:
        # Build the new order of axes:
        # For each pair index in the permutation p, add its two corresponding axes.
        axes = []
        for pair in p:
            axes.extend([2 * pair, 2 * pair + 1])
        sym_tensor += X4.transpose(*axes)
    return sym_tensor / 24

def permutation_sign(perm):
    """
    Compute the sign (parity) of a permutation.
    The permutation is given as a tuple, e.g., (1, 3, 5, 7).
    Here we compute the sign by counting inversions.
    """
    sign = 1
    perm_list = list(perm)
    for i in range(len(perm_list)):
        for j in range(i + 1, len(perm_list)):
            if perm_list[i] > perm_list[j]:
                sign *= -1
    return sign

def completely_antisymmetrize_4(X4):
    """
    Completely antisymmetrize a tensor X4 with indices X_ia jb kc ld 
    over the four indices a, b, c, d.
    
    It is assumed that the antisymmetrized indices are in positions 1, 3, 5, and 7.
    The routine averages over all 24 permutations of these indices,
    weighting each term by the sign of the permutation.
    """
    # Positions to antisymmetrize: 1 (a), 3 (b), 5 (c), 7 (d)
    antisym_axes = [1, 3, 5, 7]
    antiX = 0
    # Loop over all 24 permutations of the indices [1, 3, 5, 7]
    for perm in itertools.permutations(antisym_axes):
        # Construct the new axis order.
        # The indices i, j, k, l (positions 0, 2, 4, 6) remain unchanged.
        # The antisym indices are permuted into positions 1, 3, 5, and 7.
        new_order = [0, perm[0], 2, perm[1], 4, perm[2], 6, perm[3]]
        sign = permutation_sign(perm)
        antiX += sign * X4.transpose(new_order)
    return antiX / 24

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

import numpy as np
import itertools
import math

def permutation_sign(perm):
    """
    Compute the sign (parity) of a permutation.
    perm is a tuple or list representing a permutation of 0,1,...,n-1.
    Returns +1 for even and -1 for odd permutations.
    """
    sign = 1
    perm = list(perm)
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                sign *= -1
    return sign

def symmetrize_pairs(tensor, pairs):
    """
    Symmetrize the tensor over a list of pairs of axes.
    
    For each pair (i, j) in pairs the routine will ensure that 
    the resulting tensor is symmetric under exchange of axes i and j.
    
    Parameters:
      tensor : np.ndarray
          The tensor (as a NumPy array) to symmetrize.
      pairs : list of tuples of ints
          A list of pairs of axis numbers. For example, 
          if you want to symmetrize over (a,i), (b,j) and (d,l)
          and these correspond to axes (0,4), (1,5) and (3,7) respectively,
          then use: [(0,4), (1,5), (3,7)].
          
    Returns:
      sym_tensor : np.ndarray
          The symmetrized tensor.
    """
    # Initialize an array for accumulating the sum.
    sym_tensor = np.zeros_like(tensor, dtype=float)
    num_pairs = len(pairs)
    
    # There are 2^(number of pairs) combinations (swap or not for each pair).
    for swap_flags in itertools.product([False, True], repeat=num_pairs):
        # Start with the identity permutation of all axes.
        perm = list(range(tensor.ndim))
        # For each pair, if the flag is True, swap the corresponding axes.
        for flag, (i, j) in zip(swap_flags, pairs):
            if flag:
                perm[i], perm[j] = perm[j], perm[i]
        # Add the permuted tensor
        sym_tensor += np.transpose(tensor, axes=perm)
        
    return sym_tensor / (2**num_pairs)

def antisymmetrize_indices(tensor, antisym_axes):
    """
    Antisymmetrize the tensor over a given list of axes.
    
    That is, for the specified axes the routine produces a tensor
    whose value is the weighted (by permutation sign) average over all 
    permutations of those axes.
    
    Parameters:
      tensor : np.ndarray
          The tensor (as a NumPy array) to antisymmetrize.
      antisym_axes : list of ints
          A list of axis numbers to antisymmetrize.
          For example, if you wish to antisymmetrize indices a, b, d and
          these correspond to axes [0, 1, 3] in your tensor, pass [0, 1, 3].
          
    Returns:
      antisym_tensor : np.ndarray
          The antisymmetrized tensor.
    """
    antisym_tensor = np.zeros_like(tensor, dtype=float)
    n = len(antisym_axes)
    
    # Loop over all permutations of the positions in antisym_axes.
    # Here we are permuting the order in which the chosen axes appear.
    for sigma in itertools.permutations(range(n)):
        sgn = permutation_sign(sigma)
        # Build a full permutation of tensor axes:
        # For axes not in antisym_axes, leave them fixed.
        full_perm = list(range(tensor.ndim))
        # For each position in antisym_axes, assign the axis from the permutation.
        for pos, ax in enumerate(antisym_axes):
            # We want the axis in position "ax" to come from 
            # the original axis at antisym_axes[sigma[pos]].
            full_perm[ax] = antisym_axes[sigma[pos]]
        antisym_tensor += sgn * np.transpose(tensor, axes=full_perm)
        
    return antisym_tensor / math.factorial(n)

