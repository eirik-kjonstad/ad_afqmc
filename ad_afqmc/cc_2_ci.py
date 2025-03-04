import numpy as np

# Unrestricted CI coefficients from ccpy amplitudes
def uci_from_ccpy(mf, ccpy_driver, c_order):
    t_order = ccpy_driver.operator_params["order"]

    norb = len(mf.mo_occ[0])
    t1a = np.transpose(ccpy_driver.T.a, (1,0))
    t1b = np.transpose(ccpy_driver.T.b, (1,0))

    nfrozen = norb - np.shape(t1a)[0] - np.shape(t1a)[1]
    
    t2aa = np.transpose(ccpy_driver.T.aa, (2,3,0,1)) / 2.0
    t2ab = np.transpose(ccpy_driver.T.ab, (2,3,0,1))
    t2bb = np.transpose(ccpy_driver.T.bb, (2,3,0,1)) / 2.0
    
    if t_order >= 3:
        t3aaa = np.transpose(ccpy_driver.T.aaa, (3,4,5,0,1,2)) / 6.0
        t3aba = np.transpose(ccpy_driver.T.aab, (3,5,4,0,2,1)) / 2.0
        t3bab = np.transpose(ccpy_driver.T.abb, (4,3,5,1,0,2)) / 2.0
        t3bbb = np.transpose(ccpy_driver.T.bbb, (3,4,5,0,1,2)) / 6.0
    
    t1 = {
        "a": t1a,
        "b": t1b
    }
    
    t2 = {
        "aa": t2aa,
        "ab": t2ab,
        "bb": t2bb
    }

    if t_order >= 3:
        t3 = {
            "aaa": t3aaa,
            "aba": t3aba,
            "bab": t3bab,
            "bbb": t3bbb,
        }
    else:
        t3 = None

    if t_order >= 4:
        print("Not handle")
        crash

    t4 = None

    t = (t1, t2, t3, t4)
    ci = t2c_unrestricted(mf, nfrozen, t1, t2, t3, t4, c_order)

    return t, ci

# Unrestricted CI coefficients from ebcc amplitudes
def uci_from_ebcc(mf, ebcc_obj, c_order):

    # n frozen
    norb = len(mf.mo_occ[0])
    nfrozen = norb - np.shape(ebcc_obj.t1["aa"])[0] - np.shape(ebcc_obj.t1["aa"])[1]

    # t order
    t_order = 2

    ## t3?
    try:
        if hasattr(ebcc_obj, 't3'):
            t_order = 3
    except:
        pass

    ## t4?
    try:
        if hasattr(ebcc_obj, 't4'):
            t_order = 4
    except:
        pass

    # t
    t1 = {
        "a": ebcc_obj.t1["aa"],
        "b": ebcc_obj.t1["bb"],
    }
    
    t2 = {
        "aa": ebcc_obj.t2["aaaa"],
        "ab": ebcc_obj.t2["abab"],
        "bb": ebcc_obj.t2["bbbb"],
    }
    
    if t_order >= 3:
        t3 = {
            "aaa": ebcc_obj.t3["aaaaaa"],
            "aba": ebcc_obj.t3["abaaba"],
            "bab": ebcc_obj.t3["babbab"],
            "bbb": ebcc_obj.t3["bbbbbb"],
        }
    else:
        t3 = None

    if t_order >= 4:
        print("Not handle")
        crash

    t4 = None

    t = (t1, t2, t3, t4)
    
    # t to ci
    ci = t2c_unrestricted(mf, nfrozen, t1, t2, t3, t4, c_order)

    return t, ci

# Unrestricted CI coefficients from amplitudes using ebcc
def t2c_unrestricted(mf, nfrozen, t1, t2, t3, t4, c_order):

    from ebcc import util
    from ebcc.ham import Space
    from ebcc.util import Namespace
    from ebcc.ext.fci import _amplitudes_to_coefficients_unrestricted

    nOa, nVa = np.shape(t1["a"])
    nOb, nVb = np.shape(t1["b"])

    amps_uhf: Namespace[USpinArrayType] = util.Namespace()
    amps_uhf.t1 = util.Namespace(
        aa=t1["a"],
        bb=t1["b"],
    )
    amps_uhf.t2 = util.Namespace(
        aaaa=t2["aa"],
        abab=t2["ab"],
        bbbb=t2["bb"],
    )
    if t3 != None:
        amps_uhf.t3 = util.Namespace(
            aaaaaa=t3["aaa"],
            babbab=t3["bab"],
            abaaba=t3["aba"],
            bbbbbb=t3["bbb"],
        )
    else:
        amps_uhf.t3 = util.Namespace(
            aaaaaa=np.zeros((nOa,nOa,nOa,nVa,nVa,nVa)),
            babbab=np.zeros((nOb,nOa,nOb,nVb,nVa,nVb)),
            abaaba=np.zeros((nOa,nOb,nOa,nVa,nVb,nVa)),
            bbbbbb=np.zeros((nOb,nOb,nOb,nVb,nVb,nVb)),
        )

    if t4 != None:
        amps_uhf.t4 = util.Namespace(
            aaaaaaaa=t4["aaaa"],
            aaabaaab=t4["aaab"],
            abababab=t4["abab"],
            abbbabbb=t4["baaa"],
            bbbbbbbb=t4["aaaa"],
        )
    else:
        amps_uhf.t4 = util.Namespace(
            aaaaaaaa=np.zeros((nOa,nOa,nOa,nOa,nVa,nVa,nVa,nVa)),
            aaabaaab=np.zeros((nOa,nOa,nOa,nOb,nVa,nVa,nVa,nVb)),
            abababab=np.zeros((nOa,nOb,nOa,nOb,nVa,nVb,nVa,nVb)),
            abbbabbb=np.zeros((nOa,nOb,nOb,nOb,nVa,nVb,nVb,nVb)),
            bbbbbbbb=np.zeros((nOb,nOb,nOb,nOb,nVb,nVb,nVb,nVb)),
        )

    occupied_a = mf.mo_occ[0] > 0
    frozen_a = np.zeros_like(mf.mo_occ[0])
    active_a = np.ones_like(mf.mo_occ[0])
    frozen_a[:nfrozen] = True
    active_a[:nfrozen] = False
    space_a = Space(occupied_a, frozen_a, active_a)

    occupied_b = mf.mo_occ[1] > 0
    frozen_b = np.zeros_like(mf.mo_occ[1])
    active_b = np.ones_like(mf.mo_occ[1])
    frozen_b[:nfrozen] = True
    active_b[:nfrozen] = False
    space_b = Space(occupied_b, frozen_b, active_b)

    space = (space_a, space_b)

    #space = (
    #    Space(
    #        mf.mo_occ[0] > 0,
    #        np.zeros_like(mf.mo_occ[0]),
    #        np.ones_like(mf.mo_occ[0]),
    #    ),
    #    Space(
    #        mf.mo_occ[1] > 0,
    #        np.zeros_like(mf.mo_occ[1]),
    #        np.ones_like(mf.mo_occ[1]),
    #    ),
    #)

    assert(c_order >= 2)
    ci = _amplitudes_to_coefficients_unrestricted(amps_uhf, max_order=c_order)

    ci1a = ci["c1"]["aa"]
    ci1b = ci["c1"]["bb"]

    ci2aa = ci["c2"]["aaaa"].transpose(0, 2, 1, 3)
    ci2ab = ci["c2"]["abab"].transpose(0, 2, 1, 3)
    ci2bb = ci["c2"]["bbbb"].transpose(0, 2, 1, 3)

    if c_order >= 3:
        ci3aaa = ci["c3"]["aaaaaa"].transpose(0, 3, 1, 4, 2, 5) 
        ci3aab = ci["c3"]["abaaba"].transpose(0, 3, 2, 5, 1, 4) 
        ci3abb = ci["c3"]["babbab"].transpose(1, 4, 0, 3, 2, 5)
        ci3bbb = ci["c3"]["bbbbbb"].transpose(0, 3, 1, 4, 2, 5)
    else:
        ci3aaa = None 
        ci3aab = None 
        ci3abb = None 
        ci3bbb = None 

    if c_order >= 4:
        ci4aaaa = ci["c4"]["aaaaaaaa"].transpose(0,4,1,5,2,6,3,7)
        ci4aaab = ci["c4"]["aaabaaab"].transpose(0,4,1,5,2,6,3,7)
        ci4aabb = ci["c4"]["abababab"].transpose(0,4,2,6,1,5,3,7)
        ci4abbb = ci["c4"]["abbbabbb"].transpose(0,4,1,5,2,6,3,7)
        ci4bbbb = ci["c4"]["bbbbbbbb"].transpose(0,4,1,5,2,6,3,7)
    else:
        ci4aaaa = None
        ci4aaab = None
        ci4aabb = None
        ci4abbb = None
        ci4bbbb = None

    np.savez(
        "amplitudes.npz",
        ci1a=ci1a,
        ci1b=ci1b,
        ci2aa=ci2aa,
        ci2ab=ci2ab,
        ci2bb=ci2bb,
        ci3aaa=ci3aaa,
        ci3aab=ci3aab,
        ci3abb=ci3abb,
        ci3bbb=ci3bbb,
        ci4aaaa=ci4aaaa,
        ci4aaab=ci4aaab,
        ci4aabb=ci4aabb,
        ci4abbb=ci4abbb,
        ci4bbbb=ci4bbbb,
    )

    ci1 = (ci1a, ci1b)
    ci2 = (ci2aa, ci2ab, ci2bb)
    ci3 = (ci3aaa, ci3aab, ci3abb, ci3bbb)
    ci4 = (ci4aaaa, ci4aaab, ci4aabb, ci4abbb, ci4bbbb)

    return (ci1, ci2, ci3, ci4)

# Compare the ci coefficients obtained from ebcc and ccpy_interface
def debug_t2c(mf, norb_frozen, method):
    import os
    from ccpy.drivers.driver import Driver
    from ad_afqmc import cc_2_ci
    from ad_afqmc import ccpy_interface

    if method not in ["ccsd", "ccsdt", "CCSD", "CCSDT"]:
        print(f"Unknown '{method}' method, use CCSD or CCSDT")
        return

    driver = Driver.from_pyscf(mf, nfrozen=norb_frozen, uhf=True)
    driver.options["RHF_symmetry"] = False
    driver.amp_convergence = 1e-6
    driver.run_cc(method="ccsdt")

    t, (c1_1, c2_1, c3_1, c4) = cc_2_ci.uci_from_ccpy(mf, driver, 3)
    os.rename("amplitudes.npz", "amplitudes_tmp.npz")
    c1 = np.load("amplitudes_tmp.npz")

    [c0, c1_2, c2_2, c3_2] = ccpy_interface.prepare_ucc_amplitudes_from_ccpy(driver, True)
    c2 = np.load("amplitudes.npz")

    m = 0.0

    keys = [
        "ci1a", "ci1b", "ci2aa", "ci2ab", "ci2bb",
        "ci3aaa", "ci3aab", "ci3abb", "ci3bbb",
        "ci4aaaa", "ci4aaab", "ci4aabb", "ci4abbb", "ci4bbbb"
    ]

    print()
    for key in keys:
        try:
            e = np.max(np.abs(c1[key] - c2[key]))
            m = max(m, e)
            print(f"{key:<7}: {e:8.1e}")
        except KeyError:
            pass
        except ValueError:
            pass

    assert(m < 1e-16)

# Compare amplitudes coming from ebcc and ccpy, in addition to the
# resulting CI coefficients
def debug_amplitudes(mf, method, c_order, threshold):
    import numpy as np
    from ebcc import UEBCC, NullLogger
    from ccpy.drivers.driver import Driver

    if method not in ["CCSD", "CCSDT"]:
        print(f"Unknown '{method}' method, use CCSD or CCSDT")
        return

    # EBCC
    ebcc = UEBCC(mf, ansatz=method)
    ebcc.options.t_tol = 1e-12
    ebcc.options.e_tol = 1e-8
    ebcc.options.size_diis = 12
    ebcc.kernel()
    
    # CCPY 
    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["RHF_symmetry"] = False
    driver.amp_convergence = 1e-12
    driver.run_cc(method=method)
    t_order = driver.operator_params["order"]

    #c_order = 3
    (t1_1, t2_1, t3_1, t4_1), (ci1_1, ci2_1, ci3_1, ci4_1) = uci_from_ebcc(mf, ebcc, c_order)
    (t1_2, t2_2, t3_2, t4_2), (ci1_2, ci2_2, ci3_2, ci4_2) = uci_from_ccpy(mf, driver, c_order)
    l1 = ["a","b"]
    l2 = ["aa","ab","bb"]
    l3 = ["aaa","aba","bab","bbb"]
    l4 = ["aaaa","aaab","abab","abbb","bbbb"]

    m = 0.0
    
    t_orders = {
        1: (t1_1, t1_2, l1),
        2: (t2_1, t2_2, l2),
        3: (t3_1, t3_2, l3),
        4: (t4_1, t4_2, l4),
    }
    
    print("\n{:4} {:8} {:8}".format("","  Diff  ", "   Max    "))
    for order, (t1, t2, labels) in t_orders.items():
        if t_order < order:
            continue
    
        print(f"\nT{order}")
        for l in labels:
            diff = np.max(np.abs(t1[l] - t2[l]))
            max_t1 = max(np.max(np.abs(t1[l])), np.max(np.abs(t2[l])))
            m = max(m, diff)
            print(f"{l:>4} {diff:8.1e} {max_t1:8.1e}")

    assert(m < threshold)

    l1 = ["a","b"]
    l2 = ["aa","ab","bb"]
    l3 = ["aaa","aab","abb","bbb"]
    l4 = ["aaaa","aaab","aabb","abbb","bbbb"]

    c_orders = {
        1: (ci1_1, ci1_2, l1),
        2: (ci2_1, ci2_2, l2),
        3: (ci3_1, ci3_2, l3),
        4: (ci4_1, ci4_2, l4),
    }

    for order, (c1, c2, labels) in c_orders.items():
        if c_order < order:
            continue

        print(f"\nC{order}")
        for i, l in zip(range(order+1),labels):
            diff = np.max(np.abs(c1[i] - c2[i]))
            max_t1 = max(np.max(np.abs(c1[i])), np.max(np.abs(c2[i])))
            m = max(m, diff)
            print(f"{l:>4} {diff:8.1e} {max_t1:8.1e}")
