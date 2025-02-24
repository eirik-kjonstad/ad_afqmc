import numpy as np

def ucisdt_from_ccpy(mf, ccpy_driver):

    norb = len(mf.mo_occ[0])
    t1a = np.transpose(ccpy_driver.T.a, (1,0))
    t1b = np.transpose(ccpy_driver.T.b, (1,0))

    nfrozen = norb - np.shape(t1a)[0] - np.shape(t1a)[1]
    
    t2aa = np.transpose(ccpy_driver.T.aa, (2,3,0,1)) / 2.0
    t2ab = np.transpose(ccpy_driver.T.ab, (2,3,0,1))
    t2bb = np.transpose(ccpy_driver.T.bb, (2,3,0,1)) / 2.0
    
    t3aaa = np.transpose(ccpy_driver.T.aaa, (3,4,5,0,1,2)) / 6.0
    t3aba = np.transpose(ccpy_driver.T.aab, (3,5,4,0,2,1)) / 2.0
    t3bab = np.transpose(ccpy_driver.T.abb, (4,3,5,1,0,2)) / 2.0
    t3bbb = np.transpose(ccpy_driver.T.bbb, (3,4,5,0,1,2)) / 6.0
    
    t1 = {
        "aa": t1a,
        "bb": t1b
    }
    
    t2 = {
        "aaaa": t2aa,
        "abab": t2ab,
        "bbbb": t2bb
    }
    
    t3 = {
        "aaaaaa": t3aaa,
        "abaaba": t3aba,
        "babbab": t3bab,
        "bbbbbb": t3bbb,
    }
    
    ci = t2c_unrestricted(mf, nfrozen, t1, t2, t3)

    return ci

def ucisdt_from_ebcc(mf, ebcc_obj):

    norb = len(mf.mo_occ[0])
    nfrozen = norb - np.shape(ebcc_obj.t1["aa"])[0] - np.shape(ebcc_obj.t1["aa"])[1]

    t1 = {
        "aa": ebcc_obj.t1["aa"],
        "bb": ebcc_obj.t1["bb"],
    }
    
    t2 = {
        "aaaa": ebcc_obj.t2["aaaa"],
        "abab": ebcc_obj.t2["abab"],
        "bbbb": ebcc_obj.t2["bbbb"],
    }
    
    t3 = {
        "aaaaaa": ebcc_obj.t3["aaaaaa"],
        "abaaba": ebcc_obj.t3["abaaba"],
        "babbab": ebcc_obj.t3["babbab"],
        "bbbbbb": ebcc_obj.t3["bbbbbb"],
    }
    
    ci = t2c_unrestricted(mf, nfrozen, t1, t2, t3)

    return ci

def t2c_unrestricted(mf, nfrozen, t1, t2, t3):

    from ebcc import util
    from ebcc.ham import Space
    from ebcc.util import Namespace
    from ebcc.ext.fci import _amplitudes_to_coefficients_unrestricted

    amps_uhf: Namespace[USpinArrayType] = util.Namespace()
    amps_uhf.t1 = util.Namespace(
        aa=t1["aa"],
        bb=t1["bb"],
    )
    amps_uhf.t2 = util.Namespace(
        aaaa=t2["aaaa"],
        abab=t2["abab"],
        bbbb=t2["bbbb"],
    )
    amps_uhf.t3 = util.Namespace(
        aaaaaa=t3["aaaaaa"],
        babbab=t3["babbab"],
        abaaba=t3["abaaba"],
        bbbbbb=t3["bbbbbb"],
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

    ci = _amplitudes_to_coefficients_unrestricted(amps_uhf, max_order=3)

    ci1a = ci["c1"]["aa"]
    ci1b = ci["c1"]["bb"]
    ci2aa = ci["c2"]["aaaa"].transpose(0, 2, 1, 3)
    ci2ab = ci["c2"]["abab"].transpose(0, 2, 1, 3)
    ci2bb = ci["c2"]["bbbb"].transpose(0, 2, 1, 3)
    ci3aaa = ci["c3"]["aaaaaa"].transpose(0, 3, 1, 4, 2, 5) 
    ci3aab = ci["c3"]["abaaba"].transpose(0, 3, 2, 5, 1, 4) 
    ci3abb = ci["c3"]["babbab"].transpose(1, 4, 0, 3, 2, 5)
    ci3bbb = ci["c3"]["bbbbbb"].transpose(0, 3, 1, 4, 2, 5)

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
    )

    return ci1a, ci1b, ci2aa, ci2ab, ci2bb, ci3aaa, ci3aab, ci3abb, ci3bbb

# Compare amplitudes coming from ebcc and ccpy, in addition to the
# resulting CI coefficients
def debug_t2c(mf):
    import numpy as np
    
    from ebcc import UEBCC, NullLogger
    from ccpy.drivers.driver import Driver
    
    # EBCC
    ebcc = UEBCC(mf, ansatz="CCSDT")
    ebcc.options.t_tol = 1e-6
    ebcc.options.e_tol = 1e-8
    ebcc.options.size_diis = 12
    ebcc.kernel()
    
    # CCPY 
    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["RHF_symmetry"] = False
    driver.amp_convergence = 1e-12
    driver.run_cc(method="ccsdt")
    
    t1a = np.transpose(driver.T.a, (1,0))
    t1b = np.transpose(driver.T.b, (1,0))
    
    print("\nT1")
    print("a: {:16.6e}".format(np.max(np.abs(t1a - ebcc.t1["aa"]))))
    print("b: {:16.6e}".format(np.max(np.abs(t1b - ebcc.t1["bb"]))))
    
    t2aa = np.transpose(driver.T.aa, (2,3,0,1)) / 2.0
    t2ab = np.transpose(driver.T.ab, (2,3,0,1))
    t2bb = np.transpose(driver.T.bb, (2,3,0,1)) / 2.0
    print("\nT2")
    print("aa: {:16.6e}".format(np.max(np.abs(t2aa - ebcc.t2["aaaa"]))))
    print("ab: {:16.6e}".format(np.max(np.abs(t2ab - ebcc.t2["abab"]))))
    print("bb: {:16.6e}".format(np.max(np.abs(t2bb - ebcc.t2["bbbb"]))))
    
    t3aaa = np.transpose(driver.T.aaa, (3,4,5,0,1,2)) / 6.0
    t3aba = np.transpose(driver.T.aab, (3,5,4,0,2,1)) / 2.0
    t3bab = np.transpose(driver.T.abb, (4,3,5,1,0,2)) / 2.0
    t3bbb = np.transpose(driver.T.bbb, (3,4,5,0,1,2)) / 6.0
    print("\nT3")
    print("aaa: {:16.6e}".format(np.max(np.abs(t3aaa - ebcc.t3["aaaaaa"]))))
    print("aab: {:16.6e}".format(np.max(np.abs(t3aba - ebcc.t3["abaaba"]))))
    print("abb: {:16.6e}".format(np.max(np.abs(t3bab - ebcc.t3["babbab"]))))
    print("bbb: {:16.6e}".format(np.max(np.abs(t3bbb - ebcc.t3["bbbbbb"]))))
    
    t1 = {
        "aa": ebcc.t1["aa"],
        "bb": ebcc.t1["bb"]
    }
    
    t2 = {
        "aaaa": ebcc.t2["aaaa"],
        "abab": ebcc.t2["abab"],
        "bbbb": ebcc.t2["bbbb"]
    }
    
    t3 = {
        "aaaaaa": ebcc.t3["aaaaaa"],
        "abaaba": ebcc.t3["abaaba"],
        "babbab": ebcc.t3["babbab"],
        "bbbbbb": ebcc.t3["bbbbbb"],
    }
    
    ci_1 = t2c_unrestricted(mf, 0, t1, t2, t3)
   
    t1_2 = {
        "aa": t1a,
        "bb": t1b
    }
    
    t2_2 = {
        "aaaa": t2aa,
        "abab": t2ab,
        "bbbb": t2bb 
    }
    
    t3_2 = {
        "aaaaaa": t3aaa,
        "abaaba": t3aba,
        "babbab": t3bab,
        "bbbbbb": t3bbb,
    }
    
    ci_2 = t2c_unrestricted(mf, 0, t1_2, t2_2, t3_2)
    
    #print("T")
    #for e1, e2 in zip(t1, t1_2):
    #    print("{:16.6e}".format(np.max(np.abs(t1[e1]-t1_2[e2]))))
    #for e1, e2 in zip(t2, t2_2):
    #    print("{:16.6e}".format(np.max(np.abs(t2[e1]-t2_2[e2]))))
    #for e1, e2 in zip(t3, t3_2):
    #    print("{:16.6e}".format(np.max(np.abs(t3[e1]-t3_2[e2]))))
    
    print("\nC")
    for e1, e2 in zip(ci_1, ci_2):
        print("{:16.6e}".format(np.max(e1-e2)))
    
