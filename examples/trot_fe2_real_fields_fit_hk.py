from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from pyscf.tools.fcidump import to_scf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from trot.afqmc import Afqmc
    from trot.prop.types import QmcParams
    from trot.staging import load as load_staged
    from trot.staging import stage
    from trot.staging import stage_from_ccpy
    from trot.staging import StagedMfOrCc
    from trot.staging import TrialInput
    from trot.staging import _normalize_real_field_centers
    from trot.staging import _stage_ham_input_from_fcidump
    from trot.staging import analyze_hk_from_fcidump
    from trot.staging import build_model_fcidump_from_fcidump
    from trot.staging import fcidump_pair_spectrum
except ImportError as exc:
    raise RuntimeError(
        "This example requires trot on PYTHONPATH. For a local checkout, run with "
        f"PYTHONPATH={ROOT}:$PYTHONPATH."
    ) from exc


DEFAULT_FE_CENTERS_5_BAND = "2:7,13:18"
DEFAULT_FE_CENTERS_2_BAND = "2:4,13:15"
DEFAULT_FE2_OCCUPATION_STRING = "22aaaaa222222bbbbb22"


def _cache_tag_from_center_spec(spec: Any, *, prefix: str) -> str:
    if spec is None:
        return ""
    text = str(spec)
    text = text.replace(" ", "")
    text = text.replace(":", "-")
    text = text.replace(",", "_")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "")
    text = text.strip("_")
    return f"{prefix}{text}" if text else ""


def _cache_tag_from_fcidump(path: Path, *, default_name: str) -> str:
    stem = path.expanduser().stem
    return "" if path.name == default_name else f"_{stem}"


def _fcidump_norb(path: Path) -> int:
    from pyscf.tools import fcidump as pyscf_fcidump

    return int(pyscf_fcidump.read(str(path.expanduser()))["NORB"])


def _flatten_centers(centers: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sorted({int(orb) for center in centers for orb in center}))


def _nelec_ms2_from_occupation_string(
    occupation_string: str, orbitals: tuple[int, ...]
) -> tuple[int, int]:
    n_alpha = 0
    n_beta = 0
    for orb in orbitals:
        site = occupation_string[int(orb)]
        if site == "2":
            n_alpha += 1
            n_beta += 1
        elif site == "a":
            n_alpha += 1
        elif site == "b":
            n_beta += 1
        elif site == "0":
            pass
        else:
            raise ValueError(f"Unsupported occupation character {site!r} at orbital {orb}.")
    return n_alpha + n_beta, n_alpha - n_beta


def _occupation_arrays_from_string(occupation_string: str) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.zeros(len(occupation_string), dtype=float)
    beta = np.zeros(len(occupation_string), dtype=float)
    for idx, site in enumerate(occupation_string):
        if site == "2":
            alpha[idx] = 1.0
            beta[idx] = 1.0
        elif site == "a":
            alpha[idx] = 1.0
        elif site == "b":
            beta[idx] = 1.0
        elif site != "0":
            raise ValueError(f"Unsupported occupation character {site!r} at orbital {idx}.")
    return alpha, beta


def _project_occupation_string(occupation_string: str, orbitals: tuple[int, ...]) -> str:
    return "".join(occupation_string[int(orb)] for orb in orbitals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage and run Fe2S2 with a center-targeted real-field route. "
            "The default Fe-center ranges are the five-orbital localized blocks "
            f"{DEFAULT_FE_CENTERS_5_BAND}."
        )
    )
    parser.add_argument(
        "fcidump",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("Fe2S2.FCIDUMP"),
        help="Path to the Fe2S2 FCIDUMP file.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Real-field staged HDF5 cache path.",
    )
    parser.add_argument(
        "--standard-cache",
        type=Path,
        default=None,
        help="Ordinary Cholesky staged HDF5 cache path used with --compare-standard.",
    )
    parser.add_argument(
        "--method",
        choices=("uhf", "ccsd", "ccsdt", "ccsdtq"),
        default="ccsd",
        help=(
            "Trial/preparation method. Use 'uhf' for a UHF Slater trial without ccpy; "
            "CC methods run ccpy and stage a CI expansion from the requested --order."
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="CI excitation order staged from ccpy amplitudes. Ignored for --method uhf.",
    )
    parser.add_argument(
        "--fe-band-model",
        type=int,
        choices=(2, 5),
        default=5,
        help=(
            "Preset Fe-center orbital count. The 5-band preset uses "
            f"{DEFAULT_FE_CENTERS_5_BAND}; the 2-band preset uses "
            f"{DEFAULT_FE_CENTERS_2_BAND}. Ignored when --real-field-centers is set."
        ),
    )
    parser.add_argument(
        "--real-field-centers",
        default=None,
        help=(
            "Localized orbital ranges for the Fe centers. Use Python slice syntax per center, "
            "comma separated; e.g. '2:7,13:18'. Overrides --fe-band-model."
        ),
    )
    parser.add_argument(
        "--ligand-centers",
        default=None,
        help=(
            "Ligand/S orbital ranges. HK analysis uses these as the ligand subset; "
            "AFQMC staging also extracts their positive onsite U as real spin fields. "
            "Use Python slice syntax, comma separated; e.g. '7:9,18:20'."
        ),
    )
    parser.add_argument(
        "--no-model-extraction",
        action="store_true",
        help=(
            "Do not split out selected Fe/ligand tensor blocks before factorization. "
            "Supported with --real-field-method charge_spin or UHF charge/spin methods, "
            "where it factorizes the full tensor directly."
        ),
    )
    parser.add_argument(
        "--occupation-string",
        default=None,
        help=(
            "Optional explicit site occupation string for the UHF guess using 2/a/b/0. "
            "Required for reduced model FCIDUMPs whose orbital count differs from the original Fe2 file."
        ),
    )
    parser.add_argument(
        "--print-uhf-populations",
        action="store_true",
        help="Print diagonal alpha/beta populations and local moments after UHF convergence.",
    )
    parser.add_argument(
        "--real-field-method",
        choices=(
            "hk_density",
            "local_exact",
            "kanamori_sign",
            "kanamori_real",
            "kanamori_uj",
            "charge_spin",
            "uhf_charge_spin",
            "uhf_charge_spin_blocks",
            "uhf_charge_spin_unrham",
            "uhf_local_real_then_charge_spin_unrham",
            "kanamori_sign_full",
        ),
        default="local_exact",
        help="Real-field staging route for the specified centers.",
    )
    parser.add_argument(
        "--chkfile",
        type=Path,
        default=Path(__file__).with_name("uhf.chk"),
        help="PySCF checkpoint file for the UHF calculation.",
    )
    parser.add_argument(
        "--load-chkfile",
        action="store_true",
        help=(
            "Load saved UHF SCF data from --chkfile when it exists, instead of rerunning "
            "the UHF kernel. The loaded object is marked converged for post-SCF methods."
        ),
    )
    parser.add_argument("--chol-cut", type=float, default=1.0e-5)
    parser.add_argument("--amp-convergence", type=float, default=1.0e-5)
    parser.add_argument("--energy-convergence", type=float, default=1.0e-5)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--compare-standard",
        action="store_true",
        help="Also stage and optionally run the ordinary Cholesky decomposition.",
    )
    parser.add_argument(
        "--vanilla-afqmc",
        action="store_true",
        help=(
            "Run only the ordinary Cholesky AFQMC route. This ignores real-field and "
            "ligand-center extraction for staging."
        ),
    )
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument(
        "--analyze-hk",
        action="store_true",
        help=(
            "Analyze HK-like tensor weights in the FCIDUMP basis for the selected Fe centers "
            "and exit before UHF/CC/AFQMC."
        ),
    )
    parser.add_argument(
        "--write-model-fcidump",
        type=Path,
        default=None,
        help=(
            "Write a reduced/model FCIDUMP from the selected Fe centers and ligand centers, "
            "then exit before UHF/CC/AFQMC."
        ),
    )
    parser.add_argument(
        "--fci",
        action="store_true",
        help=(
            "Run a reference calculation from the FCIDUMP and exit before UHF/CC/AFQMC. "
            "Defaults to block2 DMRG; use --fci-solver pyscf for the old exact solver."
        ),
    )
    parser.add_argument(
        "--fci-solver",
        choices=("block2", "pyscf"),
        default="block2",
        help="Reference solver used by --fci.",
    )
    parser.add_argument(
        "--fci-nroots",
        type=int,
        default=1,
        help=(
            "Number of PySCF FCI roots to compute. "
            "block2 DMRG always computes the lowest root."
        ),
    )
    parser.add_argument(
        "--fci-basis",
        choices=("input", "restricted_mo", "uhf_natural"),
        default="input",
        help=(
            "Orbital basis for FCI. 'input' uses the FCIDUMP basis; 'restricted_mo' "
            "runs RHF and uses PySCF fci.FCI(mf, mf.mo_coeff, singlet=False); "
            "'uhf_natural' runs UHF and rotates to common spin-summed UHF natural orbitals."
        ),
    )
    parser.add_argument(
        "--fci-conv-tol",
        type=float,
        default=1.0e-10,
        help="FCI Davidson convergence tolerance.",
    )
    parser.add_argument(
        "--fci-max-cycle",
        type=int,
        default=100,
        help="Maximum FCI Davidson iterations.",
    )
    parser.add_argument(
        "--fci-max-space",
        type=int,
        default=12,
        help="Maximum FCI Davidson subspace size.",
    )
    parser.add_argument(
        "--fci-pspace-size",
        type=int,
        default=400,
        help="PySCF FCI p-space size.",
    )
    parser.add_argument(
        "--fci-max-memory",
        type=float,
        default=4000.0,
        help=(
            "Approximate PySCF FCI memory cap in MB. "
            "Also used as block2 stack memory unless overridden."
        ),
    )
    parser.add_argument(
        "--block2-bond-dim",
        type=int,
        default=1000,
        help="Maximum block2 DMRG MPS bond dimension.",
    )
    parser.add_argument(
        "--block2-bond-dims",
        default=None,
        help=(
            "Optional comma-separated block2 bond-dimension schedule. "
            "The last value is repeated to --block2-n-sweeps."
        ),
    )
    parser.add_argument(
        "--block2-n-sweeps",
        type=int,
        default=20,
        help="Maximum number of block2 DMRG sweeps.",
    )
    parser.add_argument(
        "--block2-tol",
        type=float,
        default=1.0e-8,
        help="block2 DMRG energy convergence tolerance.",
    )
    parser.add_argument(
        "--block2-davidson-tol",
        type=float,
        default=1.0e-6,
        help="block2 Davidson residual threshold used for every sweep.",
    )
    parser.add_argument(
        "--block2-noises",
        default=None,
        help=(
            "Optional comma-separated block2 noise schedule. "
            "Defaults to small noise in early sweeps and zero in final sweeps."
        ),
    )
    parser.add_argument(
        "--block2-cutoff",
        type=float,
        default=1.0e-14,
        help="block2 density-matrix truncation cutoff.",
    )
    parser.add_argument(
        "--block2-mpo-cutoff",
        type=float,
        default=1.0e-20,
        help="block2 MPO SVD cutoff.",
    )
    parser.add_argument(
        "--block2-integral-cutoff",
        type=float,
        default=1.0e-20,
        help="block2 integral cutoff when constructing the MPO.",
    )
    parser.add_argument(
        "--block2-memory",
        type=float,
        default=None,
        help="block2 stack memory in MB. Defaults to --fci-max-memory.",
    )
    parser.add_argument(
        "--block2-scratch",
        type=Path,
        default=None,
        help="block2 scratch directory. Defaults to a per-FCIDUMP directory under the system temp dir.",
    )
    parser.add_argument(
        "--block2-threads",
        type=int,
        default=1,
        help="Number of block2 OpenMP threads.",
    )
    parser.add_argument(
        "--block2-point-group",
        choices=("c1", "d2h"),
        default="c1",
        help="Point group used to interpret FCIDUMP ORBSYM labels for block2.",
    )
    parser.add_argument(
        "--block2-reorder",
        choices=("none", "fiedler", "irrep", "gaopt"),
        default="fiedler",
        help="Orbital ordering used by block2 MPO construction.",
    )
    parser.add_argument(
        "--model-orbitals",
        default=None,
        help=(
            "Explicit model orbital ranges. Defaults to Fe centers plus --ligand-centers. "
            "Use Python slice syntax, comma separated."
        ),
    )
    parser.add_argument(
        "--model-h2",
        choices=("onsite", "onsite_bridge_density", "selected_full"),
        default="onsite_bridge_density",
        help=(
            "Two-body content of --write-model-fcidump. 'onsite_bridge_density' keeps "
            "onsite U and Fe-ligand density Vdp terms."
        ),
    )
    parser.add_argument(
        "--model-h1-correction",
        choices=("none", "reference_fock"),
        default="none",
        help=(
            "Optional one-body correction for --write-model-fcidump. 'reference_fock' "
            "folds the spin-averaged Fock contribution of discarded two-body terms into h1 "
            "using --model-occupation-string/--occupation-string."
        ),
    )
    parser.add_argument(
        "--model-nelec",
        type=int,
        default=None,
        help="Override NELEC for --write-model-fcidump.",
    )
    parser.add_argument(
        "--model-ms2",
        type=int,
        default=None,
        help="Override MS2 for --write-model-fcidump.",
    )
    parser.add_argument(
        "--model-occupation-string",
        default=None,
        help=(
            "Occupation string used to infer NELEC/MS2 for --write-model-fcidump. "
            "Defaults to the Fe2 broken-symmetry guess."
        ),
    )
    parser.add_argument("--verbose-stage", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--n-walkers", type=int, default=200)
    parser.add_argument("--n-eql-blocks", type=int, default=20)
    parser.add_argument("--n-blocks", type=int, default=200)
    parser.add_argument("--n-prop-steps", type=int, default=50)
    parser.add_argument("--n-exp-terms", type=int, default=6)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-raw",
        type=Path,
        default=None,
        help="Write AFQMC block samples to this whitespace table for live monitoring.",
    )
    args = parser.parse_args()
    custom_centers = any(
        arg == "--real-field-centers" or arg.startswith("--real-field-centers=")
        for arg in sys.argv[1:]
    )
    if args.no_model_extraction and args.real_field_method not in {
        "charge_spin",
        "uhf_charge_spin",
        "uhf_charge_spin_blocks",
        "uhf_charge_spin_unrham",
    }:
        raise ValueError(
            "--no-model-extraction is currently supported only with charge_spin "
            "or UHF charge/spin methods."
        )
    if args.no_model_extraction:
        args.real_field_centers = None
        custom_centers = True
    elif args.real_field_centers is None:
        args.real_field_centers = (
            DEFAULT_FE_CENTERS_2_BAND
            if args.fe_band_model == 2
            else DEFAULT_FE_CENTERS_5_BAND
        )
    ligand_tag = _cache_tag_from_center_spec(args.ligand_centers, prefix="lig")
    if args.cache is None:
        trial_tag = "uhf" if args.method == "uhf" else f"{args.method}_order{args.order}"
        ligand_suffix = f"_{ligand_tag}" if ligand_tag else ""
        cache_name = (
            f"fe2_real_fields_{args.real_field_method}_full_{trial_tag}_staged.h5"
            if args.no_model_extraction
            else (
                f"fe2_real_fields_{args.real_field_method}_custom{ligand_suffix}_{trial_tag}_staged.h5"
                if custom_centers
                else (
                    f"fe2_real_fields_{args.real_field_method}_{args.fe_band_model}"
                    f"band{ligand_suffix}_{trial_tag}_staged.h5"
                )
            )
        )
        args.cache = Path(__file__).with_name(cache_name)
    if args.standard_cache is None:
        trial_tag = "uhf" if args.method == "uhf" else f"{args.method}_order{args.order}"
        source_tag = _cache_tag_from_fcidump(args.fcidump, default_name="Fe2S2.FCIDUMP")
        args.standard_cache = Path(__file__).with_name(
            f"fe2_standard{source_tag}_{trial_tag}_staged.h5"
        )
    if args.vanilla_afqmc:
        args.compare_standard = True
    return args


def prepare_fcidump_mf(fcidump_path: Path):
    mf = to_scf(str(fcidump_path), molpro_orbsym=True, mf=None)

    intor_symmetric = mf.mol.intor_symmetric
    mf.mol.intor_symmetric = lambda intor, **kwargs: (
        np.eye(mf.mol.nao) if intor == "int1e_ovlp" else intor_symmetric(intor, **kwargs)
    )

    mf.mol.verbose = 4
    mf.mol._symm_orig = np.zeros((3,))
    mf.mol._symm_axes = np.eye(3)
    mf.mol.irrep_name = ["A"]
    mf.mol.groupname = "C1"
    mf.mol.symmetry = "C1"

    return mf


def fe2_broken_symmetry_guess(umf, occupation_string: str | None = None) -> np.ndarray:
    n_sites = int(umf.mol.nao)
    occupation_string = occupation_string or DEFAULT_FE2_OCCUPATION_STRING
    if len(occupation_string) != n_sites:
        raise ValueError(
            "Fe2 occupation string has length "
            f"{len(occupation_string)}, but the FCIDUMP has {n_sites} orbitals."
        )

    occupations = np.array(
        [
            {
                "2": [1.0, 1.0],
                "a": [1.0, 0.0],
                "b": [0.0, 1.0],
                "0": [0.0, 0.0],
            }[site]
            for site in occupation_string
        ]
    )
    dm0 = np.zeros((2, n_sites, n_sites))
    idx = np.mgrid[:n_sites]
    dm0[0, idx, idx] = occupations[:, 0]
    dm0[1, idx, idx] = occupations[:, 1]
    return dm0


def _occupation_population_targets(
    occupation_string: str | None, orbitals: tuple[int, ...], *, norb: int
) -> tuple[float, float] | None:
    occ = occupation_string or DEFAULT_FE2_OCCUPATION_STRING
    if len(occ) != norb:
        return None
    n_alpha = 0.0
    n_beta = 0.0
    for orb in orbitals:
        site = occ[int(orb)]
        if site == "2":
            n_alpha += 1.0
            n_beta += 1.0
        elif site == "a":
            n_alpha += 1.0
        elif site == "b":
            n_beta += 1.0
        elif site != "0":
            raise ValueError(f"Unsupported occupation character {site!r} at orbital {orb}.")
    return n_alpha, n_beta


def _print_population_group(
    label: str,
    orbitals: tuple[int, ...],
    *,
    alpha_diag: np.ndarray,
    beta_diag: np.ndarray,
    occupation_string: str | None,
    norb: int,
) -> None:
    n_alpha = float(np.sum(alpha_diag[list(orbitals)]))
    n_beta = float(np.sum(beta_diag[list(orbitals)]))
    moment = n_alpha - n_beta
    target = _occupation_population_targets(occupation_string, orbitals, norb=norb)
    if target is None:
        target_text = ""
    else:
        target_alpha, target_beta = target
        target_text = (
            f"  target=({target_alpha:6.3f},{target_beta:6.3f})"
            f" target_m={target_alpha - target_beta:7.3f}"
        )
    print(
        f"  {label:16s} orbitals={list(orbitals)!s:24s} "
        f"n_alpha={n_alpha:9.5f} n_beta={n_beta:9.5f} m={moment:9.5f}"
        f"{target_text}"
    )


def print_uhf_population_diagnostics(umf, args: argparse.Namespace) -> None:
    dm = np.asarray(umf.make_rdm1())
    if dm.shape[0] != 2:
        raise ValueError(f"Expected unrestricted density matrix with spin axis, got {dm.shape}.")
    norb = int(umf.mol.nao)
    alpha_diag = np.real(np.diag(dm[0]))
    beta_diag = np.real(np.diag(dm[1]))
    print("\n[UHF population diagnostics]")
    print(
        f"total n_alpha={float(np.trace(dm[0]).real):.8f} "
        f"n_beta={float(np.trace(dm[1]).real):.8f} "
        f"m={float((np.trace(dm[0]) - np.trace(dm[1])).real):.8f}"
    )

    fe_centers = _normalize_real_field_centers(args.real_field_centers, norb=norb)
    for idx, center in enumerate(fe_centers):
        _print_population_group(
            f"Fe center {idx}",
            center,
            alpha_diag=alpha_diag,
            beta_diag=beta_diag,
            occupation_string=args.occupation_string,
            norb=norb,
        )

    if args.ligand_centers is not None:
        ligand_centers = _normalize_real_field_centers(args.ligand_centers, norb=norb)
        ligand_orbitals = _flatten_centers(ligand_centers)
        _print_population_group(
            "ligands",
            ligand_orbitals,
            alpha_diag=alpha_diag,
            beta_diag=beta_diag,
            occupation_string=args.occupation_string,
            norb=norb,
        )


def load_uhf_from_chkfile(umf, chkfile: Path) -> bool:
    chkfile = chkfile.expanduser()
    if not chkfile.exists():
        return False

    from pyscf import scf

    scf_rec = scf.chkfile.load(str(chkfile), "scf")
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot"):
        if key in scf_rec:
            setattr(umf, key, scf_rec[key])

    if not hasattr(umf, "mo_coeff"):
        raise ValueError(f"Checkpoint {chkfile} did not contain scf/mo_coeff.")

    norb = int(umf.mol.nao)
    mo_coeff = np.asarray(umf.mo_coeff)
    if mo_coeff.shape[-2:] != (norb, norb):
        raise ValueError(
            f"Checkpoint {chkfile} mo_coeff has trailing shape {mo_coeff.shape[-2:]}, "
            f"but FCIDUMP has {norb} orbitals."
        )

    umf.chkfile = str(chkfile)
    umf.converged = True
    print(f"Loaded UHF solution from {chkfile} and marked it converged.")
    return True


def build_uhf_mf(args: argparse.Namespace):
    start = time.perf_counter()
    mf = prepare_fcidump_mf(args.fcidump)
    umf = mf.to_uhf().newton()
    umf.chkfile = str(args.chkfile)
    umf.mol.verbose = 4
    loaded = False
    if getattr(args, "load_chkfile", False):
        loaded = load_uhf_from_chkfile(umf, args.chkfile)
    if not loaded:
        umf.kernel(dm0=fe2_broken_symmetry_guess(umf, args.occupation_string))
    if args.print_uhf_populations:
        print_uhf_population_diagnostics(umf, args)
    seconds = time.perf_counter() - start
    print(f"Prepared Fe2 UHF inputs in {seconds:.2f}s")
    return umf


def build_cc_driver(args: argparse.Namespace):
    try:
        from ccpy.drivers.driver import Driver
    except ImportError as exc:
        raise RuntimeError("This example requires ccpy. Please install ccpy to run it.") from exc

    start = time.perf_counter()
    umf = build_uhf_mf(args)
    cc_driver = Driver.from_pyscf(umf, nfrozen=0, uhf=True)
    cc_driver.options["amp_convergence"] = args.amp_convergence
    cc_driver.options["energy_convergence"] = args.energy_convergence
    cc_driver.options["RHF_symmetry"] = False
    cc_driver.run_cc(method=args.method)

    seconds = time.perf_counter() - start
    print(f"Prepared Fe2 UHF/{args.method.upper()} inputs in {seconds:.2f}s")
    return cc_driver, umf


def _load_cache(path: Path):
    start = time.perf_counter()
    staged = load_staged(path)
    seconds = time.perf_counter() - start
    print(f"Loaded staged inputs from {path} in {seconds:.2f}s")
    return staged, seconds


def _assert_psd_fcidump_for_standard(args: argparse.Namespace) -> None:
    spectrum = fcidump_pair_spectrum(args.fcidump)
    min_eig = float(spectrum["min_eigenvalue"])
    if min_eig < -10.0 * float(args.chol_cut):
        raise ValueError(
            "Ordinary Cholesky/vanilla AFQMC requires a positive-semidefinite "
            "two-body packed-pair matrix, but this FCIDUMP/model is indefinite "
            f"(min eigenvalue {min_eig:.6e}). Use --model-h2 onsite or "
            "--model-h2 selected_full for vanilla AFQMC, or run a real-field "
            "route instead of --vanilla-afqmc."
        )


def _load_cache_if_compatible(
    path: Path, args: argparse.Namespace, *, require_psd_fcidump: bool = False
) -> tuple[Any, float] | None:
    if require_psd_fcidump:
        _assert_psd_fcidump_for_standard(args)
    staged, seconds = _load_cache(path)
    expected_norb = _fcidump_norb(args.fcidump)
    if int(staged.ham.norb) != expected_norb:
        print(
            f"Ignoring staged cache {path}: n_orbitals={staged.ham.norb} "
            f"but current FCIDUMP has NORB={expected_norb}."
        )
        return None
    return staged, seconds


def _print_local_parameters(meta: dict[str, Any]) -> None:
    center_reports = meta.get("center_reports", [])
    if not center_reports:
        return

    print("local parameters:")
    for report in center_reports:
        center = int(report["center"])
        orbitals = report.get("orbitals", [])
        params = report.get("parameters", {})
        kanamori_terms = report.get("kanamori_terms", [])
        onsite_dec = {
            int(term["orbital"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") == "onsite_U"
        }
        onsite_pref = {
            int(term["orbital"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") == "onsite_U"
        }
        density_dec = {
            tuple(term["orbitals"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") == "interorbital_Uprime"
        }
        density_pref = {
            tuple(term["orbitals"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") == "interorbital_Uprime"
        }
        hund_dec = {
            tuple(term["orbitals"]): term["decomposition"]
            for term in kanamori_terms
            if term.get("kind") in {"hund_J_bond", "hund_J_pair", "pair_full"}
        }
        hund_pref = {
            tuple(term["orbitals"]): term.get("preferred_decomposition")
            for term in kanamori_terms
            if term.get("kind") in {"hund_J_bond", "hund_J_pair", "pair_full"}
        }
        print(f"  center={center} orbitals={orbitals}")
        for term in params.get("onsite_U", []):
            orb = int(term["orbital"])
            dec = onsite_dec.get(orb)
            suffix = f" decomposition={dec}" if dec is not None else ""
            if onsite_pref.get(orb) is not None and onsite_pref[orb] != dec:
                suffix += f" preferred={onsite_pref[orb]}"
            print(f"    U      orb={orb} value={float(term['U']): .10f}{suffix}")

        uprime = {
            tuple(term["orbitals"]): float(term["Uprime"])
            for term in params.get("interorbital_Uprime", [])
        }
        exchange = {tuple(term["orbitals"]): float(term["J"]) for term in params.get("exchange_J", [])}
        pair_hopping = {
            tuple(term["orbitals"]): float(term["P"])
            for term in params.get("pair_hopping_like", [])
        }
        for pair in sorted(set(uprime) | set(exchange) | set(pair_hopping)):
            dec_parts = []
            if pair in density_dec:
                dec_parts.append(f"Uprime_dec={density_dec[pair]}")
                if density_pref.get(pair) is not None and density_pref[pair] != density_dec[pair]:
                    dec_parts.append(f"Uprime_pref={density_pref[pair]}")
            if pair in hund_dec:
                dec_parts.append(f"J_dec={hund_dec[pair]}")
                if hund_pref.get(pair) is not None and hund_pref[pair] != hund_dec[pair]:
                    dec_parts.append(f"J_pref={hund_pref[pair]}")
            suffix = f" {' '.join(dec_parts)}" if dec_parts else ""
            print(
                f"    pair   orbs={list(pair)} "
                f"Uprime={uprime.get(pair, 0.0): .10f} "
                f"J={exchange.get(pair, 0.0): .10f} "
                f"P={pair_hopping.get(pair, 0.0): .10f}"
                f"{suffix}"
            )


def _print_field_metadata(label: str, staged: Any) -> None:
    meta = staged.meta.get("field_metadata")
    print(f"\n[{label}]")
    print(f"n_orbitals = {staged.ham.norb}")
    print(f"n_fields   = {staged.ham.chol.shape[0]}")
    if (
        hasattr(staged, "meta")
        and staged.meta.get("fe_band_model") is not None
        and not staged.meta.get("no_model_extraction")
    ):
        print(f"Fe preset  = {staged.meta.get('fe_band_model')}-band")
    if hasattr(staged, "meta") and staged.meta.get("trial_method") is not None:
        method_line = f"trial method = {staged.meta.get('trial_method')}"
        if staged.meta.get("trial_order") is not None:
            method_line += f" order {staged.meta.get('trial_order')}"
        method_line += f" ({staged.trial.kind})"
        print(method_line)
    if not meta:
        print("field route = ordinary Cholesky")
        return

    print(f"field route             = {meta.get('real_field_fit')}")
    if meta.get("decomposition_scope") == "full":
        print("decomposition scope     = full tensor (no model extraction)")
    else:
        print(f"centers                 = {meta.get('centers')}")
    if staged.meta.get("ligand_centers") is not None:
        print(f"ligand onsite centers   = {staged.meta.get('ligand_centers')}")
    if meta.get("real_field_fit") == "hk_density":
        print(f"HK real fields          = {meta.get('n_hk_real_fields')}")
    elif meta.get("decomposition_scope") == "full":
        print(f"full real fields        = {meta.get('n_full_real_fields')}")
        print(f"full complex fields     = {meta.get('n_full_complex_fields')}")
        print(f"charge-dominant fields  = {meta.get('n_full_charge_dominant_fields')}")
        print(f"spin-dominant fields    = {meta.get('n_full_spin_dominant_fields')}")
        print(f"mixed charge-spin fields= {meta.get('n_full_mixed_charge_spin_fields')}")
    else:
        print(f"local real fields       = {meta.get('n_local_real_fields')}")
        print(f"local complex fields    = {meta.get('n_local_complex_fields')}")
    if meta.get("decomposition_scope") != "full":
        print(f"residual real fields    = {meta.get('n_residual_real_fields')}")
        print(f"residual complex fields = {meta.get('n_residual_complex_fields')}")
    frob = meta.get("frobenius", {})
    if frob:
        print("Frobenius diagnostics:")
        print(f"  ||V_full||                 = {float(frob['full_norm']):.10f}")
        if "hk_onsite_norm" in frob:
            print(f"  ||V_HK onsite||            = {float(frob['hk_onsite_norm']):.10f}")
        if meta.get("decomposition_scope") != "full" and "local_block_norm" in frob:
            print(f"  ||V_local block||          = {float(frob['local_block_norm']):.10f}")
        if meta.get("decomposition_scope") != "full" and "extracted_local_block_norm" in frob:
            print(
                "  ||V_local extracted||      = "
                f"{float(frob['extracted_local_block_norm']):.10f}"
            )
        if meta.get("decomposition_scope") != "full":
            print(f"  ||V_residual||             = {float(frob['residual_norm']):.10f}")
            print(f"  ||V_center block||         = {float(frob['center_block_norm']):.10f}")
        print(f"  ||V_full pair||            = {float(frob['full_pair_norm']):.10f}")
        if meta.get("decomposition_scope") == "full":
            print(
                "  full pair rel. error       = "
                f"{float(frob['full_pair_reconstruction_relative_error']):.3e}"
            )
        else:
            print(
                "  residual pair rel. error   = "
                f"{float(frob['residual_pair_reconstruction_relative_error']):.3e}"
            )
        if "hk_fraction_full_weight" in frob:
            print("  HK/full weight fraction    = " f"{float(frob['hk_fraction_full_weight']):.6f}")
            print(
                "  HK/full pair weight frac.  = "
                f"{float(frob['hk_fraction_full_pair_weight']):.6f}"
            )
            print(
                "  HK/center weight fraction  = "
                f"{float(frob['hk_fraction_center_block_weight']):.6f}"
            )
        if meta.get("decomposition_scope") != "full" and "local_block_fraction_full_weight" in frob:
            print(
                "  local/full weight fraction = "
                f"{float(frob['local_block_fraction_full_weight']):.6f}"
            )
            print(
                "  extracted/full weight frac.= "
                f"{float(frob['extracted_local_block_fraction_full_weight']):.6f}"
            )
    extracted = meta.get("extracted_terms", [])
    if extracted:
        print("onsite U terms:")
        for term in extracted[:20]:
            print(
                f"  center={term['center']} orbital={term['orbital']} " f"U={float(term['U']):.10f}"
            )
        if len(extracted) > 20:
            print(f"  ... {len(extracted) - 20} more")
    _print_local_parameters(meta)


def _bar(fraction: float, *, width: int = 32) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "-" * (width - filled)


def _print_hk_analysis(analysis: dict[str, Any]) -> None:
    print("\n[HK analysis]")
    print(f"n_orbitals        = {analysis['n_orbitals']}")
    print(f"centers           = {analysis['centers']}")
    print(f"ligand centers    = {analysis['ligand_centers']}")
    print(f"||V_full pair||   = {float(analysis['full_pair_norm']):.10f}")

    print("\nmethod capture estimates, packed-pair weight fractions:")
    method_labels = [
        ("hk_density", "onsite U only"),
        ("hk_density_fe_s_onsite", "positive Fe + ligand onsite U"),
        ("kanamori_uj", "onsite U + Hund/pair J"),
        ("charge_spin", "formal charge/spin full local block"),
        ("kanamori_real", "real-biased U + Uprime + Hund/pair"),
        ("kanamori_sign_like", "U + Uprime + Hund/pair"),
        ("local_exact", "full same-center Fe blocks"),
    ]
    for key, label in method_labels:
        frac = float(analysis["method_weight_fractions"][key])
        print(f"  {key:19s} {frac:9.6f}  [{_bar(frac)}]  {label}")

    print("\nreal-field-friendly onsite spin candidates:")
    for key, label in (
        ("real_fe_onsite_U", "positive Fe onsite U"),
        ("real_ligand_onsite_U", "positive ligand onsite U"),
        ("real_fe_ligand_onsite_U", "positive Fe + ligand onsite U"),
    ):
        bucket = analysis["buckets"][key]
        frac = float(bucket["weight_fraction"])
        print(
            f"  {key:27s} {frac:9.6f}  [{_bar(frac)}]  "
            f"norm={float(bucket['norm']):.10f}  {label}"
        )

    print("\nlocal Fe-block composition, relative to selected same-center Fe blocks:")
    for key, label in (
        ("onsite_U", "onsite U"),
        ("interorbital_Uprime", "interorbital Uprime"),
        ("hund_pair", "Hund/pair J channel"),
        ("local_other", "non-HK local residual"),
    ):
        frac = float(analysis["local_weight_fractions"][key])
        print(f"  {key:19s} {frac:9.6f}  [{_bar(frac)}]  {label}")

    print("\nglobal packed-pair decomposition:")
    for key, label in (
        ("local_same_center", "same-center Fe local blocks"),
        ("fe_cross", "cross-center Fe terms"),
        ("fe_ligand", "Fe-ligand/mixed terms"),
        ("non_fe", "non-Fe terms"),
    ):
        bucket = analysis["buckets"][key]
        frac = float(bucket["weight_fraction"])
        print(
            f"  {key:19s} {frac:9.6f}  [{_bar(frac)}]  "
            f"norm={float(bucket['norm']):.10f}  {label}"
        )

    print("\nFe-ligand bridge diagnostics:")
    bridge = analysis["buckets"]["fe_ligand_bridge"]
    bridge_frac = float(bridge["weight_fraction"])
    print(
        f"  {'bridge_total':27s} {bridge_frac:9.6f}  [{_bar(bridge_frac)}]  "
        f"norm={float(bridge['norm']):.10f}"
    )
    for key, label in (
        ("fe_ligand_density", "density V_dp entries"),
        ("fe_ligand_exchange", "exchange/pair-like d-p entries"),
        ("fe_ligand_hk_like", "density + exchange/pair-like"),
        ("fe_ligand_bridge_other", "other bridge tensor residual"),
        ("fe_ligand_bridge_real_extractable", "guaranteed exact real spin fields"),
        ("fe_ligand_bridge_complex_or_residual", "requires complex fields or residual"),
    ):
        bucket = analysis["buckets"][key]
        frac = float(bucket["weight_fraction"])
        rel = float(bucket.get("relative_weight_fraction", 0.0))
        print(
            f"  {key:27s} {frac:9.6f}  [{_bar(frac)}]  "
            f"bridge_rel={rel:9.6f}  {label}"
        )

    one_body = analysis.get("one_body", {})
    if one_body:
        print("\none-body superexchange proxy:")
        print(f"  ||h1||                    = {float(one_body['full_h1_norm']):.10f}")
        print(
            "  ||h1_Fe-ligand||          = "
            f"{float(one_body['fe_ligand_hopping_norm']):.10f} "
            f"(weight frac {float(one_body['fe_ligand_hopping_weight_fraction']):.6f})"
        )

    print("\nper-center local capture:")
    for report in analysis["center_reports"]:
        print(
            f"  center={report['center']} orbitals={report['orbitals']} "
            f"||V_local_pair||={float(report['local_pair_norm']):.10f}"
        )
        for key, label in (
            ("onsite_U_local_weight_fraction", "U"),
            ("u_j_local_weight_fraction", "U+J"),
            ("hk_like_local_weight_fraction", "U+Uprime+J"),
            ("local_other_weight_fraction", "other"),
        ):
            frac = float(report[key])
            print(f"    {label:11s} {frac:9.6f}  [{_bar(frac, width=24)}]")


def analyze_hk(args: argparse.Namespace) -> None:
    analysis = analyze_hk_from_fcidump(
        args.fcidump,
        centers=args.real_field_centers,
        ligand_centers=args.ligand_centers,
    )
    _print_hk_analysis(analysis)


def _model_orbitals_from_args(args: argparse.Namespace, *, norb: int) -> tuple[int, ...]:
    if args.model_orbitals is not None:
        return _flatten_centers(_normalize_real_field_centers(args.model_orbitals, norb=norb))

    fe_centers = _normalize_real_field_centers(args.real_field_centers, norb=norb)
    ligand_centers = (
        _normalize_real_field_centers(args.ligand_centers, norb=norb)
        if args.ligand_centers is not None
        else ()
    )
    return _flatten_centers(fe_centers + ligand_centers)


def write_model_fcidump(
    args: argparse.Namespace, *, occupation_string: str | None = None
) -> dict[str, Any]:
    from pyscf.tools import fcidump as pyscf_fcidump

    ctx = pyscf_fcidump.read(str(args.fcidump.expanduser()))
    norb = int(ctx["NORB"])
    model_orbitals = _model_orbitals_from_args(args, norb=norb)
    if args.ligand_centers is None and args.model_orbitals is None:
        raise ValueError(
            "--write-model-fcidump needs --ligand-centers or --model-orbitals; "
            "otherwise the model would contain only Fe orbitals."
        )

    nelec = args.model_nelec
    ms2 = args.model_ms2
    occ = occupation_string
    if occ is None:
        occ = args.model_occupation_string or args.occupation_string or DEFAULT_FE2_OCCUPATION_STRING
    model_occupation_string = None
    if nelec is None or ms2 is None:
        if len(occ) != norb:
            raise ValueError(
                f"Model occupation string has length {len(occ)}, but FCIDUMP has {norb} orbitals."
            )
        inferred_nelec, inferred_ms2 = _nelec_ms2_from_occupation_string(occ, model_orbitals)
        model_occupation_string = _project_occupation_string(occ, model_orbitals)
        nelec = inferred_nelec if nelec is None else nelec
        ms2 = inferred_ms2 if ms2 is None else ms2

    reference_occupations = None
    if args.model_h1_correction == "reference_fock":
        if len(occ) != norb:
            raise ValueError(
                "--model-h1-correction reference_fock requires a full-space occupation "
                f"string of length {norb}; got length {len(occ)}."
            )
        reference_occupations = _occupation_arrays_from_string(occ)

    result = build_model_fcidump_from_fcidump(
        args.fcidump,
        args.write_model_fcidump,
        model_orbitals=model_orbitals,
        fe_centers=args.real_field_centers,
        ligand_centers=args.ligand_centers,
        h2_model=args.model_h2,
        h1_correction=args.model_h1_correction,
        reference_occupations=reference_occupations,
        nelec=int(nelec),
        ms2=int(ms2),
    )
    result["model_occupation_string"] = model_occupation_string

    print("\n[model FCIDUMP]")
    print(f"wrote                 = {result['out']}")
    print(f"h2 model              = {result['h2_model']}")
    print(f"source norb           = {result['source_norb']}")
    print(f"model norb            = {result['norb']}")
    print(f"model nelec, ms2      = {result['nelec']}, {result['ms2']}")
    if model_occupation_string is not None:
        print(f"model occupation      = {model_occupation_string}")
    print(f"h1 correction         = {result['h1_correction']}")
    print(f"||delta h1||          = {float(result['h1_correction_norm']):.10f}")
    print(f"model orbitals        = {result['model_orbitals']}")
    print(f"Fe model orbitals     = {result['fe_orbitals']}")
    print(f"ligand model orbitals = {result['ligand_orbitals']}")
    print(f"||V_selected|| pair   = {float(result['selected_pair_norm']):.10f}")
    print(f"||V_model|| pair      = {float(result['model_pair_norm']):.10f}")
    print(f"||V_discarded|| pair  = {float(result['discarded_pair_norm']):.10f}")
    min_eig = float(result["model_pair_min_eigenvalue"])
    nneg = int(result["model_pair_n_negative_eigenvalues"])
    print(f"min eig(V_model pair) = {min_eig:.10e}")
    print(f"negative pair eigs    = {nneg}")
    print(f"vanilla compatible    = {nneg == 0}")
    if nneg:
        print(
            "warning               = ordinary Cholesky/vanilla AFQMC is invalid for this "
            "indefinite model; use --model-h2 onsite/selected_full or a real-field route."
        )
    print(
        "model/selected weight = "
        f"{float(result['model_selected_pair_weight_fraction']):.6f}"
    )
    return result


def _spin_sector_from_nelec_ms2(nelec: int, ms2: int) -> tuple[int, int]:
    if (nelec + ms2) % 2 != 0 or (nelec - ms2) % 2 != 0:
        raise ValueError(f"Invalid NELEC/MS2 combination for FCI: NELEC={nelec}, MS2={ms2}.")
    n_alpha = (nelec + ms2) // 2
    n_beta = (nelec - ms2) // 2
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(f"Invalid negative spin sector: n_alpha={n_alpha}, n_beta={n_beta}.")
    return int(n_alpha), int(n_beta)


def _rotate_to_uhf_natural_orbitals(
    h1: np.ndarray,
    eri: np.ndarray,
    *,
    args: argparse.Namespace,
    fcidump_path: Path,
    occupation_string: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fci_args = argparse.Namespace(**vars(args))
    fci_args.fcidump = fcidump_path
    if occupation_string is not None:
        fci_args.occupation_string = occupation_string
    umf = build_uhf_mf(fci_args)
    dm = np.asarray(umf.make_rdm1())
    dm_total = np.real(dm[0] + dm[1])
    dm_total = 0.5 * (dm_total + dm_total.T)
    occupations, rotation = np.linalg.eigh(dm_total)
    order = np.argsort(occupations)[::-1]
    occupations = occupations[order]
    rotation = rotation[:, order]
    h1_rot = rotation.T @ h1 @ rotation
    h1_rot = 0.5 * (h1_rot + h1_rot.T.conj())
    eri_rot = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        rotation,
        rotation,
        rotation,
        rotation,
        eri,
        optimize=True,
    )
    return h1_rot, np.asarray(eri_rot), occupations


def _build_restricted_mf_for_fci(args: argparse.Namespace, fcidump_path: Path):
    start = time.perf_counter()
    mf = prepare_fcidump_mf(fcidump_path)
    mf = mf.newton()
    mf.mol.verbose = 4
    mf.kernel()
    seconds = time.perf_counter() - start
    print(f"Prepared restricted MO basis for FCI in {seconds:.2f}s")
    return mf


def _parse_int_schedule(text: str | None, *, default: int, n_sweeps: int) -> list[int]:
    if n_sweeps <= 0:
        raise ValueError(f"Number of sweeps must be positive, got {n_sweeps}.")
    if text is None:
        values = [int(default)]
    else:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
        if not values:
            raise ValueError("Empty block2 integer schedule.")
    if any(x <= 0 for x in values):
        raise ValueError(f"Block2 bond dimensions must be positive: {values}.")
    return (values + [values[-1]] * n_sweeps)[:n_sweeps]


def _parse_float_schedule(
    text: str | None,
    *,
    default: float,
    n_sweeps: int,
    noise: bool = False,
) -> list[float]:
    if n_sweeps <= 0:
        raise ValueError(f"Number of sweeps must be positive, got {n_sweeps}.")
    if text is None:
        if noise:
            noisy = max(0, min(4, n_sweeps - 2))
            quiet = min(2, n_sweeps)
            values = (
                [1.0e-4] * noisy
                + [1.0e-5] * max(0, n_sweeps - noisy - quiet)
                + [0.0] * quiet
            )
        else:
            values = [float(default)]
    else:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
        if not values:
            raise ValueError("Empty block2 floating-point schedule.")
    if any(x < 0.0 for x in values):
        raise ValueError(f"Block2 schedule values must be non-negative: {values}.")
    return (values + [values[-1]] * n_sweeps)[:n_sweeps]


def _format_bytes(n_bytes: int | float) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _estimate_block2_storage(
    driver: Any,
    mpo: Any,
    *,
    bond_dim: int,
    dot: int = 2,
) -> tuple[int, int]:
    bw = driver.bw
    mps_info = bw.brs.MPSInfo(
        driver.n_sites,
        driver.vacuum,
        driver.target,
        driver.ghamil.basis,
    )
    try:
        mps_info.set_bond_dimension(int(bond_dim))
        _, memory_bytes, scratch_bytes = mpo.estimate_storage(mps_info, dot)
        return int(memory_bytes), int(scratch_bytes)
    finally:
        mps_info.deallocate_mutable()
        mps_info.deallocate()


def run_block2_dmrg(
    args: argparse.Namespace,
    *,
    fcidump_path: Path | None = None,
) -> None:
    from pyblock2.driver.core import DMRGDriver
    from pyblock2.driver.core import SymmetryTypes

    path = (fcidump_path or args.fcidump).expanduser()
    n_sweeps = int(args.block2_n_sweeps)
    bond_dims = _parse_int_schedule(
        args.block2_bond_dims,
        default=int(args.block2_bond_dim),
        n_sweeps=n_sweeps,
    )
    noises = _parse_float_schedule(
        args.block2_noises,
        default=0.0,
        n_sweeps=n_sweeps,
        noise=True,
    )
    dav_thrds = _parse_float_schedule(
        None,
        default=float(args.block2_davidson_tol),
        n_sweeps=n_sweeps,
    )
    memory_mb = float(
        args.block2_memory if args.block2_memory is not None else args.fci_max_memory
    )
    reorder = None if args.block2_reorder == "none" else args.block2_reorder
    scratch = (
        args.block2_scratch.expanduser()
        if args.block2_scratch is not None
        else Path(tempfile.gettempdir()) / f"trot_block2_{path.stem}"
    )

    print(f"Parsing {path}")
    sys.stdout.flush()
    fcidump = None
    driver = DMRGDriver(
        symm_type=SymmetryTypes.SZ,
        stack_mem=int(memory_mb * 1024 * 1024),
        scratch=str(scratch),
        clean_scratch=True,
        n_threads=args.block2_threads,
    )
    try:
        start = time.perf_counter()
        fcidump = driver.read_fcidump(str(path), pg=args.block2_point_group, iprint=1)
        driver.initialize_system(
            driver.n_sites,
            n_elec=driver.n_elec,
            spin=driver.spin,
            pg_irrep=driver.pg_irrep,
            orb_sym=driver.orb_sym,
        )
        n_alpha = (driver.n_elec + driver.spin) // 2
        n_beta = (driver.n_elec - driver.spin) // 2
        dim = math.comb(driver.n_sites, n_alpha) * math.comb(driver.n_sites, n_beta)
        if int(args.fci_nroots) != 1:
            print(
                "[block2] ignoring --fci-nroots; "
                "block2 DMRG is configured for the lowest root only."
            )

        print("\n[block2 DMRG]")
        print(f"fcidump        = {path}")
        print(f"n_orbitals     = {driver.n_sites}")
        print(f"nelec, ms2     = {driver.n_elec}, {driver.spin}")
        print(f"sector dim     = {dim}")
        print(f"ecore          = {float(driver.ecore):.12f}")
        print(f"bond_dims      = {bond_dims}")
        print(f"noises         = {noises}")
        print(f"davidson_thrds = {dav_thrds}")
        print(f"n_sweeps       = {n_sweeps}")
        print(f"scratch        = {scratch}")
        print(f"memory         = {memory_mb:.1f} MB")
        print(f"threads        = {args.block2_threads}")
        print(f"point_group    = {args.block2_point_group}")
        print(f"reorder        = {args.block2_reorder}")
        sys.stdout.flush()

        mpo = driver.get_qc_mpo(
            driver.h1e,
            driver.g2e,
            ecore=driver.ecore,
            reorder=reorder,
            cutoff=float(args.block2_mpo_cutoff),
            integral_cutoff=float(args.block2_integral_cutoff),
            iprint=1,
        )
        peak_memory, scratch_estimate = _estimate_block2_storage(
            driver,
            mpo,
            bond_dim=max(bond_dims),
        )
        memory_limit = int(memory_mb * 1024 * 1024)
        print(f"estimated memory = {_format_bytes(peak_memory)}")
        print(f"estimated scratch = {_format_bytes(scratch_estimate)}")
        sys.stdout.flush()
        if peak_memory > memory_limit:
            raise SystemExit(
                "ERROR: block2 storage estimate exceeds --block2-memory: "
                f"need about {_format_bytes(peak_memory)} for M={max(bond_dims)}, "
                f"but --block2-memory provides {_format_bytes(memory_limit)}. "
                "Increase --block2-memory or reduce --block2-bond-dim."
            )
        ket = driver.get_random_mps(
            "KET",
            bond_dim=bond_dims[0],
            nroots=1,
            full_fci=False,
        )
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=n_sweeps,
            tol=float(args.block2_tol),
            bond_dims=bond_dims,
            noises=noises,
            thrds=dav_thrds,
            cutoff=float(args.block2_cutoff),
            iprint=1,
        )
        bond_dim_hist, discarded_weights, energies = driver.get_dmrg_results()
        seconds = time.perf_counter() - start
        print("\n[block2 DMRG sweeps]")
        for sweep, (bdim, dw, es) in enumerate(
            zip(bond_dim_hist, discarded_weights, energies),
            start=1,
        ):
            root_energy = float(np.asarray(es).reshape(-1)[0])
            print(
                f"sweep {sweep:3d}: M = {int(bdim):6d}  "
                f"E = {root_energy: .12f} Ha  discarded_weight = {float(dw):.6e}"
            )
        print(f"\nblock2 lowest-root energy: {float(energy): .12f} Ha")
        print(f"block2 timing            : {seconds:.2f}s")
    finally:
        if fcidump is not None:
            fcidump.deallocate()
        driver.finalize()


def run_fci(
    args: argparse.Namespace,
    *,
    fcidump_path: Path | None = None,
    occupation_string: str | None = None,
) -> None:
    from pyscf import ao2mo
    from pyscf import fci
    from pyscf.tools import fcidump as pyscf_fcidump

    path = (fcidump_path or args.fcidump).expanduser()
    print(f"Parsing {path}")
    ctx = pyscf_fcidump.read(str(path))
    norb = int(ctx["NORB"])
    nelec = int(ctx["NELEC"])
    ms2 = int(ctx.get("MS2", 0))
    n_alpha, n_beta = _spin_sector_from_nelec_ms2(nelec, ms2)
    nelec_tuple = (n_alpha, n_beta)
    dim = math.comb(norb, n_alpha) * math.comb(norb, n_beta)

    if args.fci_basis == "restricted_mo":
        from pyscf import fci

        print("Building restricted MO basis for FCI...")
        mf = _build_restricted_mf_for_fci(args, path)
        solver = fci.FCI(mf, mf.mo_coeff, singlet=False)
        solver.max_memory = float(args.fci_max_memory)
        solver.max_cycle = int(args.fci_max_cycle)
        solver.max_space = int(args.fci_max_space)

        print("\n[FCI]")
        print(f"fcidump        = {path}")
        print(f"n_orbitals     = {norb}")
        print(f"nelec, ms2     = {nelec}, {ms2}")
        print(f"n_alpha,beta   = {n_alpha}, {n_beta}")
        print(f"sector dim     = {dim}")
        print(f"ecore          = {float(mf.energy_nuc()):.12f}")
        print(f"orbital basis  = {args.fci_basis}")
        print(f"nroots         = {args.fci_nroots}")
        sys.stdout.flush()

        start = time.perf_counter()
        energies, ci = solver.kernel(
            nelec=nelec_tuple,
            tol=args.fci_conv_tol,
            max_cycle=args.fci_max_cycle,
            max_space=args.fci_max_space,
            nroots=args.fci_nroots,
            pspace_size=args.fci_pspace_size,
        )
        seconds = time.perf_counter() - start
        energy_list = np.asarray(energies, dtype=float).reshape(-1)
        ci_list = ci if isinstance(ci, (list, tuple)) else [ci]
        for root, (energy, ci_root) in enumerate(zip(energy_list, ci_list)):
            ss, multiplicity = solver.spin_square(ci_root, norb, nelec_tuple)
            print(
                f"root {root:2d}: E = {float(energy): .12f} Ha  "
                f"<S^2> = {float(ss):.10f}  2S+1 = {float(multiplicity):.10f}"
            )
        print(f"FCI timing     = {seconds:.2f}s")
        return

    h1 = np.asarray(ctx["H1"])
    h1 = 0.5 * (h1 + h1.T.conj())
    eri = ao2mo.restore(1, np.asarray(ctx["H2"]), norb)
    ecore = float(ctx.get("ECORE", 0.0))
    natural_occupations = None
    if args.fci_basis == "uhf_natural":
        print("Building UHF natural-orbital basis for FCI...")
        h1, eri, natural_occupations = _rotate_to_uhf_natural_orbitals(
            h1,
            eri,
            args=args,
            fcidump_path=path,
            occupation_string=occupation_string,
        )
    elif args.fci_basis != "input":
        raise ValueError(f"Unsupported FCI basis: {args.fci_basis!r}.")

    solver = fci.direct_spin1.FCI()
    solver.max_memory = float(args.fci_max_memory)
    solver.max_cycle = int(args.fci_max_cycle)
    solver.max_space = int(args.fci_max_space)

    print("\n[FCI]")
    print(f"fcidump        = {path}")
    print(f"n_orbitals     = {norb}")
    print(f"nelec, ms2     = {nelec}, {ms2}")
    print(f"n_alpha,beta   = {n_alpha}, {n_beta}")
    print(f"sector dim     = {dim}")
    print(f"ecore          = {ecore:.12f}")
    print(f"orbital basis  = {args.fci_basis}")
    if natural_occupations is not None:
        occ_text = " ".join(f"{float(x):.6f}" for x in natural_occupations[: min(12, norb)])
        print(f"NO occupations = {occ_text}")
    print(f"nroots         = {args.fci_nroots}")
    sys.stdout.flush()
    start = time.perf_counter()
    energies, ci = solver.kernel(
        h1,
        eri,
        norb,
        nelec_tuple,
        tol=args.fci_conv_tol,
        max_cycle=args.fci_max_cycle,
        max_space=args.fci_max_space,
        nroots=args.fci_nroots,
        pspace_size=args.fci_pspace_size,
        ecore=ecore,
    )
    seconds = time.perf_counter() - start

    energy_list = np.asarray(energies, dtype=float).reshape(-1)
    ci_list = ci if isinstance(ci, (list, tuple)) else [ci]
    for root, (energy, ci_root) in enumerate(zip(energy_list, ci_list)):
        ss, multiplicity = solver.spin_square(ci_root, norb, nelec_tuple)
        print(
            f"root {root:2d}: E = {float(energy): .12f} Ha  "
            f"<S^2> = {float(ss):.10f}  2S+1 = {float(multiplicity):.10f}"
        )
    print(f"FCI timing     = {seconds:.2f}s")


def staging_real_field_centers(args: argparse.Namespace, *, norb: int) -> tuple[tuple[int, ...], ...]:
    fe_centers = _normalize_real_field_centers(args.real_field_centers, norb=norb)
    if args.ligand_centers is None:
        return fe_centers

    ligand_centers = _normalize_real_field_centers(args.ligand_centers, norb=norb)
    fe_orbitals = {orb for center in fe_centers for orb in center}
    ligand_orbitals = {orb for center in ligand_centers for orb in center}
    overlap = sorted(fe_orbitals & ligand_orbitals)
    if overlap:
        raise ValueError(f"--ligand-centers overlaps Fe centers at orbitals {overlap}.")

    # Keep ligand orbitals as singleton centers so staging extracts only onsite U
    # for them, not a full ligand-local block.
    ligand_singletons = tuple((orb,) for orb in sorted(ligand_orbitals))
    return fe_centers + ligand_singletons


def _annotate_staged(staged: Any, args: argparse.Namespace) -> Any:
    staged.meta["fe_band_model"] = args.fe_band_model
    staged.meta["ligand_centers"] = args.ligand_centers
    staged.meta["no_model_extraction"] = bool(args.no_model_extraction)
    staged.meta["trial_method"] = args.method
    if args.method != "uhf":
        staged.meta["trial_order"] = int(args.order)
    return staged


def _uhf_identity_trial_for_current_basis(args: argparse.Namespace) -> TrialInput:
    norb = _fcidump_norb(args.fcidump)
    return TrialInput(
        kind="uhf",
        data={"mo_a": np.eye(norb), "mo_b": np.eye(norb)},
        frozen=0,
        source_kind="mf",
    )


def stage_ccpy_one(
    *,
    label: str,
    cache: Path,
    args: argparse.Namespace,
    cc_driver: Any,
    umf: Any,
    real_field_centers: str | None,
    real_field_method: str,
):
    start = time.perf_counter()
    staging_centers = (
        None
        if real_field_centers is None
        else staging_real_field_centers(args, norb=int(umf.mol.nao))
    )
    staged = stage_from_ccpy(
        cc_driver,
        umf,
        order=args.order,
        chol_cut=args.chol_cut,
        fcidump=args.fcidump,
        cache=cache,
        overwrite=True,
        verbose=args.verbose_stage,
        real_field_centers=staging_centers,
        real_field_method=real_field_method,
    )
    staged = _annotate_staged(staged, args)
    seconds = time.perf_counter() - start
    print(f"Wrote {label} staged inputs to {cache} in {seconds:.2f}s")
    return staged, seconds


def stage_uhf_one(
    *,
    label: str,
    cache: Path,
    args: argparse.Namespace,
    umf: Any,
    real_field_centers: str | None,
    real_field_method: str,
):
    start = time.perf_counter()
    staged_obj = StagedMfOrCc(umf, 0)
    staging_centers = (
        None
        if real_field_centers is None
        else staging_real_field_centers(args, norb=int(umf.mol.nao))
    )
    ham = _stage_ham_input_from_fcidump(
        staged_obj,
        fcidump=args.fcidump,
        chol_cut=args.chol_cut,
        verbose=args.verbose_stage,
        real_field_centers=staging_centers,
        real_field_method=real_field_method,
    )
    staged = stage(
        umf,
        chol_cut=args.chol_cut,
        cache=cache,
        overwrite=True,
        verbose=args.verbose_stage,
        ham=ham,
        trial=(
            _uhf_identity_trial_for_current_basis(args)
            if real_field_method
            in {
                "uhf_charge_spin",
                "uhf_charge_spin_blocks",
                "uhf_charge_spin_unrham",
                "uhf_local_real_then_charge_spin_unrham",
            }
            else None
        ),
    )
    staged = _annotate_staged(staged, args)
    seconds = time.perf_counter() - start
    print(f"Wrote {label} staged inputs to {cache} in {seconds:.2f}s")
    return staged, seconds


def build_or_load_staged(args: argparse.Namespace) -> dict[str, tuple[Any, float]]:
    args.cache = args.cache.expanduser().resolve()
    args.standard_cache = args.standard_cache.expanduser().resolve()
    real_staged: tuple[Any, float] | None = None
    standard_staged: tuple[Any, float] | None = None

    run_real = not args.vanilla_afqmc
    run_standard = bool(args.compare_standard or args.vanilla_afqmc)
    need_real = run_real and (args.overwrite_cache or not args.cache.exists())
    need_standard = run_standard and (args.overwrite_cache or not args.standard_cache.exists())

    if run_real and not need_real:
        loaded = _load_cache_if_compatible(args.cache, args)
        if loaded is None:
            need_real = True
        else:
            staged_inputs, seconds = loaded
            real_staged = (_annotate_staged(staged_inputs, args), seconds)
    if run_standard and not need_standard:
        loaded = _load_cache_if_compatible(args.standard_cache, args, require_psd_fcidump=True)
        if loaded is None:
            need_standard = True
        else:
            staged_inputs, seconds = loaded
            standard_staged = (_annotate_staged(staged_inputs, args), seconds)

    if need_standard:
        _assert_psd_fcidump_for_standard(args)

    if need_real or need_standard:
        if args.method == "uhf":
            umf = build_uhf_mf(args)
            if need_real:
                real_staged = stage_uhf_one(
                    label=f"real-field {args.real_field_method} UHF",
                    cache=args.cache,
                    args=args,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                    real_field_method=args.real_field_method,
                )
            if need_standard:
                standard_staged = stage_uhf_one(
                    label="standard Cholesky UHF",
                    cache=args.standard_cache,
                    args=args,
                    umf=umf,
                    real_field_centers=None,
                    real_field_method="hk_density",
                )
        else:
            cc_driver, umf = build_cc_driver(args)
            if need_real:
                real_staged = stage_ccpy_one(
                    label=f"real-field {args.real_field_method} {args.method} order {args.order}",
                    cache=args.cache,
                    args=args,
                    cc_driver=cc_driver,
                    umf=umf,
                    real_field_centers=args.real_field_centers,
                    real_field_method=args.real_field_method,
                )
            if need_standard:
                standard_staged = stage_ccpy_one(
                    label=f"standard Cholesky {args.method} order {args.order}",
                    cache=args.standard_cache,
                    args=args,
                    cc_driver=cc_driver,
                    umf=umf,
                    real_field_centers=None,
                    real_field_method="hk_density",
                )

    staged: dict[str, tuple[Any, float]] = {}
    if real_staged is not None:
        staged["real-field"] = real_staged
    if standard_staged is not None:
        staged["vanilla" if args.vanilla_afqmc else "standard"] = standard_staged
    return staged


def run_afqmc(label: str, staged: Any, args: argparse.Namespace) -> tuple[float, float, float]:
    af = Afqmc(staged)
    af.mixed_precision = args.mixed_precision
    af.params = QmcParams(
        dt=args.dt,
        n_chunks=args.n_chunks,
        n_exp_terms=args.n_exp_terms,
        n_prop_steps=args.n_prop_steps,
        n_blocks=args.n_blocks,
        n_walkers=args.n_walkers,
        seed=args.seed,
        n_eql_blocks=args.n_eql_blocks,
    )
    af.walker_kind = "generalized" if staged.ham.basis == "generalized" else "unrestricted"

    start = time.perf_counter()
    samples_path = None
    if args.samples_raw is not None:
        samples_path = args.samples_raw
        if args.compare_standard and not args.vanilla_afqmc:
            clean_label = label.replace(" ", "_").replace("/", "_")
            samples_path = samples_path.with_name(
                f"{samples_path.stem}_{clean_label}{samples_path.suffix or '.dat'}"
            )
    mean, err = af.kernel(samples_path=samples_path)
    seconds = time.perf_counter() - start
    print(f"{label} AFQMC/{staged.trial.kind} energy: {mean:.10f} +/- {err:.10f} Ha")
    print(f"{label} AFQMC timing: {seconds:.2f}s")
    return float(mean), float(err), seconds


def main() -> None:
    args = parse_args()
    if args.analyze_hk:
        analyze_hk(args)
        return
    if args.write_model_fcidump is not None:
        result = write_model_fcidump(args)
        if args.fci:
            if args.fci_solver == "block2":
                run_block2_dmrg(args, fcidump_path=Path(result["out"]))
            else:
                run_fci(
                    args,
                    fcidump_path=Path(result["out"]),
                    occupation_string=result.get("model_occupation_string"),
                )
        return
    if args.fci:
        if args.fci_solver == "block2":
            run_block2_dmrg(args)
        else:
            run_fci(args)
        return

    staged = build_or_load_staged(args)

    for label, (staged_inputs, stage_seconds) in staged.items():
        _print_field_metadata(label, staged_inputs)
        print(f"{label} staging/load timing: {stage_seconds:.2f}s")

    if args.stage_only:
        return

    for label, (staged_inputs, _) in staged.items():
        run_afqmc(label, staged_inputs, args)


if __name__ == "__main__":
    main()
