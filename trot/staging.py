from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple, TypeAlias, Union, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

from .ham.chol import HamBasis

# This file contains staging utilities to convert pyscf mf/cc objects
# into serializable data classes representing Hamiltonian and trial
# wavefunction inputs which can be used for building AFQMC objects.

Array: TypeAlias = NDArray[Any]

# to keep track of format versions when loading/saving staged inputs
STAGE_FORMAT_VERSION = 1


def _stage_begin(message: str) -> float:
    print(f"[stage] {message}...")
    return time.time()


def _stage_end(start: float, message: str, *, details: str | None = None) -> None:
    suffix = f" | {details}" if details else ""
    print(f"[stage] {message} in {time.time() - start:.2f}s{suffix}")


def _normalize_frozen_list(frozen: Any, *, nmo: int) -> NDArray:
    frozen_arr = np.asarray(frozen, dtype=np.int64)
    if frozen_arr.ndim != 1:
        raise ValueError("cc.frozen must be a one-dimensional list of MO indices.")
    if frozen_arr.size == 0:
        return frozen_arr

    frozen_arr = np.sort(frozen_arr)
    if np.unique(frozen_arr).size != frozen_arr.size:
        raise ValueError("cc.frozen contains duplicate MO indices.")
    if frozen_arr[0] < 0 or frozen_arr[-1] >= nmo:
        raise ValueError(f"cc.frozen indices must lie in [0, {nmo}).")

    return frozen_arr


def _mo_coeff_signature(mo_coeff: Any) -> tuple[tuple[int, ...], ...]:
    if isinstance(mo_coeff, (tuple, list)):
        return tuple(tuple(int(dim) for dim in np.asarray(block).shape) for block in mo_coeff)
    return (tuple(int(dim) for dim in np.asarray(mo_coeff).shape),)


def _mo_coeff_nmo(mo_coeff: Any) -> int:
    if isinstance(mo_coeff, (tuple, list)):
        shapes = [np.asarray(block).shape for block in mo_coeff]
        if len(shapes) == 0:
            raise ValueError("Spin-separated MO coefficients cannot be empty.")
        if any(len(shape) == 0 for shape in shapes):
            raise ValueError("MO coefficient blocks must have at least one dimension.")
        nmo = int(shapes[0][-1])
        if any(int(shape[-1]) != nmo for shape in shapes):
            raise ValueError(
                "Spin-separated MO coefficient blocks must have the same number of orbitals."
            )
        return nmo

    shape = np.asarray(mo_coeff).shape
    if len(shape) == 0:
        raise ValueError("MO coefficients must have at least one dimension.")
    return int(shape[-1])


def _copy_scf_with_cc_mo_coeff(cc: Any, mf: Any) -> Any:
    if not hasattr(cc, "mo_coeff") or cc.mo_coeff is None:
        return mf

    cc_sig = _mo_coeff_signature(cc.mo_coeff)
    mf_sig = _mo_coeff_signature(mf.mo_coeff)
    if cc_sig != mf_sig:
        raise ValueError(
            "CC object mo_coeff shape does not match the underlying SCF mo_coeff shape: "
            f"{cc_sig} != {mf_sig}."
        )

    mf_copy = copy.copy(mf)
    mf_copy.mo_coeff = cc.mo_coeff
    return mf_copy


def _infer_restricted_trial_freeze_from_cc(
    *,
    cc_frozen: Any,
    nmo_full: int,
    nocc_full: int,
    norb_frozen: int,
    t1_shape: tuple[int, int],
) -> tuple[int, int]:
    frozen = _normalize_frozen_list(cc_frozen, nmo=nmo_full)
    occ_frozen = frozen[frozen < nocc_full]
    vir_frozen = frozen[frozen >= nocc_full]

    nocc_cc_frozen = int(occ_frozen.size)
    nvir_cc_frozen = int(vir_frozen.size)

    if occ_frozen.size and not np.array_equal(
        occ_frozen, np.arange(nocc_cc_frozen, dtype=np.int64)
    ):
        raise ValueError(
            "Occupied orbitals in list-valued cc.frozen must form a contiguous prefix."
        )

    vir_expected = np.arange(nmo_full - nvir_cc_frozen, nmo_full, dtype=np.int64)
    if vir_frozen.size and not np.array_equal(vir_frozen, vir_expected):
        raise ValueError("Virtual orbitals in list-valued cc.frozen must form a contiguous suffix.")

    if norb_frozen > nocc_cc_frozen:
        raise ValueError("norb_frozen cannot exceed the number of occupied orbitals frozen in CC.")

    nocc_act_expected = nocc_full - nocc_cc_frozen
    nvir_act_expected = (nmo_full - nocc_full) - nvir_cc_frozen
    if t1_shape != (nocc_act_expected, nvir_act_expected):
        raise ValueError(
            "cc.frozen is inconsistent with the CC amplitudes in the restricted CISD trial."
        )

    nocc_t_core = nocc_cc_frozen - norb_frozen
    nvir_t_outer = nvir_cc_frozen
    return nocc_t_core, nvir_t_outer


def modified_cholesky(
    mat: Array,
    max_error: float = 1e-6,
) -> Array:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (Array): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors.
    """
    diag = mat.diagonal()
    norb = int(((-1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2))
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    chol0 = chol_vecs[:nchol]
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        for m in range(norb):
            for n in range(m + 1):
                triind = m * (m + 1) // 2 + n
                chol[i, m, n] = chol0[i, triind]
                chol[i, n, m] = chol0[i, triind]
    return chol


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10) -> NDArray:
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems. (copied from pauxy)

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs.")
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


def _rotate_chol_to_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate AO-space Cholesky into an MO basis."""
    C = np.asarray(basis_coeff)
    nao, norb = C.shape
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)

    reuse_storage = nao == norb and out_dtype == chol_vec.dtype
    if reuse_storage:
        chol = chol_vec.reshape(nchol, nao, nao)
    else:
        chol = np.empty((nchol, norb, norb), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    tmp = np.empty((nao, norb), dtype=out_dtype)
    for i in range(nchol):
        chol_i_ao = chol_vec[i].reshape(nao, nao)
        np.dot(chol_i_ao, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _rotate_chol_to_ghf_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate spatial AO Cholesky factors into a generalized-spin MO basis."""
    C = np.asarray(basis_coeff)
    nao2, nmo = C.shape
    if nao2 % 2 != 0:
        raise ValueError(f"Expected even GHF AO dimension, got {nao2}")

    nao = nao2 // 2
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)
    chol = np.empty((nchol, nmo, nmo), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    chol_i_full = np.zeros((nao2, nao2), dtype=out_dtype)
    tmp = np.empty((nao2, nmo), dtype=out_dtype)
    for i in range(nchol):
        chol_i = chol_vec[i].reshape(nao, nao)
        chol_i_full.fill(0)
        chol_i_full[:nao, :nao] = chol_i
        chol_i_full[nao:, nao:] = chol_i
        np.dot(chol_i_full, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _stage_frozen(frozen: int | ArrayLike | None) -> int | NDArray | None:
    if isinstance(frozen, (list, tuple, np.ndarray)):
        frozen = np.asarray(frozen, dtype=int)
    elif frozen is None:
        frozen = None
    elif isinstance(frozen, int):
        frozen = int(frozen)
    else:
        raise TypeError(f"Unsupported type '{type(frozen)}'.")

    return frozen


def _resolve_stage_frozen_arg(
    norb_frozen_core: int | None,
    norb_frozen: int | None,
    frozen_orbitals: ArrayLike | None,
) -> int | ArrayLike | None:
    if norb_frozen_core is not None and norb_frozen is not None and norb_frozen_core != norb_frozen:
        raise ValueError("norb_frozen_core and norb_frozen must match when both are passed.")
    core_frozen = norb_frozen_core if norb_frozen_core is not None else norb_frozen
    if core_frozen is not None and frozen_orbitals is not None:
        raise ValueError("Pass only one of norb_frozen_core/norb_frozen or frozen_orbitals.")
    if frozen_orbitals is not None:
        return frozen_orbitals
    return core_frozen


def _freeze_meta_value(frozen: int | NDArray | None) -> int | list[int] | None:
    if isinstance(frozen, np.ndarray):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return [int(x) for x in arr]
    if isinstance(frozen, int):
        return frozen
    if isinstance(frozen, np.integer):
        return int(frozen.item())
    if frozen is None:
        return None
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _freeze_from_meta_value(frozen: Any) -> int | NDArray | None:
    if frozen is None:
        return None
    if isinstance(frozen, list):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, np.ndarray):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _dump_frozen(group: h5py.Group, frozen: int | NDArray, *, attr_name: str = "frozen") -> None:
    if isinstance(frozen, np.ndarray):
        group.create_dataset(attr_name, data=np.asarray(frozen, dtype=np.int64))
    else:
        group.attrs[attr_name] = int(frozen)


def _load_frozen(group: h5py.Group, *, attr_name: str = "frozen") -> int | NDArray:
    if attr_name in group:
        dataset = group[attr_name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Expected dataset '{attr_name}', got {type(dataset)}")
        return np.asarray(dataset[...], dtype=np.int64)

    attr_value = group.attrs[attr_name]
    if isinstance(attr_value, np.ndarray):
        return int(np.asarray(attr_value).item())
    if isinstance(attr_value, (int, np.integer)):
        return int(attr_value)
    raise TypeError(f"Unsupported frozen attribute type: {type(attr_value)}")


@dataclass(frozen=True, slots=True)
class HamInput:
    """ham inputs in the chosen orthonormal one particle basis"""

    h0: float
    h1: Array  # (norb, norb), (2, norb, norb), or (nso, nso)
    chol: Array  # (nchol, norb, norb), (nchol, 2, norb, norb), or (nchol, nso, nso)
    nelec: Tuple[int, int]
    norb: int
    chol_cut: float
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"
    basis: HamBasis  # "restricted", "unrestricted", or "generalized"
    field_factors: Array | None = None
    field_spin_coeffs: Array | None = None
    field_metadata: Dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrialInput:
    """trial inputs used to construct an afqmc trial object"""

    kind: str  # "slater", "cisd", "ucisd"
    data: Dict[str, Array]
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"


@dataclass(frozen=True, slots=True)
class StagedInputs:
    ham: HamInput
    trial: TrialInput
    meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RealFieldFitResult:
    h1_shift: Array
    chol: Array
    field_factors: Array
    field_spin_coeffs: Array
    metadata: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class StagedCc:
    """Wrapper ensuring the validity of the CC object"""

    _delegate = {"t1", "t2", "_scf", "frozen"}
    kind: str  # "ccsd", "uccsd", "gccsd"
    cc: Any
    mf: Any
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD

        if not isinstance(cc, (CCSD, UCCSD, GCCSD)):
            raise TypeError(f"Unsupported object type: {type(cc)}")

        if not hasattr(cc, "_scf"):
            raise TypeError("CC-like object missing _scf reference to underlying scf object.")
        else:
            mf = _copy_scf_with_cc_mo_coeff(cc, cc._scf)

        if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
            raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

        if isinstance(cc, CCSD):
            kind = "ccsd"
        elif isinstance(cc, UCCSD):
            kind = "uccsd"
        elif isinstance(cc, GCCSD):
            kind = "gccsd"

        frozen = _stage_frozen(frozen)
        cc_frozen = _stage_frozen(cc.frozen)

        if cc_frozen is None:
            if frozen is not None and not (isinstance(frozen, int) and frozen == 0):
                raise ValueError(
                    "Explicit AFQMC frozen orbitals are unsupported for CC objects without cc.frozen."
                )
            afqmc_frozen = 0
            trial_frozen = 0
        elif isinstance(cc_frozen, np.ndarray):
            if kind != "ccsd":
                raise NotImplementedError(
                    "List-valued cc.frozen is currently supported only for restricted CCSD staging."
                )
            trial_frozen = _normalize_frozen_list(cc_frozen, nmo=_mo_coeff_nmo(mf.mo_coeff))
            if frozen is None:
                afqmc_frozen = 0
            elif isinstance(frozen, int):
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "List-valued cc.frozen requires an integer AFQMC frozen-core count."
                )
        else:
            if frozen is None:
                afqmc_frozen = int(cc_frozen)
            elif isinstance(frozen, int):
                if int(cc_frozen) != frozen:
                    raise ValueError("cc.frozen and staging frozen must be equal.")
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "Integer cc.frozen is incompatible with list-valued staging frozen."
                )
            trial_frozen = int(cc_frozen)

        if isinstance(trial_frozen, np.ndarray) and kind != "ccsd":
            raise NotImplementedError(
                "List-valued cc.frozen is currently supported only for restricted CCSD staging."
            )

        mf = StagedMf(mf, afqmc_frozen)

        object.__setattr__(self, "cc", cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", afqmc_frozen)
        object.__setattr__(self, "trial_frozen", trial_frozen)

    def __getattr__(self, name):
        if name in StagedCc._delegate:
            return getattr(self.cc, name)
        elif hasattr(self.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMf:
    """Wrapper ensuring the validity of the SCF object"""

    _delegate = {"mo_coeff", "mo_occ", "mol", "nelec", "get_ovlp", "energy_nuc", "get_hcore"}
    kind: str  # "rhf", "rohf", "uhf", ghf
    mf: Any  # Python SCF object
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, mf: Any, frozen: int | ArrayLike | None):
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if not isinstance(mf, (RHF, ROHF, UHF, GHF)):
            raise TypeError(f"Unsupported object type: {type(mf)}")

        if not hasattr(mf, "mo_coeff"):
            raise ValueError("MO coefficients not found; did you run mf.kernel()?")

        if isinstance(mf, ROHF):
            kind = "rohf"
        elif isinstance(mf, RHF):
            kind = "rhf"
        elif isinstance(mf, UHF):
            kind = "uhf"
        elif isinstance(mf, GHF):
            kind = "ghf"

        frozen = _stage_frozen(frozen)
        nmo = _mo_coeff_nmo(mf.mo_coeff)

        if isinstance(frozen, np.ndarray):
            frozen_arr = _normalize_frozen_list(frozen, nmo=nmo)
            if frozen_arr.size > 0:
                assert frozen_arr.size < nmo
            frozen = frozen_arr
        elif isinstance(frozen, int):
            assert frozen < nmo
            assert frozen >= 0
        elif frozen is None:
            frozen = 0
        else:
            raise TypeError(
                f"Expected a type int | np.ndarray | None, but received '{type(frozen)}'."
            )

        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", frozen)
        object.__setattr__(self, "trial_frozen", 0)

    @property
    def norb(self) -> int:
        return _mo_coeff_nmo(self.mf.mo_coeff)

    def __getattr__(self, name: str):
        if name in StagedMf._delegate:
            return getattr(self.mf, name)
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF object or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMfOrCc:
    """Wrapper ensuring the validity of the SCF/CC object"""

    _delegate_mf = StagedMf._delegate
    _delegate_cc = StagedCc._delegate
    kind: str  # StageCc.kind or StagedMf.kind
    source: str  # "cc", "mf"
    mf_or_cc: Any  # StagedMf or StagedCc
    mf: StagedMf
    afqmc_frozen: int | NDArray
    trial_frozen: int | NDArray

    def __init__(self, mf_or_cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if isinstance(mf_or_cc, (CCSD, UCCSD, GCCSD)):
            mf_or_cc = StagedCc(mf_or_cc, frozen)
            mf = mf_or_cc.mf
            source = "cc"
        elif isinstance(mf_or_cc, (RHF, ROHF, UHF, GHF)):
            mf_or_cc = StagedMf(mf_or_cc, frozen)
            mf = mf_or_cc
            source = "mf"
        else:
            raise TypeError(f"Unreachable: '{type(mf_or_cc)}'")

        object.__setattr__(self, "mf_or_cc", mf_or_cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", mf_or_cc.kind)
        object.__setattr__(self, "afqmc_frozen", mf_or_cc.afqmc_frozen)
        object.__setattr__(self, "trial_frozen", mf_or_cc.trial_frozen)

    @property
    def norb(self) -> int:
        return _mo_coeff_nmo(self.mf.mo_coeff)

    def __getattr__(self, name: str):
        if name in StagedMfOrCc._delegate_cc:
            return getattr(self.mf_or_cc, name)
        elif name in StagedMfOrCc._delegate_mf:
            return getattr(self.mf, name)
        elif self.source == "cc" and hasattr(self.mf_or_cc.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


# public API
def stage(
    obj: Any,
    *,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    frozen_orbitals: ArrayLike | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    ham: HamInput | None = None,
    trial: TrialInput | None = None,
) -> StagedInputs:
    """
    Stage inputs from a pyscf mf or cc object.

    Args:
        obj:
            pyscf mf object (RHF/ROHF/UHF) or cc object (CCSD/UCCSD).
        norb_frozen_core:
            Preferred name for the number of lowest occupied core orbitals removed from the
            AFQMC Hamiltonian.
        norb_frozen:
            Backward-compatible alias for ``norb_frozen_core``.
            For CC objects with integer ``cc.frozen``, this is inferred from ``cc.frozen``.
            For restricted CCSD objects with list-valued ``cc.frozen``, trial-space frozen
            occupied/virtual blocks are inferred from ``cc.frozen`` while
            ``norb_frozen_core``/``norb_frozen`` control the occupied core orbitals removed
            from the AFQMC Hamiltonian.
        frozen_orbitals:
            Explicit orbital list for LNO-style staging. Generic AFQMC/FNO staging should use
            ``norb_frozen_core`` instead.
        chol_cut:
            Cholesky decomposition cutoff.
        cache:
            Optional path to write on disk. If it exists and overwrite=False,
            loads it. Otherwise computes and writes it.
        overwrite:
            If True and cache is provided, recompute and overwrite cache.
        verbose:
            Print timing/info.
        ham:
            Optionally provide HamInput. If None, will be staged from obj.
        trial:
            Optionally provide TrialInput. If None, will be staged from obj.

    Returns:
        StagedInputs containing HamInput, TrialInput, and metadata.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load(cache_path)

    t0 = time.time()

    resolved_frozen = _resolve_stage_frozen_arg(norb_frozen_core, norb_frozen, frozen_orbitals)
    obj = StagedMfOrCc(obj, resolved_frozen)
    mol = obj.mol

    if ham is None:
        t_ham = _stage_begin("building Hamiltonian")
        ham = _stage_ham_input(
            obj,
            chol_cut=chol_cut,
            verbose=verbose,
        )
        _stage_end(
            t_ham,
            "Hamiltonian ready",
            details=f"norb={ham.norb} nchol={ham.chol.shape[0]}",
        )

    if trial is None:
        t_trial = _stage_begin("building trial input")
        trial = _stage_trial_input(obj)
        _stage_end(t_trial, "trial input ready", details=f"kind={trial.kind}")

    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": obj.source,
        "frozen": _freeze_meta_value(obj.afqmc_frozen),
        "chol_cut": ham.chol_cut if ham is not None else chol_cut,
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }
    if ham.field_metadata is not None:
        meta["field_metadata"] = ham.field_metadata

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)

    if cache_path is not None:
        dump(staged, cache_path)

    if verbose:
        dt = time.time() - t0
        print(f"[stage] done in {dt:.2f}s | norb={ham.norb} nchol={ham.chol.shape[0]}")

    return staged


def dump(staged: StagedInputs, path: Union[str, Path]) -> None:
    """
    Save staged inputs to a single h5 file.

    Args:
        staged: StagedInputs to serialize
        path: output file path
    """
    p = Path(path).expanduser().resolve()
    t_dump = _stage_begin(f"writing staged inputs to {p}")
    _dump_h5(staged, p)
    _stage_end(t_dump, "staged inputs written")


def load(path: Union[str, Path]) -> StagedInputs:
    """
    Load staged inputs from a single file written by dump().

    Args:
        path: input file path

    Returns:
        StagedInputs
    """
    p = Path(path).expanduser().resolve()
    t_load = _stage_begin(f"loading staged inputs from {p}")
    staged = _load_h5(p)
    _stage_end(
        t_load,
        "staged inputs loaded",
        details=f"norb={staged.ham.norb} nchol={staged.ham.chol.shape[0]} trial={staged.trial.kind}",
    )
    return staged


def _is_cc_like(obj: Any) -> bool:
    return hasattr(obj, "t1") and hasattr(obj, "t2")


def _stage_ham_input(obj: StagedMfOrCc, *, chol_cut: float, verbose: bool) -> HamInput:
    """
    Produce h0/h1/chol in a single orthonormal basis.
    For UHF, we use the alpha MO basis for integrals.
    """
    from pyscf import mcscf

    mol = obj.mol
    scf_obj = obj.mf

    match scf_obj.kind:
        case "rhf" | "rohf" | "ghf":
            basis_coeff = np.asarray(scf_obj.mo_coeff)
        case "uhf":
            basis_coeff = np.asarray(scf_obj.mo_coeff[0])
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    match scf_obj.kind:
        case "rhf" | "rohf" | "uhf":
            ham_basis = "restricted"
        case "ghf":
            ham_basis = "generalized"
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    # nuclear energy (without frozen core correction)
    h0 = float(scf_obj.energy_nuc())

    # one body
    hcore = scf_obj.get_hcore()
    h1 = basis_coeff.T.conj() @ hcore @ basis_coeff
    h1 = np.asarray(h1)

    # ao cholesky
    t0 = time.time()
    chol_vec = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print(f"[stage] AO cholesky: nchol={chol_vec.shape[0]} in {time.time() - t0:.2f}s")

    # full space electron count
    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))
    norb_frozen = scf_obj.afqmc_frozen

    assert isinstance(norb_frozen, int)

    # mo Cholesky
    C = np.asarray(basis_coeff)
    if scf_obj.kind != "ghf":
        norb = int(basis_coeff.shape[1])
        chol = _rotate_chol_to_mo(chol_vec, C)
    else:
        norb = basis_coeff.shape[1] // 2
        chol = _rotate_chol_to_ghf_mo(chol_vec, C)

    # freeze core
    if norb_frozen > 0 and scf_obj.kind != "ghf":

        if isinstance(norb_frozen, int):
            if norb_frozen > min(nelec):
                raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")

            nelec_frozen = 2 * norb_frozen
            ncas = basis_coeff.shape[1] - norb_frozen
            nelecas = mol.nelectron - nelec_frozen

        if nelecas <= 0 or ncas <= 0:
            raise ValueError("Frozen core left no active electrons/orbitals.")

        mc = mcscf.CASSCF(scf_obj.mf, ncas, nelecas)
        mc.mo_coeff = basis_coeff  # type: ignore
        h1_eff, ecore = mc.get_h1eff()  # type: ignore
        i0 = int(mc.ncore)  # type: ignore
        i1 = i0 + int(mc.ncas)  # type: ignore

        h0 = float(ecore)
        h1 = np.asarray(h1_eff)
        chol = np.array(chol[:, i0:i1, i0:i1], copy=True)
        norb = int(ncas)
        nelec = tuple(int(x) for x in mc.nelecas)  # type: ignore
    elif norb_frozen > 0 and scf_obj.kind == "ghf":
        raise NotImplementedError(
            "Frozen core approximation not available for generalised integrals."
        )

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=obj.source,
        basis=ham_basis,
    )


def _load_fcidump_context(fcidump: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(fcidump, dict):
        ctx: Dict[str, Any] = dict(fcidump)
    elif isinstance(fcidump, (str, Path)):
        from pyscf.tools import fcidump as pyscf_fcidump

        ctx = pyscf_fcidump.read(str(Path(fcidump).expanduser()))
    else:
        raise TypeError(f"fcidump must be dict | str | Path, but received '{type(fcidump)}'.")

    required_keys = ("H1", "H2", "NORB", "NELEC")
    missing = [k for k in required_keys if k not in ctx]
    if missing:
        raise ValueError(f"fcidump is missing required keys: {missing}")
    return ctx


def fcidump_pair_spectrum(
    fcidump: Union[str, Path, Dict[str, Any]],
) -> Dict[str, float | int]:
    from pyscf import ao2mo

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    eri = ao2mo.restore(1, np.asarray(ctx["H2"]), norb)
    pair = np.asarray(ao2mo.restore(4, eri, norb))
    pair = 0.5 * (pair + pair.T.conj())
    eigvals = np.linalg.eigvalsh(pair)
    return {
        "min_eigenvalue": float(eigvals[0]) if eigvals.size else 0.0,
        "max_eigenvalue": float(eigvals[-1]) if eigvals.size else 0.0,
        "n_negative_eigenvalues": int(np.sum(eigvals < -1.0e-10)),
        "negative_eigenvalue_norm": float(np.linalg.norm(eigvals[eigvals < 0.0])),
        "pair_norm": float(np.linalg.norm(pair)),
    }


def _normalize_real_field_centers(
    centers: Any,
    *,
    norb: int,
) -> tuple[tuple[int, ...], ...]:
    if centers is None:
        return ()
    if isinstance(centers, str):
        groups: list[tuple[int, ...]] = []
        for raw_group in centers.split(","):
            group = raw_group.strip()
            if not group:
                continue
            if ":" in group:
                start_s, stop_s = group.split(":", 1)
                start, stop = int(start_s), int(stop_s)
                groups.append(tuple(range(start, stop)))
            else:
                groups.append((int(group),))
        centers = groups

    normalized: list[tuple[int, ...]] = []
    for center in centers:
        if isinstance(center, slice):
            start = 0 if center.start is None else int(center.start)
            stop = norb if center.stop is None else int(center.stop)
            step = 1 if center.step is None else int(center.step)
            orbitals = tuple(range(start, stop, step))
        elif isinstance(center, str):
            orbitals = _normalize_real_field_centers(center, norb=norb)
            normalized.extend(orbitals)
            continue
        else:
            arr = np.asarray(center, dtype=np.int64)
            if arr.ndim == 0:
                orbitals = (int(arr),)
            elif arr.ndim == 1:
                orbitals = tuple(int(x) for x in arr)
            else:
                raise ValueError("Each real-field center must be a scalar, slice, or 1D list.")
        if not orbitals:
            raise ValueError("Real-field centers may not be empty.")
        if min(orbitals) < 0 or max(orbitals) >= norb:
            raise ValueError(f"Real-field center orbitals must lie in [0, {norb}): {orbitals}")
        if len(set(orbitals)) != len(orbitals):
            raise ValueError(f"Real-field center contains duplicate orbitals: {orbitals}")
        normalized.append(orbitals)
    return tuple(normalized)


def _rotate_one_body_to_mo(mat: Array, coeff: Array) -> Array:
    return np.asarray(coeff).T.conj() @ np.asarray(mat) @ np.asarray(coeff)


def _unpack_pair_vector(vec: Array, norb: int) -> Array:
    mat = np.zeros((norb, norb), dtype=np.asarray(vec).dtype)
    for m in range(norb):
        for n in range(m + 1):
            pair_idx = m * (m + 1) // 2 + n
            mat[m, n] = vec[pair_idx]
            mat[n, m] = vec[pair_idx]
    return mat


def _pack_pair_matrix(mat: Array) -> Array:
    arr = np.asarray(mat)
    norb = int(arr.shape[0])
    return np.asarray([arr[m, n] for m in range(norb) for n in range(m + 1)])


def _packed_pair_index(i: int, j: int) -> int:
    m, n = (i, j) if i >= j else (j, i)
    return m * (m + 1) // 2 + n


def _reconstruct_packed_pair_from_fields(
    chol: Array,
    field_factors: Array,
    field_spin_coeffs: Array | None = None,
    *,
    local_orbitals: tuple[int, ...] | None = None,
) -> Array:
    chol_arr = np.asarray(chol)
    if local_orbitals is not None:
        ix = np.ix_(local_orbitals, local_orbitals)
        chol_arr = chol_arr[:, ix[0], ix[1]]
    if chol_arr.shape[0] == 0:
        norb = int(chol_arr.shape[1]) if chol_arr.ndim == 3 else 0
        npair = norb * (norb + 1) // 2
        return np.zeros((npair, npair), dtype=np.complex128)
    packed = np.asarray([_pack_pair_matrix(chol_i) for chol_i in chol_arr])
    factors = np.asarray(field_factors, dtype=np.complex128)
    coeff = -(factors**2)
    if field_spin_coeffs is not None:
        spin_coeffs = np.asarray(field_spin_coeffs)
        spin_mask = np.isclose(spin_coeffs[:, 0], 1.0) & np.isclose(spin_coeffs[:, 1], -1.0)
        coeff = np.where(spin_mask, factors**2, coeff)
    out = np.einsum("g,gi,gj->ij", coeff, packed, packed, optimize=True)
    if np.linalg.norm(np.imag(out)) < 1.0e-12 * max(1.0, np.linalg.norm(out)):
        out = np.real(out)
    return out


def _relative_error(reference: Array, actual: Array) -> float:
    ref_norm = float(np.linalg.norm(reference))
    err_norm = float(np.linalg.norm(np.asarray(reference) - np.asarray(actual)))
    return err_norm / ref_norm if ref_norm > 0.0 else err_norm


def _mode_is_exact_real_spin_channel(mode: Array, eigenvalue: float, *, tol: float) -> bool:
    """Return True when a positive pair mode has no same-spin two-body part."""
    if eigenvalue <= 0.0:
        return False
    mat = 0.5 * (np.asarray(mode) + np.asarray(mode).T.conj())
    if np.linalg.norm(mat) <= tol:
        return False
    singular_values = np.linalg.svd(mat, compute_uv=False)
    if singular_values.size <= 1:
        return True
    return bool(singular_values[1] <= tol * max(1.0, singular_values[0]))


def _antisymmetrize_spin_orbital_tensor(tensor: Array) -> Array:
    arr = np.asarray(tensor)
    return 0.25 * (
        arr
        - np.swapaxes(arr, 0, 1)
        - np.swapaxes(arr, 2, 3)
        + np.swapaxes(np.swapaxes(arr, 0, 1), 2, 3)
    )


def _spin_orbital_tensor_from_eri(eri: Array) -> Array:
    arr = np.asarray(eri)
    norb = int(arr.shape[0])
    nso = 2 * norb
    out = np.zeros((nso, nso, nso, nso), dtype=arr.dtype)
    for sigma in range(2):
        for tau in range(2):
            i0 = sigma * norb
            j0 = tau * norb
            out[i0 : i0 + norb, j0 : j0 + norb, i0 : i0 + norb, j0 : j0 + norb] = (
                arr.transpose(0, 2, 1, 3)
            )
    return _antisymmetrize_spin_orbital_tensor(out)


def _reconstruct_spin_orbital_tensor_from_fields(
    chol: Array,
    field_factors: Array,
    field_spin_coeffs: Array,
) -> Array:
    chol_arr = np.asarray(chol)
    norb = int(chol_arr.shape[1])
    nso = 2 * norb
    out = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    factors = np.asarray(field_factors, dtype=np.complex128)
    spin_coeffs = np.asarray(field_spin_coeffs, dtype=np.complex128)
    for g, mode in enumerate(chol_arr):
        for sigma in range(2):
            for tau in range(2):
                i0 = sigma * norb
                j0 = tau * norb
                coeff = -(factors[g] ** 2) * spin_coeffs[g, sigma] * spin_coeffs[g, tau]
                out[i0 : i0 + norb, j0 : j0 + norb, i0 : i0 + norb, j0 : j0 + norb] += (
                    coeff * np.asarray(mode)[:, None, :, None] * np.asarray(mode)[None, :, None, :]
                )
    out = _antisymmetrize_spin_orbital_tensor(out)
    if np.linalg.norm(np.imag(out)) < 1.0e-12 * max(1.0, np.linalg.norm(out)):
        out = np.real(out)
    return out


def _factorize_symmetric_supermatrix(
    eri: Array,
    *,
    coeff: Array,
    chol_cut: float,
) -> tuple[Array, Array, Array, list[str], dict[str, float | int]]:
    from pyscf import ao2mo

    norb = int(eri.shape[0])
    supermat = ao2mo.restore(4, np.asarray(eri), norb)
    supermat = 0.5 * (supermat + supermat.T.conj())
    eigvals, eigvecs = np.linalg.eigh(supermat)
    keep = np.abs(eigvals) > float(chol_cut)
    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]
    discarded_vals = eigvals[~keep]
    residual_pair_norm = float(np.linalg.norm(eigvals))
    residual_pair_reconstruction_error_norm = float(np.linalg.norm(discarded_vals))
    diagnostics: dict[str, float | int] = {
        "residual_pair_norm": residual_pair_norm,
        "residual_pair_retained_norm": float(np.linalg.norm(kept_vals)),
        "residual_pair_reconstruction_error_norm": residual_pair_reconstruction_error_norm,
        "residual_pair_reconstruction_relative_error": (
            residual_pair_reconstruction_error_norm / residual_pair_norm
            if residual_pair_norm > 0.0
            else 0.0
        ),
        "n_residual_pair_positive_eigenvalues": int(np.sum(kept_vals > 0.0)),
        "n_residual_pair_negative_eigenvalues": int(np.sum(kept_vals < 0.0)),
        "n_residual_pair_discarded_eigenvalues": int(np.sum(~keep)),
    }

    chol_blocks: list[Array] = []
    factors: list[complex] = []
    spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    for idx, value in enumerate(kept_vals):
        local = _unpack_pair_vector(kept_vecs[:, idx] * np.sqrt(abs(value)), norb)
        chol_blocks.append(_rotate_one_body_to_mo(local, coeff))
        if value > 0.0:
            factors.append(1.0j)
            labels.append("residual_complex")
        else:
            factors.append(1.0)
            labels.append("residual_real")
        spin_coeffs.append((1.0, 1.0))

    if not chol_blocks:
        nmo = int(np.asarray(coeff).shape[1])
        return (
            np.zeros((0, nmo, nmo), dtype=np.asarray(eri).dtype),
            np.zeros((0,), dtype=np.complex128),
            np.zeros((0, 2), dtype=np.float64),
            [],
            diagnostics,
        )

    return (
        np.asarray(chol_blocks),
        np.asarray(factors, dtype=np.complex128),
        np.asarray(spin_coeffs, dtype=np.float64),
        labels,
        diagnostics,
    )


def _local_parameter_diagnostics(block: Array, orbitals: tuple[int, ...]) -> dict[str, Any]:
    arr = np.asarray(block)
    onsite = []
    interorbital = []
    exchange = []
    pair_hopping = []
    nloc = len(orbitals)
    for p in range(nloc):
        onsite.append({"orbital": int(orbitals[p]), "U": float(np.real(arr[p, p, p, p]))})
        for q in range(p + 1, nloc):
            interorbital.append(
                {
                    "orbitals": [int(orbitals[p]), int(orbitals[q])],
                    "Uprime": float(np.real(arr[p, p, q, q])),
                }
            )
            exchange.append(
                {
                    "orbitals": [int(orbitals[p]), int(orbitals[q])],
                    "J": float(np.real(arr[p, q, q, p])),
                }
            )
            pair_hopping.append(
                {
                    "orbitals": [int(orbitals[p]), int(orbitals[q])],
                    "P": float(np.real(arr[p, q, p, q])),
                }
            )
    return {
        "onsite_U": onsite,
        "interorbital_Uprime": interorbital,
        "exchange_J": exchange,
        "pair_hopping_like": pair_hopping,
    }


def analyze_hk_from_fcidump(
    fcidump: Union[str, Path, Dict[str, Any]],
    *,
    centers: Any,
    ligand_centers: Any = None,
) -> Dict[str, Any]:
    """Analyze HK-like tensor weights in the FCIDUMP orbital basis."""
    from pyscf import ao2mo

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    real_centers = _normalize_real_field_centers(centers, norb=norb)
    if not real_centers:
        raise ValueError("HK analysis requires at least one Fe center.")
    fe_orbitals = tuple(sorted({orb for center in real_centers for orb in center}))

    if ligand_centers is None:
        fe_set = set(fe_orbitals)
        ligand_centers_norm = tuple((orb,) for orb in range(norb) if orb not in fe_set)
    else:
        ligand_centers_norm = _normalize_real_field_centers(ligand_centers, norb=norb)
    ligand_orbitals = tuple(sorted({orb for center in ligand_centers_norm for orb in center}))
    overlap = sorted(set(fe_orbitals) & set(ligand_orbitals))
    if overlap:
        raise ValueError(f"Fe centers and ligand centers overlap at orbitals {overlap}.")

    h1 = np.asarray(ctx["H1"])
    h1 = 0.5 * (h1 + h1.T.conj())
    h2_raw = np.asarray(ctx["H2"])
    eri = ao2mo.restore(1, h2_raw, norb)
    pair_mat = np.asarray(ao2mo.restore(4, eri, norb))
    pair_mat = 0.5 * (pair_mat + pair_mat.T.conj())
    npair = int(pair_mat.shape[0])
    pair_orbitals = [(m, n) for m in range(norb) for n in range(m + 1)]

    center_of = np.full(norb, -1, dtype=np.int64)
    for center_idx, orbitals in enumerate(real_centers):
        for orb in orbitals:
            if center_of[orb] != -1:
                raise ValueError(f"Fe centers overlap at orbital {orb}.")
            center_of[orb] = center_idx
    fe_mask_orb = np.zeros(norb, dtype=bool)
    ligand_mask_orb = np.zeros(norb, dtype=bool)
    fe_mask_orb[list(fe_orbitals)] = True
    if ligand_orbitals:
        ligand_mask_orb[list(ligand_orbitals)] = True

    def _empty_mask() -> NDArray[np.bool_]:
        return np.zeros((npair, npair), dtype=bool)

    local_mask = _empty_mask()
    fe_cross_mask = _empty_mask()
    fe_ligand_mask = _empty_mask()
    non_fe_mask = _empty_mask()
    for a, (i, j) in enumerate(pair_orbitals):
        for b, (k, l) in enumerate(pair_orbitals):
            ids = [int(center_of[x]) for x in (i, j, k, l)]
            in_fe = [idx >= 0 for idx in ids]
            if all(in_fe) and len(set(ids)) == 1:
                local_mask[a, b] = True
            elif all(in_fe):
                fe_cross_mask[a, b] = True
            elif any(in_fe):
                fe_ligand_mask[a, b] = True
            else:
                non_fe_mask[a, b] = True

    onsite_mask = _empty_mask()
    real_onsite_mask = _empty_mask()
    uprime_mask = _empty_mask()
    hund_mask = _empty_mask()
    per_center_masks: list[dict[str, NDArray[np.bool_]]] = []
    for orbitals in real_centers:
        center_local = _empty_mask()
        center_onsite = _empty_mask()
        center_uprime = _empty_mask()
        center_hund = _empty_mask()
        orbitals = tuple(int(x) for x in orbitals)
        orbital_set = set(orbitals)
        for a, (i, j) in enumerate(pair_orbitals):
            if i not in orbital_set or j not in orbital_set:
                continue
            for b, (k, l) in enumerate(pair_orbitals):
                if k in orbital_set and l in orbital_set:
                    center_local[a, b] = True
        for p in orbitals:
            pp = _packed_pair_index(p, p)
            center_onsite[pp, pp] = True
            if float(np.real(pair_mat[pp, pp])) > 0.0:
                real_onsite_mask[pp, pp] = True
        for p_idx, p in enumerate(orbitals):
            for q in orbitals[p_idx + 1 :]:
                pp = _packed_pair_index(p, p)
                qq = _packed_pair_index(q, q)
                pq = _packed_pair_index(p, q)
                center_uprime[pp, qq] = True
                center_uprime[qq, pp] = True
                center_hund[pq, pq] = True
        onsite_mask |= center_onsite
        uprime_mask |= center_uprime
        hund_mask |= center_hund
        per_center_masks.append(
            {
                "local": center_local,
                "onsite_U": center_onsite,
                "interorbital_Uprime": center_uprime,
                "hund_pair": center_hund,
            }
        )

    ligand_onsite_mask = _empty_mask()
    real_ligand_onsite_mask = _empty_mask()
    for p in ligand_orbitals:
        pp = _packed_pair_index(p, p)
        ligand_onsite_mask[pp, pp] = True
        if float(np.real(pair_mat[pp, pp])) > 0.0:
            real_ligand_onsite_mask[pp, pp] = True

    fe_ligand_bridge_mask = _empty_mask()
    fe_ligand_density_mask = _empty_mask()
    fe_ligand_exchange_mask = _empty_mask()
    fe_orbital_set = set(fe_orbitals)
    ligand_orbital_set = set(ligand_orbitals)
    model_orbital_set = fe_orbital_set | ligand_orbital_set
    for a, (i, j) in enumerate(pair_orbitals):
        for b, (k, l) in enumerate(pair_orbitals):
            all_orbs = (i, j, k, l)
            if not all(orb in model_orbital_set for orb in all_orbs):
                continue
            has_fe = any(orb in fe_orbital_set for orb in all_orbs)
            has_ligand = any(orb in ligand_orbital_set for orb in all_orbs)
            if has_fe and has_ligand:
                fe_ligand_bridge_mask[a, b] = True

    for p in fe_orbitals:
        pp = _packed_pair_index(p, p)
        for q in ligand_orbitals:
            qq = _packed_pair_index(q, q)
            pq = _packed_pair_index(p, q)
            fe_ligand_density_mask[pp, qq] = True
            fe_ligand_density_mask[qq, pp] = True
            fe_ligand_exchange_mask[pq, pq] = True

    fe_ligand_hk_like_mask = fe_ligand_density_mask | fe_ligand_exchange_mask
    fe_ligand_bridge_other_mask = fe_ligand_bridge_mask & ~fe_ligand_hk_like_mask

    hk_mask = onsite_mask | uprime_mask | hund_mask
    uj_mask = onsite_mask | hund_mask
    fe_s_onsite_real_mask = real_onsite_mask | real_ligand_onsite_mask
    local_other_mask = local_mask & ~hk_mask

    def _norm(mask: NDArray[np.bool_]) -> float:
        return float(np.linalg.norm(pair_mat[mask]))

    full_pair_norm = float(np.linalg.norm(pair_mat))
    full_pair_weight = full_pair_norm * full_pair_norm

    def _bucket(name: str, mask: NDArray[np.bool_]) -> dict[str, Any]:
        norm = _norm(mask)
        weight = norm * norm
        return {
            "name": name,
            "norm": norm,
            "weight_fraction": float(weight / full_pair_weight) if full_pair_weight > 0.0 else 0.0,
            "n_entries": int(np.sum(mask)),
        }

    def _relative_bucket(
        name: str, mask: NDArray[np.bool_], reference_mask: NDArray[np.bool_]
    ) -> dict[str, Any]:
        bucket = _bucket(name, mask)
        ref_norm = _norm(reference_mask)
        ref_weight = ref_norm * ref_norm
        norm = float(bucket["norm"])
        weight = norm * norm
        bucket["relative_weight_fraction"] = (
            float(weight / ref_weight) if ref_weight > 0.0 else 0.0
        )
        return bucket

    local_norm = _norm(local_mask)
    local_weight = local_norm * local_norm

    def _local_fraction(mask: NDArray[np.bool_]) -> float:
        weight = _norm(mask) ** 2
        return float(weight / local_weight) if local_weight > 0.0 else 0.0

    h1_full_norm = float(np.linalg.norm(h1))
    h1_fe_ligand = np.zeros_like(h1)
    if ligand_orbitals:
        h1_fe_ligand[np.ix_(fe_orbitals, ligand_orbitals)] = h1[
            np.ix_(fe_orbitals, ligand_orbitals)
        ]
        h1_fe_ligand[np.ix_(ligand_orbitals, fe_orbitals)] = h1[
            np.ix_(ligand_orbitals, fe_orbitals)
        ]
    h1_fe_ligand_norm = float(np.linalg.norm(h1_fe_ligand))

    center_reports: list[dict[str, Any]] = []
    for center_idx, orbitals in enumerate(real_centers):
        masks = per_center_masks[center_idx]
        center_local_other = masks["local"] & ~(
            masks["onsite_U"] | masks["interorbital_Uprime"] | masks["hund_pair"]
        )
        center_local_norm = _norm(masks["local"])
        center_local_weight = center_local_norm * center_local_norm

        def _center_fraction(mask: NDArray[np.bool_]) -> float:
            weight = _norm(mask) ** 2
            return float(weight / center_local_weight) if center_local_weight > 0.0 else 0.0

        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "local_pair_norm": center_local_norm,
                "onsite_U_norm": _norm(masks["onsite_U"]),
                "interorbital_Uprime_norm": _norm(masks["interorbital_Uprime"]),
                "hund_pair_norm": _norm(masks["hund_pair"]),
                "local_other_norm": _norm(center_local_other),
                "onsite_U_local_weight_fraction": _center_fraction(masks["onsite_U"]),
                "u_j_local_weight_fraction": _center_fraction(
                    masks["onsite_U"] | masks["hund_pair"]
                ),
                "hk_like_local_weight_fraction": _center_fraction(
                    masks["onsite_U"] | masks["interorbital_Uprime"] | masks["hund_pair"]
                ),
                "local_other_weight_fraction": _center_fraction(center_local_other),
                "parameters": _local_parameter_diagnostics(np.asarray(eri[ix4]), orbitals),
            }
        )

    return {
        "n_orbitals": norb,
        "centers": [list(center) for center in real_centers],
        "ligand_centers": [list(center) for center in ligand_centers_norm],
        "full_pair_norm": full_pair_norm,
        "one_body": {
            "full_h1_norm": h1_full_norm,
            "fe_ligand_hopping_norm": h1_fe_ligand_norm,
            "fe_ligand_hopping_norm_fraction": (
                float(h1_fe_ligand_norm / h1_full_norm) if h1_full_norm > 0.0 else 0.0
            ),
            "fe_ligand_hopping_weight_fraction": (
                float((h1_fe_ligand_norm * h1_fe_ligand_norm) / (h1_full_norm * h1_full_norm))
                if h1_full_norm > 0.0
                else 0.0
            ),
        },
        "buckets": {
            "onsite_U": _bucket("onsite_U", onsite_mask),
            "real_fe_onsite_U": _bucket("real_fe_onsite_U", real_onsite_mask),
            "ligand_onsite_U": _bucket("ligand_onsite_U", ligand_onsite_mask),
            "real_ligand_onsite_U": _bucket(
                "real_ligand_onsite_U", real_ligand_onsite_mask
            ),
            "real_fe_ligand_onsite_U": _bucket(
                "real_fe_ligand_onsite_U", fe_s_onsite_real_mask
            ),
            "interorbital_Uprime": _bucket("interorbital_Uprime", uprime_mask),
            "hund_pair": _bucket("hund_pair", hund_mask),
            "local_other": _bucket("local_other", local_other_mask),
            "local_same_center": _bucket("local_same_center", local_mask),
            "fe_cross": _bucket("fe_cross", fe_cross_mask),
            "fe_ligand": _bucket("fe_ligand", fe_ligand_mask),
            "fe_ligand_bridge": _bucket("fe_ligand_bridge", fe_ligand_bridge_mask),
            "fe_ligand_density": _relative_bucket(
                "fe_ligand_density", fe_ligand_density_mask, fe_ligand_bridge_mask
            ),
            "fe_ligand_exchange": _relative_bucket(
                "fe_ligand_exchange", fe_ligand_exchange_mask, fe_ligand_bridge_mask
            ),
            "fe_ligand_hk_like": _relative_bucket(
                "fe_ligand_hk_like", fe_ligand_hk_like_mask, fe_ligand_bridge_mask
            ),
            "fe_ligand_bridge_other": _relative_bucket(
                "fe_ligand_bridge_other", fe_ligand_bridge_other_mask, fe_ligand_bridge_mask
            ),
            "fe_ligand_bridge_real_extractable": _relative_bucket(
                "fe_ligand_bridge_real_extractable",
                _empty_mask(),
                fe_ligand_bridge_mask,
            ),
            "fe_ligand_bridge_complex_or_residual": _relative_bucket(
                "fe_ligand_bridge_complex_or_residual",
                fe_ligand_bridge_mask,
                fe_ligand_bridge_mask,
            ),
            "non_fe": _bucket("non_fe", non_fe_mask),
        },
        "method_weight_fractions": {
            "hk_density": _bucket("hk_density", onsite_mask)["weight_fraction"],
            "hk_density_fe_s_onsite": _bucket(
                "hk_density_fe_s_onsite", fe_s_onsite_real_mask
            )["weight_fraction"],
            "kanamori_uj": _bucket("kanamori_uj", uj_mask)["weight_fraction"],
            "charge_spin": _bucket("charge_spin", local_mask)["weight_fraction"],
            "kanamori_real": _bucket("kanamori_real", hk_mask)["weight_fraction"],
            "kanamori_sign_like": _bucket("kanamori_sign_like", hk_mask)["weight_fraction"],
            "local_exact": _bucket("local_exact", local_mask)["weight_fraction"],
        },
        "local_weight_fractions": {
            "onsite_U": _local_fraction(onsite_mask),
            "interorbital_Uprime": _local_fraction(uprime_mask),
            "hund_pair": _local_fraction(hund_mask),
            "u_j": _local_fraction(uj_mask),
            "hk_like": _local_fraction(hk_mask),
            "local_other": _local_fraction(local_other_mask),
        },
        "center_reports": center_reports,
    }


def _flatten_center_orbitals(centers: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sorted({int(orb) for center in centers for orb in center}))


def build_model_fcidump_from_fcidump(
    fcidump: Union[str, Path, Dict[str, Any]],
    out: Union[str, Path],
    *,
    model_orbitals: tuple[int, ...],
    fe_centers: Any,
    ligand_centers: Any = None,
    h2_model: str = "onsite_bridge_density",
    h1_correction: str = "none",
    reference_occupations: tuple[Array, Array] | None = None,
    nelec: int | tuple[int, int] | None = None,
    ms2: int | None = None,
    tol: float = 1.0e-15,
) -> Dict[str, Any]:
    """Write a reduced/model FCIDUMP in the source FCIDUMP orbital basis.

    The one-body Hamiltonian is projected onto ``model_orbitals``. The two-body
    tensor is filtered according to ``h2_model``:

    ``"onsite"``
        Keep only ``(p p | p p)`` onsite terms on selected model orbitals.
    ``"onsite_bridge_density"``
        Keep onsite terms plus Fe-ligand density ``(d d | p p)`` terms.
    ``"selected_full"``
        Keep the full selected-orbital two-body tensor.

    If ``h1_correction="reference_fock"``, the spin-averaged UHF Fock
    contribution from discarded two-body terms is folded into the model h1 at
    ``reference_occupations``. This preserves the reference Fock balance while
    still changing the explicit two-body model.
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    orbitals = tuple(int(orb) for orb in model_orbitals)
    if not orbitals:
        raise ValueError("model_orbitals may not be empty.")
    if len(set(orbitals)) != len(orbitals):
        raise ValueError(f"model_orbitals contains duplicates: {orbitals}")
    if min(orbitals) < 0 or max(orbitals) >= norb:
        raise ValueError(f"model_orbitals must lie in [0, {norb}): {orbitals}")

    fe_centers_norm = _normalize_real_field_centers(fe_centers, norb=norb)
    ligand_centers_norm = (
        _normalize_real_field_centers(ligand_centers, norb=norb)
        if ligand_centers is not None
        else ()
    )
    fe_orbitals = set(_flatten_center_orbitals(fe_centers_norm)) & set(orbitals)
    ligand_orbitals = set(_flatten_center_orbitals(ligand_centers_norm)) & set(orbitals)
    overlap = sorted(fe_orbitals & ligand_orbitals)
    if overlap:
        raise ValueError(f"Fe and ligand model orbitals overlap at {overlap}.")

    h1_src = np.asarray(ctx["H1"])
    h1_src = 0.5 * (h1_src + h1_src.T.conj())
    h1_model = np.asarray(h1_src[np.ix_(orbitals, orbitals)])

    eri_src = ao2mo.restore(1, np.asarray(ctx["H2"]), norb)
    eri_selected = np.asarray(eri_src[np.ix_(orbitals, orbitals, orbitals, orbitals)])
    nmodel = len(orbitals)
    eri_model = np.zeros((nmodel, nmodel, nmodel, nmodel), dtype=eri_selected.dtype)
    global_to_local = {orb: idx for idx, orb in enumerate(orbitals)}

    if h2_model == "selected_full":
        eri_model = np.array(eri_selected, copy=True)
    else:
        if h2_model not in {"onsite", "onsite_bridge_density"}:
            raise ValueError(
                "h2_model must be one of {'onsite', 'onsite_bridge_density', 'selected_full'}, "
                f"got {h2_model!r}."
            )
        for orb, idx in global_to_local.items():
            eri_model[idx, idx, idx, idx] = eri_src[orb, orb, orb, orb]

        if h2_model == "onsite_bridge_density":
            for fe_orb in sorted(fe_orbitals):
                i = global_to_local[fe_orb]
                for ligand_orb in sorted(ligand_orbitals):
                    j = global_to_local[ligand_orb]
                    eri_model[i, i, j, j] = eri_src[fe_orb, fe_orb, ligand_orb, ligand_orb]
                    eri_model[j, j, i, i] = eri_src[ligand_orb, ligand_orb, fe_orb, fe_orb]

    h1_correction_norm = 0.0
    if h1_correction == "reference_fock":
        if reference_occupations is None:
            raise ValueError(
                "h1_correction='reference_fock' requires reference_occupations=(alpha,beta)."
            )
        occ_alpha = np.asarray(reference_occupations[0], dtype=float)
        occ_beta = np.asarray(reference_occupations[1], dtype=float)
        if occ_alpha.shape != (norb,) or occ_beta.shape != (norb,):
            raise ValueError(
                "reference_occupations must have shape (NORB,) for alpha and beta; "
                f"got {occ_alpha.shape} and {occ_beta.shape} for NORB={norb}."
            )

        def _two_body_fock(
            eri: Array, alpha_occ: Array, beta_occ: Array
        ) -> tuple[Array, Array]:
            dm_alpha = np.diag(alpha_occ)
            dm_beta = np.diag(beta_occ)
            dm_total = dm_alpha + dm_beta
            coulomb = np.einsum("pqrs,rs->pq", eri, dm_total, optimize=True)
            exchange_alpha = np.einsum("prsq,rs->pq", eri, dm_alpha, optimize=True)
            exchange_beta = np.einsum("prsq,rs->pq", eri, dm_beta, optimize=True)
            return coulomb - exchange_alpha, coulomb - exchange_beta

        full_fock_alpha, full_fock_beta = _two_body_fock(eri_src, occ_alpha, occ_beta)
        model_occ_alpha = occ_alpha[list(orbitals)]
        model_occ_beta = occ_beta[list(orbitals)]
        model_fock_alpha, model_fock_beta = _two_body_fock(
            eri_model, model_occ_alpha, model_occ_beta
        )
        full_fock_avg = 0.5 * (
            full_fock_alpha[np.ix_(orbitals, orbitals)]
            + full_fock_beta[np.ix_(orbitals, orbitals)]
        )
        model_fock_avg = 0.5 * (model_fock_alpha + model_fock_beta)
        h1_delta = np.asarray(full_fock_avg - model_fock_avg)
        h1_delta = 0.5 * (h1_delta + h1_delta.T.conj())
        h1_model = h1_model + h1_delta
        h1_correction_norm = float(np.linalg.norm(h1_delta))
    elif h1_correction != "none":
        raise ValueError("h1_correction must be one of {'none', 'reference_fock'}.")

    if nelec is None:
        nelec = int(ctx["NELEC"])
    if ms2 is None:
        ms2 = int(ctx.get("MS2", 0))
    ecore = float(ctx.get("ECORE", 0.0))
    orbsym = ctx.get("ORBSYM")
    model_orbsym = [int(orbsym[orb]) for orb in orbitals] if orbsym is not None else None

    out_path = Path(out).expanduser().resolve()
    pyscf_fcidump.from_integrals(
        str(out_path),
        h1_model,
        eri_model,
        nmodel,
        nelec,
        nuc=ecore,
        ms=ms2,
        orbsym=model_orbsym,
        tol=tol,
    )

    pair_full = ao2mo.restore(4, eri_selected, nmodel)
    pair_model = ao2mo.restore(4, eri_model, nmodel)
    eigvals = np.linalg.eigvalsh(0.5 * (pair_model + pair_model.T.conj()))
    full_norm = float(np.linalg.norm(pair_full))
    model_norm = float(np.linalg.norm(pair_model))
    residual_norm = float(np.linalg.norm(pair_full - pair_model))
    return {
        "out": str(out_path),
        "source_norb": norb,
        "norb": nmodel,
        "nelec": nelec,
        "ms2": ms2,
        "model_orbitals": list(orbitals),
        "fe_orbitals": sorted(fe_orbitals),
        "ligand_orbitals": sorted(ligand_orbitals),
        "h2_model": h2_model,
        "h1_correction": h1_correction,
        "h1_correction_norm": h1_correction_norm,
        "selected_pair_norm": full_norm,
        "model_pair_norm": model_norm,
        "discarded_pair_norm": residual_norm,
        "model_pair_min_eigenvalue": float(eigvals[0]) if eigvals.size else 0.0,
        "model_pair_max_eigenvalue": float(eigvals[-1]) if eigvals.size else 0.0,
        "model_pair_n_negative_eigenvalues": int(np.sum(eigvals < -1.0e-10)),
        "model_pair_negative_eigenvalue_norm": float(np.linalg.norm(eigvals[eigvals < 0.0])),
        "model_selected_pair_weight_fraction": (
            float((model_norm * model_norm) / (full_norm * full_norm))
            if full_norm > 0.0
            else 0.0
        ),
    }


def _kanamori_field_coeffs(
    coefficient: float,
    *,
    channel: str,
) -> tuple[complex, tuple[float, float], str]:
    if channel == "spin":
        return (1.0 if coefficient >= 0.0 else 1.0j), (1.0, -1.0), "spin"
    if channel == "charge":
        return (1.0j if coefficient >= 0.0 else 1.0), (1.0, 1.0), "charge"
    raise ValueError(f"Unsupported Kanamori field channel: {channel!r}")


def _kanamori_field_is_real(coefficient: float, *, channel: str) -> bool:
    factor, _, _ = _kanamori_field_coeffs(coefficient, channel=channel)
    return bool(np.isclose(np.imag(factor), 0.0))


def _append_kanamori_field(
    *,
    local_mode: Array,
    coefficient: float,
    local_chol: list[Array],
    local_factors: list[complex],
    local_spin_coeffs: list[tuple[float, float]],
    labels: list[str],
    basis_coeff: Array,
    global_orbitals: tuple[int, ...],
    norb: int,
    label: str,
    channel: str,
) -> Array:
    factor, spin_coeff, channel = _kanamori_field_coeffs(coefficient, channel=channel)
    scaled_mode = np.asarray(local_mode) * np.sqrt(abs(coefficient))
    global_mode = np.zeros((norb, norb), dtype=np.asarray(scaled_mode).dtype)
    global_mode[np.ix_(global_orbitals, global_orbitals)] = scaled_mode
    local_chol.append(_rotate_one_body_to_mo(global_mode, basis_coeff))
    local_factors.append(factor)
    local_spin_coeffs.append(spin_coeff)
    complexity = "complex" if abs(np.imag(factor)) > 0.0 else "real"
    labels.append(f"kanamori_{complexity}:{label}:{channel}")
    return _reconstruct_packed_pair_from_fields(
        np.asarray([scaled_mode]),
        np.asarray([factor], dtype=np.complex128),
        np.asarray([spin_coeff], dtype=np.float64),
    )


def _build_kanamori_sign_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    eri_residual = np.array(eri_ao, copy=True)
    h1_shift_ao = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)

    local_chol: list[Array] = []
    local_factors: list[complex] = []
    local_spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    center_reports: list[dict[str, Any]] = []

    for center_idx, orbitals in enumerate(centers):
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        block = np.asarray(eri_ao_arr[ix4])
        nloc = len(orbitals)
        full_block_pair = ao2mo.restore(4, block, nloc)
        full_block_pair = 0.5 * (full_block_pair + full_block_pair.T.conj())
        residual_pair = np.array(full_block_pair, copy=True)
        extracted_pair = np.zeros_like(full_block_pair)

        center_start = len(local_chol)
        kanamori_terms: list[dict[str, Any]] = []

        for local_orb, orb in enumerate(orbitals):
            pair_idx = _packed_pair_index(local_orb, local_orb)
            u_value = float(np.real(residual_pair[pair_idx, pair_idx]))
            if abs(u_value) <= float(chol_cut):
                continue
            mode = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
            mode[local_orb, local_orb] = 1.0
            contribution = _append_kanamori_field(
                local_mode=mode,
                coefficient=u_value,
                local_chol=local_chol,
                local_factors=local_factors,
                local_spin_coeffs=local_spin_coeffs,
                labels=labels,
                basis_coeff=basis_coeff,
                global_orbitals=orbitals,
                norb=norb,
                label=f"center{center_idx}:onsite{orb}",
                channel="spin" if u_value >= 0.0 else "charge",
            )
            residual_pair -= contribution
            extracted_pair += contribution
            kanamori_terms.append(
                {
                    "kind": "onsite_U",
                    "center": int(center_idx),
                    "orbital": int(orb),
                    "coefficient": u_value,
                    "preferred_decomposition": "spin" if u_value >= 0.0 else "charge",
                    "decomposition": "spin" if u_value >= 0.0 else "charge",
                }
            )

        for p in range(nloc):
            for q in range(p + 1, nloc):
                pp_idx = _packed_pair_index(p, p)
                qq_idx = _packed_pair_index(q, q)
                density_value = float(np.real(residual_pair[pp_idx, qq_idx]))
                if abs(density_value) > float(chol_cut):
                    amplitude = 0.5 * abs(density_value)
                    plus_mode = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                    plus_mode[p, p] = 1.0
                    plus_mode[q, q] = 1.0
                    minus_mode = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                    minus_mode[p, p] = 1.0
                    minus_mode[q, q] = -1.0
                    for mode, sign, component in (
                        (plus_mode, np.sign(density_value), "plus"),
                        (minus_mode, -np.sign(density_value), "minus"),
                    ):
                        contribution = _append_kanamori_field(
                            local_mode=mode,
                            coefficient=float(sign * amplitude),
                            local_chol=local_chol,
                            local_factors=local_factors,
                            local_spin_coeffs=local_spin_coeffs,
                            labels=labels,
                            basis_coeff=basis_coeff,
                            global_orbitals=orbitals,
                            norb=norb,
                            label=(
                                f"center{center_idx}:density{orbitals[p]}-{orbitals[q]}:"
                                f"{component}"
                            ),
                            channel="charge",
                        )
                        residual_pair -= contribution
                        extracted_pair += contribution
                    kanamori_terms.append(
                        {
                            "kind": "interorbital_Uprime",
                            "center": int(center_idx),
                            "orbitals": [int(orbitals[p]), int(orbitals[q])],
                            "coefficient": density_value,
                            "preferred_decomposition": (
                                "spin" if density_value >= 0.0 else "charge"
                            ),
                            "decomposition": "charge",
                        }
                    )

                pq_idx = _packed_pair_index(p, q)
                hund_value = float(np.real(residual_pair[pq_idx, pq_idx]))
                if abs(hund_value) <= float(chol_cut):
                    continue
                bond_mode = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                bond_mode[p, q] = 1.0
                bond_mode[q, p] = 1.0
                contribution = _append_kanamori_field(
                    local_mode=bond_mode,
                    coefficient=hund_value,
                    local_chol=local_chol,
                    local_factors=local_factors,
                    local_spin_coeffs=local_spin_coeffs,
                    labels=labels,
                    basis_coeff=basis_coeff,
                    global_orbitals=orbitals,
                    norb=norb,
                    label=f"center{center_idx}:hund{orbitals[p]}-{orbitals[q]}",
                    channel="charge",
                )
                residual_pair -= contribution
                extracted_pair += contribution
                kanamori_terms.append(
                    {
                        "kind": "hund_J_bond",
                        "center": int(center_idx),
                        "orbitals": [int(orbitals[p]), int(orbitals[q])],
                        "coefficient": hund_value,
                        "preferred_decomposition": "spin" if hund_value >= 0.0 else "charge",
                        "decomposition": "charge",
                    }
                )

        center_stop = len(local_chol)
        center_chol_ao = []
        for chol_mo in local_chol[center_start:center_stop]:
            center_chol_ao.append(np.asarray(basis_coeff) @ chol_mo @ np.asarray(basis_coeff).T.conj())
        center_chol_ao = np.asarray(center_chol_ao)
        if center_chol_ao.size:
            center_chol_ao = center_chol_ao[:, orbitals, :][:, :, orbitals]
        else:
            center_chol_ao = np.zeros((0, nloc, nloc), dtype=eri_ao_arr.dtype)
        center_factors = np.asarray(local_factors[center_start:center_stop], dtype=np.complex128)
        center_spin_coeffs = np.asarray(
            local_spin_coeffs[center_start:center_stop], dtype=np.float64
        )
        reconstructed_pair = _reconstruct_packed_pair_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )

        eri_residual[ix4] = ao2mo.restore(1, residual_pair, nloc)
        block_norm = float(np.linalg.norm(block.reshape(-1)))
        extracted_block_norm = float(np.linalg.norm(ao2mo.restore(1, extracted_pair, nloc)))
        residual_block_norm = float(np.linalg.norm(eri_residual[ix4].reshape(-1)))
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "kanamori_terms": kanamori_terms,
                "local_block_norm": block_norm,
                "extracted_local_block_norm": extracted_block_norm,
                "residual_center_block_norm": residual_block_norm,
                "packed_pair_norm": float(np.linalg.norm(full_block_pair)),
                "kanamori_packed_pair_norm": float(np.linalg.norm(extracted_pair)),
                "residual_packed_pair_norm": float(np.linalg.norm(residual_pair)),
                "packed_pair_reconstruction_relative_error": _relative_error(
                    extracted_pair, reconstructed_pair
                ),
                "spin_orbital_reconstruction_relative_error": _relative_error(
                    _spin_orbital_tensor_from_eri(ao2mo.restore(1, extracted_pair, nloc)),
                    _reconstruct_spin_orbital_tensor_from_fields(
                        center_chol_ao, center_factors, center_spin_coeffs
                    ),
                ),
                "n_local_real_fields": int(
                    sum(
                        label.startswith("kanamori_real:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "n_local_complex_fields": int(
                    sum(
                        label.startswith("kanamori_complex:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "parameters": _local_parameter_diagnostics(block, orbitals),
            }
        )

    residual_chol, residual_factors, residual_spin_coeffs, residual_labels, residual_diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_residual,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )

    if local_chol:
        chol = np.concatenate([np.asarray(local_chol), residual_chol], axis=0)
        field_factors = np.concatenate(
            [np.asarray(local_factors, dtype=np.complex128), residual_factors], axis=0
        )
        field_spin_coeffs = np.concatenate(
            [np.asarray(local_spin_coeffs, dtype=np.float64), residual_spin_coeffs], axis=0
        )
    else:
        chol = residual_chol
        field_factors = residual_factors
        field_spin_coeffs = residual_spin_coeffs
    labels.extend(residual_labels)

    n_local_real = sum(label.startswith("kanamori_real:") for label in labels)
    n_local_complex = sum(label.startswith("kanamori_complex:") for label in labels)
    n_residual_complex = sum(label == "residual_complex" for label in residual_labels)
    n_residual_real = sum(label == "residual_real" for label in residual_labels)
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    residual_frobenius_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))
    local_block_norm = float(
        np.sqrt(sum(report["local_block_norm"] ** 2 for report in center_reports))
    )
    extracted_local_block_norm = float(
        np.sqrt(sum(report["extracted_local_block_norm"] ** 2 for report in center_reports))
    )
    residual_center_block_norm = float(
        np.sqrt(sum(report["residual_center_block_norm"] ** 2 for report in center_reports))
    )

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    metadata: Dict[str, Any] = {
        "real_field_fit": "kanamori_sign",
        "centers": [list(center) for center in centers],
        "center_orbitals": list(sorted({orb for center in centers for orb in center})),
        "center_reports": center_reports,
        "local_real_fields": int(n_local_real),
        "local_complex_fields": int(n_local_complex),
        "residual_real_fields": int(n_residual_real),
        "residual_complex_fields": int(n_residual_complex),
        "n_local_real_fields": int(n_local_real),
        "n_local_complex_fields": int(n_local_complex),
        "n_residual_real_fields": int(n_residual_real),
        "n_residual_complex_fields": int(n_residual_complex),
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "local_block_norm": local_block_norm,
            "extracted_local_block_norm": extracted_local_block_norm,
            "residual_norm": residual_frobenius_norm,
            "center_block_norm": local_block_norm,
            "center_block_residual_norm": residual_center_block_norm,
            "full_pair_norm": full_pair_norm,
            **residual_diagnostics,
            "local_block_fraction_full_norm": _frac(local_block_norm, full_frobenius_norm),
            "local_block_fraction_full_weight": _frac(
                local_block_norm * local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
            "extracted_local_block_fraction_full_weight": _frac(
                extracted_local_block_norm * extracted_local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
        },
        "field_labels": tuple(labels),
    }

    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=eri_ao_arr.dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _build_kanamori_pair_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
    include_density_target: bool,
    real_field_fit: str,
    prefer_real: bool = True,
) -> RealFieldFitResult:
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    eri_residual = np.array(eri_ao, copy=True)
    h1_shift_ao = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)

    local_chol: list[Array] = []
    local_factors: list[complex] = []
    local_spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    center_reports: list[dict[str, Any]] = []

    def candidate_tensor(mode: Array, channel: str, coefficient: float = 1.0) -> Array:
        factor, spin_coeff, _ = _kanamori_field_coeffs(coefficient, channel=channel)
        return _reconstruct_spin_orbital_tensor_from_fields(
            np.asarray([mode]),
            np.asarray([factor], dtype=np.complex128),
            np.asarray([spin_coeff], dtype=np.float64),
        )

    for center_idx, orbitals in enumerate(centers):
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        block = np.asarray(eri_ao_arr[ix4])
        nloc = len(orbitals)
        full_block_pair = ao2mo.restore(4, block, nloc)
        full_block_pair = 0.5 * (full_block_pair + full_block_pair.T.conj())
        residual_pair = np.array(full_block_pair, copy=True)
        extracted_pair = np.zeros_like(full_block_pair)

        center_start = len(local_chol)
        kanamori_terms: list[dict[str, Any]] = []

        for local_orb, orb in enumerate(orbitals):
            pair_idx = _packed_pair_index(local_orb, local_orb)
            u_value = float(np.real(residual_pair[pair_idx, pair_idx]))
            if abs(u_value) <= float(chol_cut):
                continue
            mode = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
            mode[local_orb, local_orb] = 1.0
            channel = "spin" if u_value >= 0.0 else "charge"
            contribution = _append_kanamori_field(
                local_mode=mode,
                coefficient=u_value,
                local_chol=local_chol,
                local_factors=local_factors,
                local_spin_coeffs=local_spin_coeffs,
                labels=labels,
                basis_coeff=basis_coeff,
                global_orbitals=orbitals,
                norb=norb,
                label=f"center{center_idx}:onsite{orb}",
                channel=channel,
            )
            residual_pair -= contribution
            extracted_pair += contribution
            kanamori_terms.append(
                {
                    "kind": "onsite_U",
                    "center": int(center_idx),
                    "orbital": int(orb),
                    "coefficient": u_value,
                    "preferred_decomposition": channel,
                    "decomposition": channel,
                }
            )

        for p in range(nloc):
            for q in range(p + 1, nloc):
                pp_idx = _packed_pair_index(p, p)
                pq_idx = _packed_pair_index(p, q)
                qq_idx = _packed_pair_index(q, q)
                target_pair = np.zeros_like(residual_pair)
                target_entries = [(pq_idx, pq_idx)]
                if include_density_target:
                    target_entries.extend([(pp_idx, qq_idx), (qq_idx, pp_idx)])
                for a, b in target_entries:
                    target_pair[a, b] = residual_pair[a, b]
                if np.linalg.norm(target_pair) <= float(chol_cut):
                    continue

                target_spin = _spin_orbital_tensor_from_eri(ao2mo.restore(1, target_pair, nloc))
                plus = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                plus[p, p] = 1.0
                plus[q, q] = 1.0
                minus = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                minus[p, p] = 1.0
                minus[q, q] = -1.0
                bond = np.zeros((nloc, nloc), dtype=eri_ao_arr.dtype)
                bond[p, q] = 1.0
                bond[q, p] = 1.0
                hund_value = float(np.real(residual_pair[pq_idx, pq_idx]))
                preferred_bond = "spin" if hund_value >= 0.0 else "charge"
                other_bond = "charge" if preferred_bond == "spin" else "spin"
                fallback_candidates = []
                if include_density_target:
                    fallback_candidates.extend(
                        [
                            ("density_plus_charge", plus, "charge"),
                            ("density_minus_charge", minus, "charge"),
                            ("density_plus_spin", plus, "spin"),
                            ("density_minus_spin", minus, "spin"),
                        ]
                    )
                fallback_candidates.extend(
                    [
                        ("hund_preferred", bond, preferred_bond),
                        ("hund_other", bond, other_bond),
                    ]
                )
                rhs = target_spin.reshape(-1).real
                tolerance = max(10.0 * float(chol_cut), 1.0e-10)
                decomposition = "spin_orbital_lstsq"
                real_relerr = None

                if prefer_real:
                    from scipy.optimize import nnls

                    # The real continuous channels are positive spin squares and
                    # negative charge squares. The density modes are also allowed
                    # when only the Hund packed-pair entry is targeted; they supply
                    # the spin-dependent counterterms in the HK Hund identity.
                    real_candidates = [
                        ("density_plus_spin_real", plus, "spin", 1.0),
                        ("density_minus_spin_real", minus, "spin", 1.0),
                        ("density_plus_charge_real", plus, "charge", -1.0),
                        ("density_minus_charge_real", minus, "charge", -1.0),
                        ("hund_spin_real", bond, "spin", 1.0),
                        ("hund_charge_real", bond, "charge", -1.0),
                    ]
                    real_matrix = np.stack(
                        [
                            candidate_tensor(mode, channel, sign).reshape(-1).real
                            for _, mode, channel, sign in real_candidates
                        ],
                        axis=1,
                    )
                    magnitudes, _ = nnls(real_matrix, rhs)
                    real_coeffs = np.asarray(
                        [
                            sign * magnitude
                            for magnitude, (_name, _mode, _channel, sign) in zip(
                                magnitudes, real_candidates
                            )
                        ]
                    )
                    real_relerr = _relative_error(rhs, real_matrix @ magnitudes)
                    if real_relerr <= tolerance:
                        candidates = [
                            (name, mode, channel)
                            for name, mode, channel, _sign in real_candidates
                        ]
                        coeffs = real_coeffs
                        spin_relerr = real_relerr
                        decomposition = "real_spin_orbital_nnls"
                    else:
                        candidates = fallback_candidates
                        matrix = np.stack(
                            [
                                candidate_tensor(mode, channel).reshape(-1).real
                                for _, mode, channel in candidates
                            ],
                            axis=1,
                        )
                        coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
                        spin_relerr = _relative_error(rhs, matrix @ coeffs)
                else:
                    candidates = fallback_candidates
                    matrix = np.stack(
                        [
                            candidate_tensor(mode, channel).reshape(-1).real
                            for _, mode, channel in candidates
                        ],
                        axis=1,
                    )
                    coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
                    spin_relerr = _relative_error(rhs, matrix @ coeffs)

                if spin_relerr > tolerance:
                    continue

                pair_extracted = np.zeros_like(residual_pair)
                accepted_terms = []
                for coeff, (name, mode, channel) in zip(coeffs, candidates):
                    coeff = float(coeff)
                    if abs(coeff) <= float(chol_cut):
                        continue
                    contribution = _append_kanamori_field(
                        local_mode=mode,
                        coefficient=coeff,
                        local_chol=local_chol,
                        local_factors=local_factors,
                        local_spin_coeffs=local_spin_coeffs,
                        labels=labels,
                        basis_coeff=basis_coeff,
                        global_orbitals=orbitals,
                        norb=norb,
                        label=f"center{center_idx}:pair{orbitals[p]}-{orbitals[q]}:{name}",
                        channel=channel,
                    )
                    pair_extracted += contribution
                    accepted_terms.append(
                        {
                            "component": name,
                            "coefficient": coeff,
                            "channel": channel,
                            "real_field": _kanamori_field_is_real(coeff, channel=channel),
                        }
                    )
                residual_pair -= pair_extracted
                extracted_pair += pair_extracted
                kanamori_terms.append(
                    {
                        "kind": "pair_full" if include_density_target else "hund_J_pair",
                        "center": int(center_idx),
                        "orbitals": [int(orbitals[p]), int(orbitals[q])],
                        "preferred_decomposition": preferred_bond,
                        "decomposition": decomposition,
                        "spin_orbital_fit_relative_error": spin_relerr,
                        "real_spin_orbital_fit_relative_error": real_relerr,
                        "components": accepted_terms,
                    }
                )

        center_stop = len(local_chol)
        center_chol_ao = []
        for chol_mo in local_chol[center_start:center_stop]:
            center_chol_ao.append(np.asarray(basis_coeff) @ chol_mo @ np.asarray(basis_coeff).T.conj())
        center_chol_ao = np.asarray(center_chol_ao)
        if center_chol_ao.size:
            center_chol_ao = center_chol_ao[:, orbitals, :][:, :, orbitals]
        else:
            center_chol_ao = np.zeros((0, nloc, nloc), dtype=eri_ao_arr.dtype)
        center_factors = np.asarray(local_factors[center_start:center_stop], dtype=np.complex128)
        center_spin_coeffs = np.asarray(
            local_spin_coeffs[center_start:center_stop], dtype=np.float64
        )
        reconstructed_pair = _reconstruct_packed_pair_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )
        eri_residual[ix4] = ao2mo.restore(1, residual_pair, nloc)
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "kanamori_terms": kanamori_terms,
                "local_block_norm": float(np.linalg.norm(block.reshape(-1))),
                "extracted_local_block_norm": float(
                    np.linalg.norm(ao2mo.restore(1, extracted_pair, nloc))
                ),
                "residual_center_block_norm": float(np.linalg.norm(eri_residual[ix4].reshape(-1))),
                "packed_pair_norm": float(np.linalg.norm(full_block_pair)),
                "kanamori_packed_pair_norm": float(np.linalg.norm(extracted_pair)),
                "residual_packed_pair_norm": float(np.linalg.norm(residual_pair)),
                "packed_pair_reconstruction_relative_error": _relative_error(
                    extracted_pair, reconstructed_pair
                ),
                "spin_orbital_reconstruction_relative_error": _relative_error(
                    _spin_orbital_tensor_from_eri(ao2mo.restore(1, extracted_pair, nloc)),
                    _reconstruct_spin_orbital_tensor_from_fields(
                        center_chol_ao, center_factors, center_spin_coeffs
                    ),
                ),
                "n_local_real_fields": int(
                    sum(
                        label.startswith("kanamori_real:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "n_local_complex_fields": int(
                    sum(
                        label.startswith("kanamori_complex:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "parameters": _local_parameter_diagnostics(block, orbitals),
            }
        )

    residual_chol, residual_factors, residual_spin_coeffs, residual_labels, residual_diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_residual,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )
    if local_chol:
        chol = np.concatenate([np.asarray(local_chol), residual_chol], axis=0)
        field_factors = np.concatenate(
            [np.asarray(local_factors, dtype=np.complex128), residual_factors], axis=0
        )
        field_spin_coeffs = np.concatenate(
            [np.asarray(local_spin_coeffs, dtype=np.float64), residual_spin_coeffs], axis=0
        )
    else:
        chol = residual_chol
        field_factors = residual_factors
        field_spin_coeffs = residual_spin_coeffs
    labels.extend(residual_labels)

    n_local_real = sum(label.startswith("kanamori_real:") for label in labels)
    n_local_complex = sum(label.startswith("kanamori_complex:") for label in labels)
    n_residual_complex = sum(label == "residual_complex" for label in residual_labels)
    n_residual_real = sum(label == "residual_real" for label in residual_labels)
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    residual_frobenius_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))
    local_block_norm = float(
        np.sqrt(sum(report["local_block_norm"] ** 2 for report in center_reports))
    )
    extracted_local_block_norm = float(
        np.sqrt(sum(report["extracted_local_block_norm"] ** 2 for report in center_reports))
    )
    residual_center_block_norm = float(
        np.sqrt(sum(report["residual_center_block_norm"] ** 2 for report in center_reports))
    )

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    metadata: Dict[str, Any] = {
        "real_field_fit": real_field_fit,
        "centers": [list(center) for center in centers],
        "center_orbitals": list(sorted({orb for center in centers for orb in center})),
        "center_reports": center_reports,
        "local_real_fields": int(n_local_real),
        "local_complex_fields": int(n_local_complex),
        "residual_real_fields": int(n_residual_real),
        "residual_complex_fields": int(n_residual_complex),
        "n_local_real_fields": int(n_local_real),
        "n_local_complex_fields": int(n_local_complex),
        "n_residual_real_fields": int(n_residual_real),
        "n_residual_complex_fields": int(n_residual_complex),
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "local_block_norm": local_block_norm,
            "extracted_local_block_norm": extracted_local_block_norm,
            "residual_norm": residual_frobenius_norm,
            "center_block_norm": local_block_norm,
            "center_block_residual_norm": residual_center_block_norm,
            "full_pair_norm": full_pair_norm,
            **residual_diagnostics,
            "local_block_fraction_full_norm": _frac(local_block_norm, full_frobenius_norm),
            "local_block_fraction_full_weight": _frac(
                local_block_norm * local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
            "extracted_local_block_fraction_full_weight": _frac(
                extracted_local_block_norm * extracted_local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
        },
        "field_labels": tuple(labels),
    }
    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=eri_ao_arr.dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _build_kanamori_sign_full_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    return _build_kanamori_pair_real_fields_from_eri(
        eri_ao,
        basis_coeff=basis_coeff,
        centers=centers,
        chol_cut=chol_cut,
        include_density_target=True,
        real_field_fit="kanamori_sign_full",
    )


def _build_kanamori_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    return _build_kanamori_pair_real_fields_from_eri(
        eri_ao,
        basis_coeff=basis_coeff,
        centers=centers,
        chol_cut=chol_cut,
        include_density_target=True,
        real_field_fit="kanamori_real",
        prefer_real=True,
    )


def _build_kanamori_uj_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    return _build_kanamori_pair_real_fields_from_eri(
        eri_ao,
        basis_coeff=basis_coeff,
        centers=centers,
        chol_cut=chol_cut,
        include_density_target=False,
        real_field_fit="kanamori_uj",
    )


def _build_charge_spin_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    """Factor selected local blocks through the formal charge block.

    For the spin-independent FCIDUMP Hamiltonians handled here, rotating the
    alpha/beta kernel to the {charge, spin} basis leaves only the charge-charge
    block. Spin and charge-spin mixed blocks are exact zeros unless a null-channel
    freedom is introduced separately.
    """
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    eri_residual = np.array(eri_ao, copy=True)
    h1_shift_ao = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)

    local_chol: list[Array] = []
    local_factors: list[complex] = []
    local_spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    center_reports: list[dict[str, Any]] = []

    for center_idx, orbitals in enumerate(centers):
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        block = np.asarray(eri_ao_arr[ix4])
        nloc = len(orbitals)
        full_block_pair = ao2mo.restore(4, block, nloc)
        full_block_pair = 0.5 * (full_block_pair + full_block_pair.T.conj())

        eigvals, eigvecs = np.linalg.eigh(full_block_pair)
        keep = np.abs(eigvals) > float(chol_cut)

        center_start = len(local_chol)
        for local_mode_idx, value in enumerate(eigvals[keep]):
            vec = eigvecs[:, keep][:, local_mode_idx]
            small = _unpack_pair_vector(vec * np.sqrt(abs(value)), nloc)
            small = 0.5 * (small + small.T.conj())
            global_mode = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)
            global_mode[np.ix_(orbitals, orbitals)] = small
            local_chol.append(_rotate_one_body_to_mo(global_mode, basis_coeff))

            if value > 0.0:
                local_factors.append(1.0j)
                labels.append(f"charge_spin_complex:center{center_idx}:charge_mode{local_mode_idx}")
            else:
                local_factors.append(1.0)
                labels.append(f"charge_spin_real:center{center_idx}:charge_mode{local_mode_idx}")
            local_spin_coeffs.append((1.0, 1.0))

        center_stop = len(local_chol)
        center_chol_ao = []
        for chol_mo in local_chol[center_start:center_stop]:
            center_chol_ao.append(np.asarray(basis_coeff) @ chol_mo @ np.asarray(basis_coeff).T.conj())
        center_chol_ao = np.asarray(center_chol_ao)
        if center_chol_ao.size:
            center_chol_ao = center_chol_ao[:, orbitals, :][:, :, orbitals]
        else:
            center_chol_ao = np.zeros((0, nloc, nloc), dtype=eri_ao_arr.dtype)
        center_factors = np.asarray(local_factors[center_start:center_stop], dtype=np.complex128)
        center_spin_coeffs = np.asarray(
            local_spin_coeffs[center_start:center_stop], dtype=np.float64
        )
        reconstructed_pair = _reconstruct_packed_pair_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )
        reconstructed_spin_orbital = _reconstruct_spin_orbital_tensor_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )
        reference_spin_orbital = _spin_orbital_tensor_from_eri(block)
        discarded = eigvals[~keep]
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "local_block_norm": float(np.linalg.norm(block.reshape(-1))),
                "extracted_local_block_norm": float(
                    np.linalg.norm(ao2mo.restore(1, np.asarray(reconstructed_pair), nloc))
                ),
                "residual_center_block_norm": 0.0,
                "packed_pair_norm": float(np.linalg.norm(full_block_pair)),
                "charge_pair_norm": float(np.linalg.norm(full_block_pair)),
                "spin_pair_norm": 0.0,
                "charge_spin_mixed_pair_norm": 0.0,
                "retained_packed_pair_norm": float(np.linalg.norm(eigvals[keep])),
                "discarded_packed_pair_norm": float(np.linalg.norm(discarded)),
                "packed_pair_reconstruction_relative_error": _relative_error(
                    full_block_pair, reconstructed_pair
                ),
                "spin_orbital_reconstruction_relative_error": _relative_error(
                    reference_spin_orbital, reconstructed_spin_orbital
                ),
                "n_charge_fields": int(center_stop - center_start),
                "n_spin_fields": 0,
                "n_mixed_charge_spin_fields": 0,
                "n_local_real_fields": int(
                    sum(
                        label.startswith(f"charge_spin_real:center{center_idx}:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "n_local_complex_fields": int(
                    sum(
                        label.startswith(f"charge_spin_complex:center{center_idx}:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "parameters": _local_parameter_diagnostics(block, orbitals),
            }
        )

        eri_residual[ix4] = 0.0

    residual_chol, residual_factors, residual_spin_coeffs, residual_labels, residual_diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_residual,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )

    if local_chol:
        chol = np.concatenate([np.asarray(local_chol), residual_chol], axis=0)
        field_factors = np.concatenate(
            [np.asarray(local_factors, dtype=np.complex128), residual_factors], axis=0
        )
        field_spin_coeffs = np.concatenate(
            [np.asarray(local_spin_coeffs, dtype=np.float64), residual_spin_coeffs], axis=0
        )
    else:
        chol = residual_chol
        field_factors = residual_factors
        field_spin_coeffs = residual_spin_coeffs
    labels.extend(residual_labels)

    n_local_real = sum(label.startswith("charge_spin_real:") for label in labels)
    n_local_complex = sum(label.startswith("charge_spin_complex:") for label in labels)
    n_residual_complex = sum(label == "residual_complex" for label in residual_labels)
    n_residual_real = sum(label == "residual_real" for label in residual_labels)
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    residual_frobenius_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))
    local_block_norm = float(
        np.sqrt(sum(report["local_block_norm"] ** 2 for report in center_reports))
    )
    extracted_local_block_norm = float(
        np.sqrt(sum(report["extracted_local_block_norm"] ** 2 for report in center_reports))
    )

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    metadata: Dict[str, Any] = {
        "real_field_fit": "charge_spin",
        "centers": [list(center) for center in centers],
        "center_orbitals": list(sorted({orb for center in centers for orb in center})),
        "center_reports": center_reports,
        "local_real_fields": int(n_local_real),
        "local_complex_fields": int(n_local_complex),
        "residual_real_fields": int(n_residual_real),
        "residual_complex_fields": int(n_residual_complex),
        "n_local_real_fields": int(n_local_real),
        "n_local_complex_fields": int(n_local_complex),
        "n_residual_real_fields": int(n_residual_real),
        "n_residual_complex_fields": int(n_residual_complex),
        "charge_spin_blocks": {
            "charge_pair_norm": local_block_norm,
            "spin_pair_norm": 0.0,
            "mixed_pair_norm": 0.0,
            "note": (
                "Restricted spin-independent FCIDUMP integrals rotate to a nonzero "
                "charge-charge block and exact-zero spin/mixed blocks in the formal "
                "{charge, spin} E_pq basis."
            ),
        },
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "local_block_norm": local_block_norm,
            "extracted_local_block_norm": extracted_local_block_norm,
            "residual_norm": residual_frobenius_norm,
            "center_block_norm": local_block_norm,
            "center_block_residual_norm": 0.0,
            "full_pair_norm": full_pair_norm,
            **residual_diagnostics,
            "local_block_fraction_full_norm": _frac(local_block_norm, full_frobenius_norm),
            "local_block_fraction_full_weight": _frac(
                local_block_norm * local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
            "extracted_local_block_fraction_full_weight": _frac(
                extracted_local_block_norm * extracted_local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
        },
        "field_labels": tuple(labels),
    }

    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=eri_ao_arr.dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _build_full_charge_spin_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    chol_cut: float,
) -> RealFieldFitResult:
    """Factor the full spin-independent FCIDUMP tensor in the formal charge sector."""
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    h1_shift_ao = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)

    chol, field_factors, field_spin_coeffs, labels, diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_ao_arr,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )
    full_labels = tuple(
        f"charge_spin_full_{'complex' if label == 'residual_complex' else 'real'}:"
        f"charge_mode{idx}"
        for idx, label in enumerate(labels)
    )
    n_full_real = sum(label.startswith("charge_spin_full_real:") for label in full_labels)
    n_full_complex = sum(label.startswith("charge_spin_full_complex:") for label in full_labels)
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))

    metadata: Dict[str, Any] = {
        "real_field_fit": "charge_spin",
        "decomposition_scope": "full",
        "centers": [],
        "center_orbitals": [],
        "center_reports": [],
        "full_real_fields": int(n_full_real),
        "full_complex_fields": int(n_full_complex),
        "n_full_real_fields": int(n_full_real),
        "n_full_complex_fields": int(n_full_complex),
        "local_real_fields": 0,
        "local_complex_fields": 0,
        "residual_real_fields": 0,
        "residual_complex_fields": 0,
        "n_local_real_fields": 0,
        "n_local_complex_fields": 0,
        "n_residual_real_fields": 0,
        "n_residual_complex_fields": 0,
        "n_full_charge_dominant_fields": int(n_full_real + n_full_complex),
        "n_full_spin_dominant_fields": 0,
        "n_full_mixed_charge_spin_fields": 0,
        "charge_spin_blocks": {
            "charge_pair_norm": full_pair_norm,
            "spin_pair_norm": 0.0,
            "mixed_pair_norm": 0.0,
            "note": (
                "Restricted spin-independent FCIDUMP integrals rotate to a nonzero "
                "charge-charge block and exact-zero spin/mixed blocks in the formal "
                "{charge, spin} E_pq basis."
            ),
        },
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "local_block_norm": 0.0,
            "extracted_local_block_norm": full_frobenius_norm,
            "residual_norm": 0.0,
            "center_block_norm": 0.0,
            "center_block_residual_norm": 0.0,
            "full_pair_norm": full_pair_norm,
            "full_pair_retained_norm": diagnostics["residual_pair_retained_norm"],
            "full_pair_reconstruction_error_norm": diagnostics[
                "residual_pair_reconstruction_error_norm"
            ],
            "full_pair_reconstruction_relative_error": diagnostics[
                "residual_pair_reconstruction_relative_error"
            ],
            "n_full_pair_positive_eigenvalues": diagnostics[
                "n_residual_pair_positive_eigenvalues"
            ],
            "n_full_pair_negative_eigenvalues": diagnostics[
                "n_residual_pair_negative_eigenvalues"
            ],
            "n_full_pair_discarded_eigenvalues": diagnostics[
                "n_residual_pair_discarded_eigenvalues"
            ],
            "local_block_fraction_full_norm": 0.0,
            "local_block_fraction_full_weight": 0.0,
            "extracted_local_block_fraction_full_weight": (
                1.0 if full_frobenius_norm > 0.0 else 0.0
            ),
        },
        "field_labels": full_labels,
    }

    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=eri_ao_arr.dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _block_diag2(a: Array, b: Array) -> Array:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    out = np.zeros(
        (a_arr.shape[0] + b_arr.shape[0], a_arr.shape[1] + b_arr.shape[1]),
        dtype=np.result_type(a_arr, b_arr),
    )
    out[: a_arr.shape[0], : a_arr.shape[1]] = a_arr
    out[a_arr.shape[0] :, a_arr.shape[1] :] = b_arr
    return out


def _block_diag_chol_to_unrestricted(chol: Array, norb: int) -> Array:
    chol_arr = np.asarray(chol)
    if chol_arr.shape[0] == 0:
        return np.zeros((0, 2, norb, norb), dtype=chol_arr.dtype)
    return np.stack(
        [
            chol_arr[:, :norb, :norb],
            chol_arr[:, norb : 2 * norb, norb : 2 * norb],
        ],
        axis=1,
    )


def _append_uhf_unrestricted_real_spin_field(
    *,
    scaled_local_mode: Array,
    local_chol: list[Array],
    local_factors: list[complex],
    labels: list[str],
    coeff_alpha: Array,
    coeff_beta: Array,
    global_orbitals: tuple[int, ...],
    norb: int,
    label: str,
) -> Array:
    scaled = np.asarray(scaled_local_mode)
    global_mode = np.zeros((norb, norb), dtype=scaled.dtype)
    global_mode[np.ix_(global_orbitals, global_orbitals)] = scaled
    alpha_mode = np.asarray(coeff_alpha).T.conj() @ global_mode @ np.asarray(coeff_alpha)
    beta_mode = -(np.asarray(coeff_beta).T.conj() @ global_mode @ np.asarray(coeff_beta))
    local_chol.append(np.stack([alpha_mode, beta_mode], axis=0))
    local_factors.append(1.0)
    labels.append(f"uhf_local_real:spin_{label}")
    return _reconstruct_packed_pair_from_fields(
        np.asarray([scaled]),
        np.asarray([1.0], dtype=np.complex128),
        np.asarray([[1.0, -1.0]], dtype=np.float64),
    )


def _pack_eri_pair_block(eri: Array, *, symmetrize_exchange: bool) -> Array:
    arr = np.asarray(eri)
    norb = int(arr.shape[0])
    pairs = [(p, q) for p in range(norb) for q in range(p + 1)]
    packed = np.empty((len(pairs), len(pairs)), dtype=arr.dtype)
    for left, (p, q) in enumerate(pairs):
        for right, (r, s) in enumerate(pairs):
            packed[left, right] = 0.25 * (
                arr[p, q, r, s]
                + arr[q, p, r, s]
                + arr[p, q, s, r]
                + arr[q, p, s, r]
            )
    if symmetrize_exchange:
        packed = 0.5 * (packed + packed.T.conj())
    return packed


def _transform_eri_pair_basis(eri: Array, coeff_left: Array, coeff_right: Array) -> Array:
    left = np.asarray(coeff_left)
    right = np.asarray(coeff_right)
    return np.einsum(
        "up,vq,wr,xs,uvwx->pqrs",
        left.conj(),
        left.conj(),
        right,
        right,
        np.asarray(eri),
        optimize=True,
    )


def _uhf_charge_spin_supermatrix_from_eri(
    eri: Array,
    *,
    coeff_alpha: Array,
    coeff_beta: Array,
) -> tuple[Array, dict[str, float]]:
    vaa = _pack_eri_pair_block(
        _transform_eri_pair_basis(eri, coeff_alpha, coeff_alpha),
        symmetrize_exchange=True,
    )
    vab = _pack_eri_pair_block(
        _transform_eri_pair_basis(eri, coeff_alpha, coeff_beta),
        symmetrize_exchange=False,
    )
    vba = _pack_eri_pair_block(
        _transform_eri_pair_basis(eri, coeff_beta, coeff_alpha),
        symmetrize_exchange=False,
    )
    vbb = _pack_eri_pair_block(
        _transform_eri_pair_basis(eri, coeff_beta, coeff_beta),
        symmetrize_exchange=True,
    )
    npair = int(vaa.shape[0])
    ab = np.block([[vaa, vab], [vba, vbb]])
    ab = 0.5 * (ab + ab.T.conj())
    identity = np.eye(npair)
    j_spin = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    transform = np.kron(j_spin, identity)
    charge_spin = 0.25 * (transform.T.conj() @ ab @ transform)
    charge_spin = 0.5 * (charge_spin + charge_spin.T.conj())
    diagnostics = {
        "alpha_beta_pair_norm": float(np.linalg.norm(ab)),
        "charge_charge_pair_norm": float(np.linalg.norm(charge_spin[:npair, :npair])),
        "charge_spin_pair_norm": float(np.linalg.norm(charge_spin[:npair, npair:])),
        "spin_charge_pair_norm": float(np.linalg.norm(charge_spin[npair:, :npair])),
        "spin_spin_pair_norm": float(np.linalg.norm(charge_spin[npair:, npair:])),
    }
    return charge_spin, diagnostics


def _factorize_uhf_charge_spin_supermatrix(
    supermat_cs: Array,
    *,
    norb: int,
    chol_cut: float,
    label_prefix: str,
) -> tuple[Array, Array, list[str], dict[str, float | int]]:
    npair = norb * (norb + 1) // 2
    supermat = 0.5 * (np.asarray(supermat_cs) + np.asarray(supermat_cs).T.conj())
    eigvals, eigvecs = np.linalg.eigh(supermat)
    keep = np.abs(eigvals) > float(chol_cut)
    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]
    discarded_vals = eigvals[~keep]
    pair_norm = float(np.linalg.norm(eigvals))
    diagnostics: dict[str, float | int] = {
        f"{label_prefix}_pair_norm": pair_norm,
        f"{label_prefix}_pair_retained_norm": float(np.linalg.norm(kept_vals)),
        f"{label_prefix}_pair_reconstruction_error_norm": float(np.linalg.norm(discarded_vals)),
        f"{label_prefix}_pair_reconstruction_relative_error": (
            float(np.linalg.norm(discarded_vals)) / pair_norm if pair_norm > 0.0 else 0.0
        ),
        f"{label_prefix}_pair_positive_eigenvalues": int(np.sum(kept_vals > 0.0)),
        f"{label_prefix}_pair_negative_eigenvalues": int(np.sum(kept_vals < 0.0)),
        f"{label_prefix}_pair_discarded_eigenvalues": int(np.sum(~keep)),
    }

    chol_blocks: list[Array] = []
    factors: list[complex] = []
    labels: list[str] = []
    for idx, value in enumerate(kept_vals):
        vec = kept_vecs[:, idx]
        charge_mode = _unpack_pair_vector(vec[:npair] * np.sqrt(abs(value)), norb)
        spin_mode = _unpack_pair_vector(vec[npair:] * np.sqrt(abs(value)), norb)
        charge_mode = 0.5 * (charge_mode + charge_mode.T.conj())
        spin_mode = 0.5 * (spin_mode + spin_mode.T.conj())
        alpha_mode = charge_mode + spin_mode
        beta_mode = charge_mode - spin_mode
        chol_blocks.append(_block_diag2(alpha_mode, beta_mode))
        if value > 0.0:
            factors.append(1.0j)
        else:
            factors.append(1.0)

        charge_norm = float(np.linalg.norm(charge_mode))
        spin_norm = float(np.linalg.norm(spin_mode))
        if spin_norm <= 1.0e-10 * max(1.0, charge_norm):
            channel = "charge"
        elif charge_norm <= 1.0e-10 * max(1.0, spin_norm):
            channel = "spin"
        else:
            channel = "mixed"
        complexity = "complex" if value > 0.0 else "real"
        labels.append(f"{label_prefix}_{complexity}:{channel}_mode{idx}")

    if not chol_blocks:
        return (
            np.zeros((0, 2 * norb, 2 * norb), dtype=np.asarray(supermat_cs).dtype),
            np.zeros((0,), dtype=np.complex128),
            [],
            diagnostics,
        )
    return (
        np.asarray(chol_blocks),
        np.asarray(factors, dtype=np.complex128),
        labels,
        diagnostics,
    )


def _relabel_uhf_charge_spin_block_labels(
    labels: list[str],
    *,
    label_prefix: str,
    block_name: str,
) -> list[str]:
    relabeled: list[str] = []
    for label in labels:
        for complexity in ("complex", "real"):
            prefix = f"{label_prefix}_{block_name}_{complexity}:"
            if label.startswith(prefix):
                channel_and_mode = label.removeprefix(prefix)
                channel, _, mode = channel_and_mode.partition("_")
                relabeled.append(f"{label_prefix}_{complexity}:{channel}_{block_name}_{mode}")
                break
        else:
            relabeled.append(label)
    return relabeled


def _factorize_uhf_charge_spin_blocked_supermatrix(
    supermat_cs: Array,
    *,
    norb: int,
    chol_cut: float,
    label_prefix: str,
) -> tuple[Array, Array, list[str], dict[str, float | int]]:
    npair = norb * (norb + 1) // 2
    supermat = 0.5 * (np.asarray(supermat_cs) + np.asarray(supermat_cs).T.conj())
    charge = np.zeros_like(supermat)
    spin = np.zeros_like(supermat)
    charge[:npair, :npair] = supermat[:npair, :npair]
    spin[npair:, npair:] = supermat[npair:, npair:]
    residual = supermat - charge - spin

    all_chol: list[Array] = []
    all_factors: list[Array] = []
    all_labels: list[str] = []
    diagnostics: dict[str, float | int] = {}
    err_sq = 0.0
    retained_sq = 0.0
    n_pos = 0
    n_neg = 0
    n_discarded = 0
    for block_name, block in (
        ("charge", charge),
        ("spin", spin),
        ("residual", residual),
    ):
        block_chol, block_factors, block_labels, block_diagnostics = (
            _factorize_uhf_charge_spin_supermatrix(
                block,
                norb=norb,
                chol_cut=chol_cut,
                label_prefix=f"{label_prefix}_{block_name}",
            )
        )
        diagnostics.update(block_diagnostics)
        if block_chol.shape[0]:
            all_chol.append(block_chol)
            all_factors.append(block_factors)
            all_labels.extend(
                _relabel_uhf_charge_spin_block_labels(
                    block_labels,
                    label_prefix=label_prefix,
                    block_name=block_name,
                )
            )
        err_sq += float(
            block_diagnostics[f"{label_prefix}_{block_name}_pair_reconstruction_error_norm"]
        ) ** 2
        retained_sq += float(
            block_diagnostics[f"{label_prefix}_{block_name}_pair_retained_norm"]
        ) ** 2
        n_pos += int(block_diagnostics[f"{label_prefix}_{block_name}_pair_positive_eigenvalues"])
        n_neg += int(block_diagnostics[f"{label_prefix}_{block_name}_pair_negative_eigenvalues"])
        n_discarded += int(
            block_diagnostics[f"{label_prefix}_{block_name}_pair_discarded_eigenvalues"]
        )

    pair_norm = float(np.linalg.norm(np.linalg.eigvalsh(supermat)))
    err_norm = float(np.sqrt(err_sq))
    diagnostics.update(
        {
            f"{label_prefix}_pair_norm": pair_norm,
            f"{label_prefix}_pair_retained_norm": float(np.sqrt(retained_sq)),
            f"{label_prefix}_pair_reconstruction_error_norm": err_norm,
            f"{label_prefix}_pair_reconstruction_relative_error": (
                err_norm / pair_norm if pair_norm > 0.0 else 0.0
            ),
            f"{label_prefix}_pair_positive_eigenvalues": n_pos,
            f"{label_prefix}_pair_negative_eigenvalues": n_neg,
            f"{label_prefix}_pair_discarded_eigenvalues": n_discarded,
            f"{label_prefix}_charge_block_norm": float(np.linalg.norm(charge)),
            f"{label_prefix}_spin_block_norm": float(np.linalg.norm(spin)),
            f"{label_prefix}_charge_spin_residual_block_norm": float(np.linalg.norm(residual)),
        }
    )

    if not all_chol:
        return (
            np.zeros((0, 2 * norb, 2 * norb), dtype=supermat.dtype),
            np.zeros((0,), dtype=np.complex128),
            [],
            diagnostics,
        )
    return (
        np.concatenate(all_chol, axis=0),
        np.concatenate(all_factors, axis=0),
        all_labels,
        diagnostics,
    )


def _extract_uhf_local_real_spin_fields(
    eri_full: Array,
    *,
    coeff_alpha: Array,
    coeff_beta: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> tuple[Array, Array, tuple[str, ...], Array, dict[str, Any]]:
    from pyscf import ao2mo

    norb = int(np.asarray(eri_full).shape[0])
    eri_arr = np.asarray(eri_full)
    eri_residual = np.array(eri_arr, copy=True)
    local_chol: list[Array] = []
    local_factors: list[complex] = []
    labels: list[str] = []
    center_reports: list[dict[str, Any]] = []

    for center_idx, orbitals in enumerate(centers):
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        block = np.asarray(eri_residual[ix4])
        nloc = len(orbitals)
        full_block_pair = ao2mo.restore(4, block, nloc)
        full_block_pair = 0.5 * (full_block_pair + full_block_pair.T.conj())
        residual_pair = np.array(full_block_pair, copy=True)
        extracted_pair = np.zeros_like(full_block_pair)
        center_start = len(local_chol)
        extracted_terms: list[dict[str, Any]] = []

        for local_orb, orb in enumerate(orbitals):
            pair_idx = _packed_pair_index(local_orb, local_orb)
            u_value = float(np.real(residual_pair[pair_idx, pair_idx]))
            if u_value <= float(chol_cut):
                continue
            scaled_mode = np.zeros((nloc, nloc), dtype=eri_arr.dtype)
            scaled_mode[local_orb, local_orb] = np.sqrt(u_value)
            contribution = _append_uhf_unrestricted_real_spin_field(
                scaled_local_mode=scaled_mode,
                local_chol=local_chol,
                local_factors=local_factors,
                labels=labels,
                coeff_alpha=coeff_alpha,
                coeff_beta=coeff_beta,
                global_orbitals=orbitals,
                norb=norb,
                label=f"center{center_idx}:onsite{orb}",
            )
            residual_pair -= contribution
            extracted_pair += contribution
            extracted_terms.append(
                {
                    "kind": "onsite_U",
                    "center": int(center_idx),
                    "orbital": int(orb),
                    "coefficient": u_value,
                }
            )

        residual_pair = 0.5 * (residual_pair + residual_pair.T.conj())
        eigvals, eigvecs = np.linalg.eigh(residual_pair)
        keep = eigvals > float(chol_cut)
        spin_tol = max(float(chol_cut), 1.0e-12)
        for local_mode_idx, value in enumerate(eigvals[keep]):
            vec = eigvecs[:, keep][:, local_mode_idx]
            scaled_mode = _unpack_pair_vector(vec * np.sqrt(float(value)), nloc)
            scaled_mode = 0.5 * (scaled_mode + scaled_mode.T.conj())
            if not _mode_is_exact_real_spin_channel(scaled_mode, float(value), tol=spin_tol):
                continue
            contribution = _append_uhf_unrestricted_real_spin_field(
                scaled_local_mode=scaled_mode,
                local_chol=local_chol,
                local_factors=local_factors,
                labels=labels,
                coeff_alpha=coeff_alpha,
                coeff_beta=coeff_beta,
                global_orbitals=orbitals,
                norb=norb,
                label=f"center{center_idx}:mode{local_mode_idx}",
            )
            residual_pair -= contribution
            extracted_pair += contribution
            extracted_terms.append(
                {
                    "kind": "rank_one_positive_spin_mode",
                    "center": int(center_idx),
                    "mode": int(local_mode_idx),
                    "coefficient": float(value),
                }
            )

        residual_pair = 0.5 * (residual_pair + residual_pair.T.conj())
        eri_residual[ix4] = ao2mo.restore(1, residual_pair, nloc)
        center_stop = len(local_chol)
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "extracted_terms": extracted_terms,
                "local_block_norm": float(np.linalg.norm(block.reshape(-1))),
                "extracted_local_block_norm": float(
                    np.linalg.norm(ao2mo.restore(1, extracted_pair, nloc))
                ),
                "residual_center_block_norm": float(np.linalg.norm(eri_residual[ix4].reshape(-1))),
                "packed_pair_norm": float(np.linalg.norm(full_block_pair)),
                "extracted_packed_pair_norm": float(np.linalg.norm(extracted_pair)),
                "residual_packed_pair_norm": float(np.linalg.norm(residual_pair)),
                "n_local_real_fields": int(center_stop - center_start),
                "n_local_complex_fields": 0,
                "parameters": _local_parameter_diagnostics(block, orbitals),
            }
        )

    if local_chol:
        chol = np.asarray(local_chol)
        factors = np.asarray(local_factors, dtype=np.complex128)
    else:
        chol = np.zeros((0, 2, norb, norb), dtype=eri_arr.dtype)
        factors = np.zeros((0,), dtype=np.complex128)

    full_norm = float(np.linalg.norm(eri_arr.reshape(-1)))
    residual_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    local_block_norm = float(
        np.sqrt(sum(report["local_block_norm"] ** 2 for report in center_reports))
    )
    extracted_local_block_norm = float(
        np.sqrt(sum(report["extracted_local_block_norm"] ** 2 for report in center_reports))
    )
    residual_center_block_norm = float(
        np.sqrt(sum(report["residual_center_block_norm"] ** 2 for report in center_reports))
    )

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    diagnostics: dict[str, Any] = {
        "center_reports": center_reports,
        "n_local_real_fields": int(chol.shape[0]),
        "n_local_complex_fields": 0,
        "local_block_norm": local_block_norm,
        "extracted_local_block_norm": extracted_local_block_norm,
        "residual_center_block_norm": residual_center_block_norm,
        "residual_norm": residual_norm,
        "local_block_fraction_full_norm": _frac(local_block_norm, full_norm),
        "local_block_fraction_full_weight": _frac(local_block_norm * local_block_norm, full_norm * full_norm),
        "extracted_local_block_fraction_full_weight": _frac(
            extracted_local_block_norm * extracted_local_block_norm,
            full_norm * full_norm,
        ),
    }
    return chol, factors, tuple(labels), eri_residual, diagnostics


def _stage_uhf_charge_spin_ham_input_from_fcidump(
    obj: StagedMfOrCc,
    *,
    fcidump: Union[str, Path, Dict[str, Any]],
    chol_cut: float,
    verbose: bool,
    real_field_centers: Any = None,
    block_diagonalize: bool = False,
    unrestricted_ham: bool = False,
) -> HamInput:
    """Build UHF charge/spin fields in UHF alpha/beta MO bases."""
    from pyscf import ao2mo

    scf_obj = obj.mf
    if scf_obj.kind != "uhf":
        raise ValueError("uhf_charge_spin requires a UHF reference with alpha/beta MO coefficients.")
    if int(scf_obj.afqmc_frozen) != 0:
        raise NotImplementedError("uhf_charge_spin currently supports only afqmc_frozen=0.")

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    coeff_alpha = np.asarray(scf_obj.mo_coeff[0])
    coeff_beta = np.asarray(scf_obj.mo_coeff[1])
    if coeff_alpha.shape != (norb, norb) or coeff_beta.shape != (norb, norb):
        raise ValueError(
            "UHF MO coefficient shapes must both match FCIDUMP NORB: "
            f"got {coeff_alpha.shape}, {coeff_beta.shape}, NORB={norb}."
        )

    h0 = float(ctx.get("ECORE", 0.0))
    h1_ao = np.asarray(ctx["H1"])
    h1_ao = 0.5 * (h1_ao + h1_ao.T.conj())
    h1_alpha = coeff_alpha.T.conj() @ h1_ao @ coeff_alpha
    h1_beta = coeff_beta.T.conj() @ h1_ao @ coeff_beta
    h1_unrestricted = np.stack([h1_alpha, h1_beta], axis=0)
    h1 = h1_unrestricted if unrestricted_ham else _block_diag2(h1_alpha, h1_beta)
    if block_diagonalize:
        fit_name = "uhf_charge_spin_blocks"
    elif unrestricted_ham:
        fit_name = "uhf_charge_spin_unrham"
    else:
        fit_name = "uhf_charge_spin"
    factorize_supermatrix = (
        _factorize_uhf_charge_spin_blocked_supermatrix
        if block_diagonalize
        else _factorize_uhf_charge_spin_supermatrix
    )

    eri_full = ao2mo.restore(1, np.asarray(ctx["H2"]), norb)
    real_centers = _normalize_real_field_centers(real_field_centers, norb=norb)
    eri_selected = np.zeros_like(eri_full)
    for orbitals in real_centers:
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        eri_selected[ix4] = eri_full[ix4]
    eri_residual = np.asarray(eri_full) - eri_selected

    local_chol = np.zeros((0, 2 * norb, 2 * norb), dtype=np.asarray(eri_full).dtype)
    local_factors = np.zeros((0,), dtype=np.complex128)
    local_labels: list[str] = []
    local_diagnostics: dict[str, float | int] = {}
    full_diagnostics: dict[str, float | int] = {}
    local_block_norm = float(np.linalg.norm(eri_selected.reshape(-1)))
    t0 = time.time()
    if real_centers:
        local_cs, local_blocks = _uhf_charge_spin_supermatrix_from_eri(
            eri_selected,
            coeff_alpha=coeff_alpha,
            coeff_beta=coeff_beta,
        )
        local_chol, local_factors, local_labels, local_diagnostics = (
            factorize_supermatrix(
                local_cs,
                norb=norb,
                chol_cut=chol_cut,
                label_prefix="uhf_charge_spin_local",
            )
        )
    else:
        local_blocks = {
            "alpha_beta_pair_norm": 0.0,
            "charge_charge_pair_norm": 0.0,
            "charge_spin_pair_norm": 0.0,
            "spin_charge_pair_norm": 0.0,
            "spin_spin_pair_norm": 0.0,
        }

    if real_centers:
        decomposition_scope = "selected_plus_residual"
        residual_cs, residual_blocks = _uhf_charge_spin_supermatrix_from_eri(
            eri_residual,
            coeff_alpha=coeff_alpha,
            coeff_beta=coeff_beta,
        )
        residual_chol, residual_factors, residual_labels, residual_diagnostics = (
            factorize_supermatrix(
                residual_cs,
                norb=norb,
                chol_cut=chol_cut,
                label_prefix="uhf_charge_spin_residual",
            )
        )
        if local_chol.shape[0]:
            chol = np.concatenate([local_chol, residual_chol], axis=0)
            field_factors = np.concatenate([local_factors, residual_factors], axis=0)
        else:
            chol = residual_chol
            field_factors = residual_factors
        labels = tuple(local_labels + residual_labels)
        full_blocks = {
            "alpha_beta_pair_norm": 0.0,
            "charge_charge_pair_norm": 0.0,
            "charge_spin_pair_norm": 0.0,
            "spin_charge_pair_norm": 0.0,
            "spin_spin_pair_norm": 0.0,
        }
    else:
        decomposition_scope = "full"
        full_cs, full_blocks = _uhf_charge_spin_supermatrix_from_eri(
            eri_full,
            coeff_alpha=coeff_alpha,
            coeff_beta=coeff_beta,
        )
        full_chol, full_factors, full_labels, full_diagnostics = (
            factorize_supermatrix(
                full_cs,
                norb=norb,
                chol_cut=chol_cut,
                label_prefix="uhf_charge_spin_full",
            )
        )
        residual_blocks = {
            "alpha_beta_pair_norm": 0.0,
            "charge_charge_pair_norm": 0.0,
            "charge_spin_pair_norm": 0.0,
            "spin_charge_pair_norm": 0.0,
            "spin_spin_pair_norm": 0.0,
        }
        residual_diagnostics = {
            "uhf_charge_spin_residual_pair_norm": 0.0,
            "uhf_charge_spin_residual_pair_retained_norm": 0.0,
            "uhf_charge_spin_residual_pair_reconstruction_error_norm": 0.0,
            "uhf_charge_spin_residual_pair_reconstruction_relative_error": 0.0,
            "uhf_charge_spin_residual_pair_positive_eigenvalues": 0,
            "uhf_charge_spin_residual_pair_negative_eigenvalues": 0,
            "uhf_charge_spin_residual_pair_discarded_eigenvalues": 0,
        }
        chol = full_chol
        field_factors = full_factors
        labels = tuple(full_labels)

    nelec_tot = int(ctx["NELEC"])
    ms2 = int(ctx.get("MS2", 0))
    nelec: Tuple[int, int] = ((nelec_tot + ms2) // 2, (nelec_tot - ms2) // 2)
    full_norm = float(np.linalg.norm(np.asarray(eri_full).reshape(-1)))
    residual_norm = float(np.linalg.norm(eri_residual.reshape(-1)))

    def _count(prefix: str, needle: str) -> int:
        return int(sum(label.startswith(prefix) and f":{needle}_" in label for label in labels))

    field_metadata: Dict[str, Any] = {
        "real_field_fit": fit_name,
        "decomposition_variant": (
            "charge_spin_blocks_then_residual" if block_diagonalize else "full_diagonalization"
        ),
        "decomposition_scope": decomposition_scope,
        "centers": [list(center) for center in real_centers],
        "center_orbitals": list(sorted({orb for center in real_centers for orb in center})),
        "basis": "uhf_alpha_beta_mo_generalized",
        "full_real_fields": int(sum(label.startswith("uhf_charge_spin_full_real:") for label in labels)),
        "full_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_full_complex:") for label in labels)
        ),
        "n_full_real_fields": int(
            sum(label.startswith("uhf_charge_spin_full_real:") for label in labels)
        ),
        "n_full_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_full_complex:") for label in labels)
        ),
        "local_real_fields": int(sum(label.startswith("uhf_charge_spin_local_real:") for label in labels)),
        "local_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_local_complex:") for label in labels)
        ),
        "residual_real_fields": int(
            sum(label.startswith("uhf_charge_spin_residual_real:") for label in labels)
        ),
        "residual_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_residual_complex:") for label in labels)
        ),
        "n_local_real_fields": int(
            sum(label.startswith("uhf_charge_spin_local_real:") for label in labels)
        ),
        "n_local_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_local_complex:") for label in labels)
        ),
        "n_residual_real_fields": int(
            sum(label.startswith("uhf_charge_spin_residual_real:") for label in labels)
        ),
        "n_residual_complex_fields": int(
            sum(label.startswith("uhf_charge_spin_residual_complex:") for label in labels)
        ),
        "n_local_charge_dominant_fields": _count("uhf_charge_spin_local_", "charge"),
        "n_local_spin_dominant_fields": _count("uhf_charge_spin_local_", "spin"),
        "n_local_mixed_charge_spin_fields": _count("uhf_charge_spin_local_", "mixed"),
        "n_residual_charge_dominant_fields": _count("uhf_charge_spin_residual_", "charge"),
        "n_residual_spin_dominant_fields": _count("uhf_charge_spin_residual_", "spin"),
        "n_residual_mixed_charge_spin_fields": _count("uhf_charge_spin_residual_", "mixed"),
        "n_full_charge_dominant_fields": _count("uhf_charge_spin_full_", "charge"),
        "n_full_spin_dominant_fields": _count("uhf_charge_spin_full_", "spin"),
        "n_full_mixed_charge_spin_fields": _count("uhf_charge_spin_full_", "mixed"),
        "charge_spin_blocks": {
            "full": full_blocks,
            "local": local_blocks,
            "residual": residual_blocks,
        },
        "frobenius": {
            "full_norm": full_norm,
            "local_block_norm": local_block_norm,
            "extracted_local_block_norm": local_block_norm,
            "residual_norm": residual_norm,
            "center_block_norm": local_block_norm,
            "center_block_residual_norm": 0.0,
            "full_pair_norm": float(np.linalg.norm(ao2mo.restore(4, np.asarray(eri_full), norb))),
            **local_diagnostics,
            **residual_diagnostics,
            **full_diagnostics,
            "local_block_fraction_full_norm": (
                float(local_block_norm / full_norm) if full_norm > 0.0 else 0.0
            ),
            "local_block_fraction_full_weight": (
                float((local_block_norm * local_block_norm) / (full_norm * full_norm))
                if full_norm > 0.0
                else 0.0
            ),
            "extracted_local_block_fraction_full_weight": (
                float((local_block_norm * local_block_norm) / (full_norm * full_norm))
                if full_norm > 0.0
                else 0.0
            ),
            "residual_pair_reconstruction_relative_error": residual_diagnostics[
                "uhf_charge_spin_residual_pair_reconstruction_relative_error"
            ],
            "full_pair_reconstruction_relative_error": full_diagnostics.get(
                "uhf_charge_spin_full_pair_reconstruction_relative_error",
                0.0,
            ),
        },
        "field_labels": labels,
    }
    if verbose:
        if decomposition_scope == "full":
            print(
                f"[stage] FCIDUMP {fit_name} fit: "
                f"scope=full full_real={field_metadata['n_full_real_fields']} "
                f"full_complex={field_metadata['n_full_complex_fields']} "
                f"nchol={chol.shape[0]} in {time.time() - t0:.2f}s"
            )
        else:
            print(
                f"[stage] FCIDUMP {fit_name} fit: "
                f"scope=selected_plus_residual "
                f"local_real={field_metadata['n_local_real_fields']} "
                f"local_complex={field_metadata['n_local_complex_fields']} "
                f"residual_real={field_metadata['n_residual_real_fields']} "
                f"residual_complex={field_metadata['n_residual_complex_fields']} "
                f"nchol={chol.shape[0]} in {time.time() - t0:.2f}s"
            )

    if chol.shape[0] == 0:
        chol = np.zeros((0, 2 * norb, 2 * norb), dtype=np.asarray(eri_full).dtype)
    ham_basis: HamBasis = "unrestricted" if unrestricted_ham else "generalized"
    if unrestricted_ham:
        chol = _block_diag_chol_to_unrestricted(chol, norb)
        field_metadata["basis"] = "uhf_alpha_beta_mo_unrestricted"
    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=0,
        source_kind=obj.source,
        basis=ham_basis,
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=None,
        field_metadata=field_metadata,
    )


def _stage_uhf_local_real_then_charge_spin_unrham_from_fcidump(
    obj: StagedMfOrCc,
    *,
    fcidump: Union[str, Path, Dict[str, Any]],
    chol_cut: float,
    verbose: bool,
    real_field_centers: Any = None,
) -> HamInput:
    """Extract local real spin fields, then factorize the full residual in UHF charge/spin form."""
    from pyscf import ao2mo

    scf_obj = obj.mf
    if scf_obj.kind != "uhf":
        raise ValueError(
            "uhf_local_real_then_charge_spin_unrham requires a UHF reference with "
            "alpha/beta MO coefficients."
        )
    if int(scf_obj.afqmc_frozen) != 0:
        raise NotImplementedError(
            "uhf_local_real_then_charge_spin_unrham currently supports only afqmc_frozen=0."
        )

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    coeff_alpha = np.asarray(scf_obj.mo_coeff[0])
    coeff_beta = np.asarray(scf_obj.mo_coeff[1])
    if coeff_alpha.shape != (norb, norb) or coeff_beta.shape != (norb, norb):
        raise ValueError(
            "UHF MO coefficient shapes must both match FCIDUMP NORB: "
            f"got {coeff_alpha.shape}, {coeff_beta.shape}, NORB={norb}."
        )

    real_centers = _normalize_real_field_centers(real_field_centers, norb=norb)
    if not real_centers:
        raise ValueError(
            "uhf_local_real_then_charge_spin_unrham requires --real-field-centers; "
            "use uhf_charge_spin_unrham for a pure full-tensor residual decomposition."
        )

    t0 = time.time()
    h0 = float(ctx.get("ECORE", 0.0))
    h1_ao = np.asarray(ctx["H1"])
    h1_ao = 0.5 * (h1_ao + h1_ao.T.conj())
    h1 = np.stack(
        [
            coeff_alpha.T.conj() @ h1_ao @ coeff_alpha,
            coeff_beta.T.conj() @ h1_ao @ coeff_beta,
        ],
        axis=0,
    )
    eri_full = ao2mo.restore(1, np.asarray(ctx["H2"]), norb)

    local_chol, local_factors, local_labels, eri_residual, local_diagnostics = (
        _extract_uhf_local_real_spin_fields(
            eri_full,
            coeff_alpha=coeff_alpha,
            coeff_beta=coeff_beta,
            centers=real_centers,
            chol_cut=chol_cut,
        )
    )
    residual_cs, residual_blocks = _uhf_charge_spin_supermatrix_from_eri(
        eri_residual,
        coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta,
    )
    residual_chol_g, residual_factors, residual_labels, residual_diagnostics = (
        _factorize_uhf_charge_spin_supermatrix(
            residual_cs,
            norb=norb,
            chol_cut=chol_cut,
            label_prefix="uhf_charge_spin_residual",
        )
    )
    residual_chol = _block_diag_chol_to_unrestricted(residual_chol_g, norb)
    if local_chol.shape[0]:
        chol = np.concatenate([local_chol, residual_chol], axis=0)
        field_factors = np.concatenate([local_factors, residual_factors], axis=0)
    else:
        chol = residual_chol
        field_factors = residual_factors
    labels = tuple(local_labels + tuple(residual_labels))

    nelec_tot = int(ctx["NELEC"])
    ms2 = int(ctx.get("MS2", 0))
    nelec: Tuple[int, int] = ((nelec_tot + ms2) // 2, (nelec_tot - ms2) // 2)
    full_norm = float(np.linalg.norm(np.asarray(eri_full).reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, np.asarray(eri_full), norb)))

    def _count(prefix: str, needle: str) -> int:
        return int(sum(label.startswith(prefix) and f":{needle}_" in label for label in labels))

    n_local_real = int(local_diagnostics["n_local_real_fields"])
    n_residual_real = int(
        sum(label.startswith("uhf_charge_spin_residual_real:") for label in labels)
    )
    n_residual_complex = int(
        sum(label.startswith("uhf_charge_spin_residual_complex:") for label in labels)
    )
    field_metadata: Dict[str, Any] = {
        "real_field_fit": "uhf_local_real_then_charge_spin_unrham",
        "decomposition_variant": "local_real_then_full_charge_spin_residual",
        "decomposition_scope": "selected_local_real_plus_full_residual",
        "centers": [list(center) for center in real_centers],
        "center_orbitals": list(sorted({orb for center in real_centers for orb in center})),
        "basis": "uhf_alpha_beta_mo_unrestricted",
        "local_real_fields": n_local_real,
        "local_complex_fields": 0,
        "residual_real_fields": n_residual_real,
        "residual_complex_fields": n_residual_complex,
        "n_local_real_fields": n_local_real,
        "n_local_complex_fields": 0,
        "n_residual_real_fields": n_residual_real,
        "n_residual_complex_fields": n_residual_complex,
        "n_local_charge_dominant_fields": 0,
        "n_local_spin_dominant_fields": n_local_real,
        "n_local_mixed_charge_spin_fields": 0,
        "n_residual_charge_dominant_fields": _count("uhf_charge_spin_residual_", "charge"),
        "n_residual_spin_dominant_fields": _count("uhf_charge_spin_residual_", "spin"),
        "n_residual_mixed_charge_spin_fields": _count("uhf_charge_spin_residual_", "mixed"),
        "charge_spin_blocks": {
            "residual": residual_blocks,
        },
        "center_reports": local_diagnostics["center_reports"],
        "frobenius": {
            "full_norm": full_norm,
            "local_block_norm": local_diagnostics["local_block_norm"],
            "extracted_local_block_norm": local_diagnostics["extracted_local_block_norm"],
            "residual_norm": local_diagnostics["residual_norm"],
            "center_block_norm": local_diagnostics["local_block_norm"],
            "center_block_residual_norm": local_diagnostics["residual_center_block_norm"],
            "full_pair_norm": full_pair_norm,
            **residual_diagnostics,
            "residual_pair_reconstruction_relative_error": residual_diagnostics[
                "uhf_charge_spin_residual_pair_reconstruction_relative_error"
            ],
            "local_block_fraction_full_norm": local_diagnostics[
                "local_block_fraction_full_norm"
            ],
            "local_block_fraction_full_weight": local_diagnostics[
                "local_block_fraction_full_weight"
            ],
            "extracted_local_block_fraction_full_weight": local_diagnostics[
                "extracted_local_block_fraction_full_weight"
            ],
        },
        "field_labels": labels,
    }

    if verbose:
        print(
            "[stage] FCIDUMP uhf_local_real_then_charge_spin_unrham fit: "
            "scope=selected_local_real_plus_full_residual "
            f"local_real={n_local_real} residual_real={n_residual_real} "
            f"residual_complex={n_residual_complex} nchol={chol.shape[0]} "
            f"in {time.time() - t0:.2f}s"
        )

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=0,
        source_kind=obj.source,
        basis="unrestricted",
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=None,
        field_metadata=field_metadata,
    )


def _build_local_exact_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    eri_residual = np.array(eri_ao, copy=True)
    h1_shift_ao = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)

    local_chol: list[Array] = []
    local_factors: list[complex] = []
    local_spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    center_reports: list[dict[str, Any]] = []

    for center_idx, orbitals in enumerate(centers):
        ix4 = np.ix_(orbitals, orbitals, orbitals, orbitals)
        block = np.asarray(eri_ao_arr[ix4])
        local_residual_block = np.array(block, copy=True)
        full_block_pair = ao2mo.restore(4, block, len(orbitals))
        full_block_pair = 0.5 * (full_block_pair + full_block_pair.T.conj())

        center_start = len(local_chol)
        onsite_terms: list[dict[str, Any]] = []
        for local_orb, orb in enumerate(orbitals):
            u_value = float(np.real(local_residual_block[local_orb, local_orb, local_orb, local_orb]))
            if u_value <= float(chol_cut):
                continue
            global_mode = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)
            global_mode[orb, orb] = np.sqrt(u_value)
            local_chol.append(_rotate_one_body_to_mo(global_mode, basis_coeff))
            local_factors.append(1.0)
            local_spin_coeffs.append((1.0, -1.0))
            labels.append(f"local_exact_real:center{center_idx}:onsite{orb}")
            onsite_terms.append({"center": int(center_idx), "orbital": int(orb), "U": u_value})
            local_residual_block[local_orb, local_orb, local_orb, local_orb] -= u_value

        block_pair = ao2mo.restore(4, local_residual_block, len(orbitals))
        block_pair = 0.5 * (block_pair + block_pair.T.conj())
        eigvals, eigvecs = np.linalg.eigh(block_pair)
        keep = np.abs(eigvals) > float(chol_cut)

        for local_mode_idx, value in enumerate(eigvals[keep]):
            vec = eigvecs[:, keep][:, local_mode_idx]
            small = _unpack_pair_vector(vec * np.sqrt(abs(value)), len(orbitals))
            small = 0.5 * (small + small.T.conj())
            global_mode = np.zeros((norb, norb), dtype=eri_ao_arr.dtype)
            global_mode[np.ix_(orbitals, orbitals)] = small
            local_chol.append(_rotate_one_body_to_mo(global_mode, basis_coeff))

            spin_tol = max(float(chol_cut), 1.0e-12)
            if _mode_is_exact_real_spin_channel(small, float(np.real(value)), tol=spin_tol):
                local_factors.append(1.0)
                local_spin_coeffs.append((1.0, -1.0))
                labels.append(f"local_exact_real:center{center_idx}:mode{local_mode_idx}")
            else:
                local_factors.append(1.0j if value > 0.0 else 1.0)
                local_spin_coeffs.append((1.0, 1.0))
                labels.append(f"local_exact_complex:center{center_idx}:mode{local_mode_idx}")

        center_stop = len(local_chol)
        center_chol_ao = []
        for chol_mo in local_chol[center_start:center_stop]:
            # basis_coeff is unitary in the FCIDUMP local-orbital route; this inverse
            # rotation keeps diagnostics in the same local basis as the extracted block.
            center_chol_ao.append(np.asarray(basis_coeff) @ chol_mo @ np.asarray(basis_coeff).T.conj())
        center_chol_ao = np.asarray(center_chol_ao)
        if center_chol_ao.size:
            center_chol_ao = center_chol_ao[:, orbitals, :][:, :, orbitals]
        else:
            center_chol_ao = np.zeros((0, len(orbitals), len(orbitals)), dtype=eri_ao_arr.dtype)
        center_factors = np.asarray(local_factors[center_start:center_stop], dtype=np.complex128)
        center_spin_coeffs = np.asarray(
            local_spin_coeffs[center_start:center_stop], dtype=np.float64
        )
        reconstructed_pair = _reconstruct_packed_pair_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )
        reconstructed_spin_orbital = _reconstruct_spin_orbital_tensor_from_fields(
            center_chol_ao, center_factors, center_spin_coeffs
        )
        reference_spin_orbital = _spin_orbital_tensor_from_eri(block)

        residual_eigs = eigvals[~keep]
        block_norm = float(np.linalg.norm(block.reshape(-1)))
        kept_pair_norm = float(np.linalg.norm(eigvals[keep]))
        discarded_pair_norm = float(np.linalg.norm(residual_eigs))
        extracted_block_norm = float(
            np.linalg.norm(ao2mo.restore(1, np.asarray(reconstructed_pair), len(orbitals)))
        )
        center_reports.append(
            {
                "center": int(center_idx),
                "orbitals": [int(x) for x in orbitals],
                "onsite_real_terms": onsite_terms,
                "local_block_norm": block_norm,
                "extracted_local_block_norm": extracted_block_norm,
                "residual_center_block_norm": 0.0,
                "packed_pair_norm": float(np.linalg.norm(full_block_pair)),
                "onsite_subtracted_packed_pair_norm": float(np.linalg.norm(block_pair)),
                "retained_packed_pair_norm": kept_pair_norm,
                "discarded_packed_pair_norm": discarded_pair_norm,
                "packed_pair_reconstruction_relative_error": _relative_error(
                    full_block_pair, reconstructed_pair
                ),
                "spin_orbital_reconstruction_relative_error": _relative_error(
                    reference_spin_orbital, reconstructed_spin_orbital
                ),
                "n_local_real_fields": int(
                    sum(
                        label.startswith(f"local_exact_real:center{center_idx}:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "n_local_complex_fields": int(
                    sum(
                        label.startswith(f"local_exact_complex:center{center_idx}:")
                        for label in labels[center_start:center_stop]
                    )
                ),
                "parameters": _local_parameter_diagnostics(block, orbitals),
            }
        )

        eri_residual[ix4] = 0.0

    residual_chol, residual_factors, residual_spin_coeffs, residual_labels, residual_diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_residual,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )

    if local_chol:
        chol = np.concatenate([np.asarray(local_chol), residual_chol], axis=0)
        field_factors = np.concatenate(
            [np.asarray(local_factors, dtype=np.complex128), residual_factors], axis=0
        )
        field_spin_coeffs = np.concatenate(
            [np.asarray(local_spin_coeffs, dtype=np.float64), residual_spin_coeffs], axis=0
        )
    else:
        chol = residual_chol
        field_factors = residual_factors
        field_spin_coeffs = residual_spin_coeffs
    labels.extend(residual_labels)

    n_local_real = sum(label.startswith("local_exact_real:") for label in labels)
    n_local_complex = sum(label.startswith("local_exact_complex:") for label in labels)
    n_residual_complex = sum(label == "residual_complex" for label in residual_labels)
    n_residual_real = sum(label == "residual_real" for label in residual_labels)
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    residual_frobenius_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))
    local_block_norm = float(
        np.sqrt(sum(report["local_block_norm"] ** 2 for report in center_reports))
    )
    extracted_local_block_norm = float(
        np.sqrt(sum(report["extracted_local_block_norm"] ** 2 for report in center_reports))
    )

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    metadata: Dict[str, Any] = {
        "real_field_fit": "local_exact",
        "centers": [list(center) for center in centers],
        "center_orbitals": list(sorted({orb for center in centers for orb in center})),
        "center_reports": center_reports,
        "local_real_fields": int(n_local_real),
        "local_complex_fields": int(n_local_complex),
        "residual_real_fields": int(n_residual_real),
        "residual_complex_fields": int(n_residual_complex),
        "n_local_real_fields": int(n_local_real),
        "n_local_complex_fields": int(n_local_complex),
        "n_residual_real_fields": int(n_residual_real),
        "n_residual_complex_fields": int(n_residual_complex),
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "local_block_norm": local_block_norm,
            "extracted_local_block_norm": extracted_local_block_norm,
            "residual_norm": residual_frobenius_norm,
            "center_block_norm": local_block_norm,
            "center_block_residual_norm": 0.0,
            "full_pair_norm": full_pair_norm,
            **residual_diagnostics,
            "local_block_fraction_full_norm": _frac(local_block_norm, full_frobenius_norm),
            "local_block_fraction_full_weight": _frac(
                local_block_norm * local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
            "extracted_local_block_fraction_full_weight": _frac(
                extracted_local_block_norm * extracted_local_block_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
        },
        "field_labels": tuple(labels),
    }

    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=eri_ao_arr.dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _build_hk_density_real_fields_from_eri(
    eri_ao: Array,
    *,
    basis_coeff: Array,
    centers: tuple[tuple[int, ...], ...],
    chol_cut: float,
) -> RealFieldFitResult:
    from pyscf import ao2mo

    norb = int(eri_ao.shape[0])
    nmo = int(np.asarray(basis_coeff).shape[1])
    eri_ao_arr = np.asarray(eri_ao)
    eri_residual = np.array(eri_ao, copy=True)
    h1_shift_ao = np.zeros((norb, norb), dtype=np.asarray(eri_ao).dtype)
    center_orbitals = tuple(sorted({orb for center in centers for orb in center}))
    if center_orbitals:
        center_ix = np.ix_(center_orbitals, center_orbitals, center_orbitals, center_orbitals)
        center_block_norm = float(np.linalg.norm(eri_ao_arr[center_ix].reshape(-1)))
    else:
        center_block_norm = 0.0

    real_chol: list[Array] = []
    real_factors: list[complex] = []
    real_spin_coeffs: list[tuple[float, float]] = []
    labels: list[str] = []
    extracted: list[dict[str, Any]] = []

    for center_idx, orbitals in enumerate(centers):
        for orb in orbitals:
            u_value = float(np.real(eri_residual[orb, orb, orb, orb]))
            if u_value <= float(chol_cut):
                continue
            local = np.zeros((norb, norb), dtype=np.asarray(eri_ao).dtype)
            local[orb, orb] = np.sqrt(u_value)
            real_chol.append(_rotate_one_body_to_mo(local, basis_coeff))
            real_factors.append(1.0)
            real_spin_coeffs.append((1.0, -1.0))
            labels.append(f"hk_density_real:center{center_idx}:orb{orb}")
            extracted.append({"center": center_idx, "orbital": int(orb), "U": u_value})

            # The normal-ordering correction for -0.5 [sqrt(U)(n_up - n_down)]^2
            # is applied by the propagation/measurement h1_eff builders.
            eri_residual[orb, orb, orb, orb] -= u_value

    residual_chol, residual_factors, residual_spin_coeffs, residual_labels, residual_diagnostics = (
        _factorize_symmetric_supermatrix(
            eri_residual,
            coeff=basis_coeff,
            chol_cut=chol_cut,
        )
    )

    if real_chol:
        chol = np.concatenate([np.asarray(real_chol), residual_chol], axis=0)
        field_factors = np.concatenate(
            [np.asarray(real_factors, dtype=np.complex128), residual_factors], axis=0
        )
        field_spin_coeffs = np.concatenate(
            [np.asarray(real_spin_coeffs, dtype=np.float64), residual_spin_coeffs], axis=0
        )
    else:
        chol = residual_chol
        field_factors = residual_factors
        field_spin_coeffs = residual_spin_coeffs
    labels.extend(residual_labels)

    n_hk = len(real_chol)
    n_residual_complex = sum(label == "residual_complex" for label in residual_labels)
    n_residual_real = sum(label == "residual_real" for label in residual_labels)
    hk_frobenius_norm = float(np.sqrt(sum(term["U"] * term["U"] for term in extracted)))
    full_frobenius_norm = float(np.linalg.norm(eri_ao_arr.reshape(-1)))
    residual_frobenius_norm = float(np.linalg.norm(eri_residual.reshape(-1)))
    full_pair_norm = float(np.linalg.norm(ao2mo.restore(4, eri_ao_arr, norb)))
    if center_orbitals:
        residual_center_block_norm = float(np.linalg.norm(eri_residual[center_ix].reshape(-1)))
    else:
        residual_center_block_norm = 0.0

    def _frac(num: float, den: float) -> float:
        return float(num / den) if den > 0.0 else 0.0

    metadata: Dict[str, Any] = {
        "real_field_fit": "hk_density",
        "centers": [list(center) for center in centers],
        "center_orbitals": list(center_orbitals),
        "extracted_terms": extracted,
        "n_hk_real_fields": int(n_hk),
        "n_residual_complex_fields": int(n_residual_complex),
        "n_residual_real_fields": int(n_residual_real),
        "frobenius": {
            "full_norm": full_frobenius_norm,
            "hk_onsite_norm": hk_frobenius_norm,
            "residual_norm": residual_frobenius_norm,
            "center_block_norm": center_block_norm,
            "center_block_residual_norm": residual_center_block_norm,
            "full_pair_norm": full_pair_norm,
            "hk_onsite_pair_norm": hk_frobenius_norm,
            **residual_diagnostics,
            "hk_fraction_full_norm": _frac(hk_frobenius_norm, full_frobenius_norm),
            "hk_fraction_full_weight": _frac(
                hk_frobenius_norm * hk_frobenius_norm,
                full_frobenius_norm * full_frobenius_norm,
            ),
            "hk_fraction_full_pair_weight": _frac(
                hk_frobenius_norm * hk_frobenius_norm,
                full_pair_norm * full_pair_norm,
            ),
            "hk_fraction_center_block_norm": _frac(hk_frobenius_norm, center_block_norm),
            "hk_fraction_center_block_weight": _frac(
                hk_frobenius_norm * hk_frobenius_norm,
                center_block_norm * center_block_norm,
            ),
        },
        "field_labels": tuple(labels),
    }

    if chol.shape[0] == 0:
        chol = np.zeros((0, nmo, nmo), dtype=np.asarray(eri_ao).dtype)
    return RealFieldFitResult(
        h1_shift=_rotate_one_body_to_mo(h1_shift_ao, basis_coeff),
        chol=np.asarray(chol),
        field_factors=np.asarray(field_factors, dtype=np.complex128),
        field_spin_coeffs=np.asarray(field_spin_coeffs, dtype=np.float64),
        metadata=metadata,
    )


def _stage_ham_input_from_fcidump(
    obj: StagedMfOrCc,
    *,
    fcidump: Union[str, Path, Dict[str, Any]],
    chol_cut: float,
    verbose: bool,
    real_field_centers: Any = None,
    real_field_method: str = "hk_density",
) -> HamInput:
    """
    Build HamInput from FCIDUMP integrals while preserving the trial MO basis convention.
    """
    from pyscf import ao2mo

    scf_obj = obj.mf
    if scf_obj.kind == "ghf":
        raise NotImplementedError("FCIDUMP staging for stage_from_ccpy does not support GHF.")
    if real_field_method == "uhf_local_real_then_charge_spin_unrham":
        return _stage_uhf_local_real_then_charge_spin_unrham_from_fcidump(
            obj,
            fcidump=fcidump,
            chol_cut=chol_cut,
            verbose=verbose,
            real_field_centers=real_field_centers,
        )
    if real_field_method in {"uhf_charge_spin", "uhf_charge_spin_blocks", "uhf_charge_spin_unrham"}:
        return _stage_uhf_charge_spin_ham_input_from_fcidump(
            obj,
            fcidump=fcidump,
            chol_cut=chol_cut,
            verbose=verbose,
            real_field_centers=real_field_centers,
            block_diagonalize=(real_field_method == "uhf_charge_spin_blocks"),
            unrestricted_ham=(real_field_method == "uhf_charge_spin_unrham"),
        )

    match scf_obj.kind:
        case "rhf" | "rohf":
            basis_coeff = np.asarray(scf_obj.mo_coeff)
        case "uhf":
            basis_coeff = np.asarray(scf_obj.mo_coeff[0])
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    if basis_coeff.shape[1] != norb:
        raise ValueError(
            "FCIDUMP NORB does not match mf orbital count: " f"{norb} != {basis_coeff.shape[1]}."
        )

    h0 = float(ctx.get("ECORE", 0.0))
    h1_ao = np.asarray(ctx["H1"])
    if h1_ao.shape != (norb, norb):
        raise ValueError(f"FCIDUMP H1 must have shape ({norb}, {norb}), got {h1_ao.shape}.")
    h1_ao = 0.5 * (h1_ao + h1_ao.T.conj())
    h1 = basis_coeff.T.conj() @ h1_ao @ basis_coeff
    h1 = np.asarray(h1)

    h2_raw = np.asarray(ctx["H2"])
    eri_ao = ao2mo.restore(1, h2_raw, norb)

    nelec_tot = int(ctx["NELEC"])
    ms2 = int(ctx.get("MS2", 0))
    if (nelec_tot + ms2) % 2 != 0 or (nelec_tot - ms2) % 2 != 0:
        raise ValueError(f"Inconsistent FCIDUMP NELEC/MS2 pair: NELEC={nelec_tot}, MS2={ms2}.")
    nelec: Tuple[int, int] = ((nelec_tot + ms2) // 2, (nelec_tot - ms2) // 2)

    norb_frozen = scf_obj.afqmc_frozen
    assert isinstance(norb_frozen, int)
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")
    if norb_frozen >= norb:
        raise ValueError(f"norb_frozen={norb_frozen} leaves no active orbitals (norb={norb}).")
    if norb_frozen > 0:
        raise NotImplementedError(
            "FCIDUMP staging currently supports only norb_frozen=0. "
            "Use stage_from_ccpy(..., fcidump=...) which falls back to the existing MF frozen-core path."
        )

    field_factors: Array | None = None
    field_spin_coeffs: Array | None = None
    field_metadata: Dict[str, Any] | None = None

    t0 = time.time()
    real_centers = _normalize_real_field_centers(real_field_centers, norb=norb)
    if real_centers or real_field_method == "charge_spin":
        match real_field_method:
            case "hk_density":
                fit = _build_hk_density_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case "local_exact":
                fit = _build_local_exact_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case "kanamori_sign":
                fit = _build_kanamori_sign_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case "kanamori_real":
                fit = _build_kanamori_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case "kanamori_uj":
                fit = _build_kanamori_uj_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case "charge_spin":
                if real_centers:
                    fit = _build_charge_spin_real_fields_from_eri(
                        eri_ao,
                        basis_coeff=basis_coeff,
                        centers=real_centers,
                        chol_cut=chol_cut,
                    )
                else:
                    fit = _build_full_charge_spin_real_fields_from_eri(
                        eri_ao,
                        basis_coeff=basis_coeff,
                        chol_cut=chol_cut,
                    )
            case "kanamori_sign_full":
                fit = _build_kanamori_sign_full_real_fields_from_eri(
                    eri_ao,
                    basis_coeff=basis_coeff,
                    centers=real_centers,
                    chol_cut=chol_cut,
                )
            case _:
                raise ValueError(
                    "real_field_method must be one of "
                    "{'hk_density', 'local_exact', 'kanamori_sign', "
                    "'kanamori_real', 'kanamori_uj', 'charge_spin', 'uhf_charge_spin', "
                    "'uhf_charge_spin_blocks', 'uhf_charge_spin_unrham', "
                    "'uhf_local_real_then_charge_spin_unrham', 'kanamori_sign_full'}, "
                    f"got {real_field_method!r}."
                )
        h1 = h1 + fit.h1_shift
        chol = fit.chol
        field_factors = fit.field_factors
        field_spin_coeffs = fit.field_spin_coeffs
        field_metadata = fit.metadata
    else:
        eri_mo = np.einsum(
            "pi,qj,rk,sl,pqrs->ijkl",
            basis_coeff.conj(),
            basis_coeff.conj(),
            basis_coeff,
            basis_coeff,
            eri_ao,
            optimize=True,
        )
        eri_s4 = ao2mo.restore(4, np.asarray(eri_mo), norb)
        eri_s4 = 0.5 * (eri_s4 + eri_s4.T.conj())
        min_pair_eig = float(np.linalg.eigvalsh(eri_s4)[0])
        if min_pair_eig < -10.0 * float(chol_cut):
            raise ValueError(
                "Ordinary Cholesky staging requires a positive-semidefinite two-body "
                "packed-pair matrix, but this FCIDUMP/model is indefinite "
                f"(min eigenvalue {min_pair_eig:.6e}). Use a PSD model such as "
                "--model-h2 onsite or --model-h2 selected_full for --vanilla-afqmc, "
                "or use a real-field route with residual eigendecomposition instead."
            )
        chol = modified_cholesky(eri_s4, max_error=chol_cut)
    if verbose:
        if field_metadata is not None:
            print(
                "[stage] FCIDUMP real-field fit: "
                f"method={field_metadata['real_field_fit']} "
                f"local_real={field_metadata.get('n_local_real_fields', field_metadata.get('n_hk_real_fields', 0))} "
                f"local_complex={field_metadata.get('n_local_complex_fields', 0)} "
                f"residual_complex={field_metadata['n_residual_complex_fields']} "
                f"residual_real={field_metadata['n_residual_real_fields']} "
                f"nchol={chol.shape[0]} in {time.time() - t0:.2f}s"
            )
        else:
            print(f"[stage] FCIDUMP cholesky: nchol={chol.shape[0]} in {time.time() - t0:.2f}s")

    return HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=obj.source,
        basis="restricted",
        field_factors=field_factors,
        field_spin_coeffs=field_spin_coeffs,
        field_metadata=field_metadata,
    )


def _stage_trial_input(obj: StagedMfOrCc) -> TrialInput:
    """
    Produce TrialInput consistent with the Hamiltonian basis and frozen core choice
    """

    match obj.kind:
        case "rhf" | "rohf" | "uhf" | "ghf":
            stage_tr_fun = _stage_mf_input
        case "ccsd":
            stage_tr_fun = _stage_cisd_input
        case "uccsd":
            stage_tr_fun = _stage_ucisd_input
        case "gccsd":
            stage_tr_fun = _stage_gcisd_input
        case "pt2ccsd":
            stage_tr_fun = _stage_pt2ccsd_input
        case _:
            raise ValueError(f"Unreachable: '{obj.kind}'.")

    return stage_tr_fun(obj)


def _active_orbital_indices(norb: int, frozen: int | NDArray) -> NDArray:
    if isinstance(frozen, int):
        return np.arange(frozen, norb, dtype=np.int64)
    if isinstance(frozen, np.ndarray):
        return np.delete(np.arange(norb, dtype=np.int64), frozen)
    raise TypeError(f"frozen must be an integer or a np.ndarray, but received '{type(frozen)}'.")


def _apply_frozen_mask(vec: NDArray, frozen: int | NDArray) -> NDArray:
    return vec[_active_orbital_indices(vec.shape[0], frozen)]


def _stage_mf_input(obj: StagedMfOrCc) -> TrialInput:

    mol = obj.mol
    S = obj.get_ovlp(mol)
    frozen = obj.afqmc_frozen

    match obj.mf.kind:
        case "rhf" | "ghf":
            Ca = np.asarray(obj.mo_coeff)
            mo = _mf_coeff_helper(Ca, Ca, S, frozen)
            data = {"mo": np.asarray(mo)}

        case "rohf":
            Ca = np.asarray(obj.mo_coeff)
            mo = _mf_coeff_helper(Ca, Ca, S, frozen)
            mo_occ = _apply_frozen_mask(np.asarray(obj.mf.mo_occ), frozen)
            data = {
                "mo_a": np.asarray(mo[:, mo_occ > 0.0]),
                "mo_b": np.asarray(mo[:, mo_occ > 1.0]),
            }

        case "uhf":
            Ca = np.asarray(obj.mo_coeff[0])
            Cb = np.asarray(obj.mo_coeff[1])

            # basis is alpha MOs, represent alpha and beta orbitals in this basis
            moa = _mf_coeff_helper(Ca, Ca, S, frozen)
            mob = _mf_coeff_helper(Ca, Cb, S, frozen)
            data = {"mo_a": np.asarray(moa), "mo_b": np.asarray(mob)}
        case _:
            raise ValueError(f"Unreachable: '{obj.kind}'.")

    return TrialInput(
        kind=obj.kind,
        data=data,
        frozen=frozen,
        source_kind=obj.source,
    )


def _mf_coeff_helper(
    Ca: NDArray,
    Cb: NDArray,
    S: NDArray,
    frozen: int | NDArray,
) -> NDArray:
    q, r = np.linalg.qr(Ca.T @ S @ Cb)
    sgn = np.sign(np.diag(r))
    q = q * sgn[None, :]
    idx = _active_orbital_indices(len(q), frozen)
    return q[np.ix_(idx, idx)]


def _stage_cisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "ccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1_arr = np.asarray(obj.t1)
    t2_arr = np.asarray(obj.t2)
    nocc_t_core = 0
    nvir_t_outer = 0

    if isinstance(obj.trial_frozen, (np.ndarray)):
        if obj.mol.nelec[0] != obj.mol.nelec[1]:
            raise ValueError(
                "List-valued cc.frozen is currently supported only for closed-shell restricted CCSD."
            )

        assert isinstance(obj.afqmc_frozen, int)

        nocc_t_core, nvir_t_outer = _infer_restricted_trial_freeze_from_cc(
            cc_frozen=obj.trial_frozen,
            nmo_full=_mo_coeff_nmo(obj.mf.mo_coeff),
            nocc_full=int(obj.mol.nelectron // 2),
            norb_frozen=int(obj.afqmc_frozen),
            t1_shape=(int(t1_arr.shape[0]), int(t1_arr.shape[1])),
        )

    ci2 = t2_arr + np.einsum("ia,jb->ijab", t1_arr, t1_arr)
    ci2 = ci2.transpose(0, 2, 1, 3)  # (i,a,j,b) -> (i,j,a,b)
    ci1 = t1_arr

    data = {
        "ci1": ci1,
        "ci2": ci2,
        "nocc_t_core": np.array(nocc_t_core, dtype=np.int64),
        "nvir_t_outer": np.array(nvir_t_outer, dtype=np.int64),
    }
    return TrialInput(
        kind="cisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_ucisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "uccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1a, t1b = obj.t1
    t2aa, t2ab, t2bb = obj.t2

    ci2aa = np.asarray(t2aa) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1a))
    ci2aa = 0.5 * (ci2aa - ci2aa.transpose(0, 1, 3, 2))
    ci2aa = ci2aa.transpose(0, 2, 1, 3)

    ci2bb = np.asarray(t2bb) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1b), np.asarray(t1b))
    ci2bb = 0.5 * (ci2bb - ci2bb.transpose(0, 1, 3, 2))
    ci2bb = ci2bb.transpose(0, 2, 1, 3)

    ci2ab = np.asarray(t2ab) + np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1b))
    ci2ab = ci2ab.transpose(0, 2, 1, 3)

    _uhf_input = _stage_mf_input(obj)
    moa = _uhf_input.data["mo_a"]
    mob = _uhf_input.data["mo_b"]

    data = {
        "mo_coeff_a": np.asarray(moa),
        "mo_coeff_b": np.asarray(mob),
        "ci1a": np.asarray(t1a),
        "ci1b": np.asarray(t1b),
        "ci2aa": np.asarray(ci2aa),
        "ci2ab": np.asarray(ci2ab),
        "ci2bb": np.asarray(ci2bb),
    }

    return TrialInput(
        kind="ucisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_gcisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "gccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2

    ci2 = (
        np.einsum("ijab->iajb", t2)
        + np.einsum("ia,jb->iajb", t1, t1)
        - np.einsum("ib,ja->iajb", t1, t1)
    )
    ci1 = np.asarray(t1)

    _ghf_input = _stage_mf_input(obj)
    mo = _ghf_input.data["mo"]

    data = {"mo_coeff": mo, "ci1": ci1, "ci2": ci2}

    return TrialInput(
        kind="gcisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_pt2ccsd_input(obj):
    # TODO obj.kind is frozen... figure out how to assign more flexible trial
    # if obj.kind != "pt2ccsd":
    #     raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2
    nocc, nvir = t1.shape
    norb = nocc + nvir

    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t2 = t2.transpose(0, 2, 1, 3)  # (i,j,a,b) -> (i,a,j,b)

    def _thouless(init_slater, t1):
        # Thouless transformation: |psi'> = exp(t1_ia a+ i)|psi>
        # init slater: mo_coeff of psi (in mo basis)
        # return mo_coeff of psi' (in mo basis)
        norb, nvir = t1.shape
        norb = nocc + nvir
        exp_t1 = np.eye(norb, dtype=np.float64)
        exp_t1[:nocc, nocc:] = t1
        # exp_t1 = jsp.linalg.expm(t1_full)
        return exp_t1.T @ init_slater

    mo_coeff = np.eye(norb, dtype=np.float64)[:, :nocc]
    mo_t = _thouless(mo_coeff, t1)

    data = {"mo_t": mo_t, "t2": t2}
    return TrialInput(
        kind="pt2ccsd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


# ---------------------------------------------------------------------------
# ccpy interface helpers
# ---------------------------------------------------------------------------


def _ccpy_t_to_c_amplitudes(driver: Any, order: int, order_cc: int) -> dict:
    from .staging_ccpy import _ccpy_t_to_c_amplitudes as _impl

    return _impl(driver, order=order, order_cc=order_cc)


def stage_from_ccpy(
    driver: Any,
    mf: Any,
    *,
    order: int = -1,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    fcidump: Union[str, Path, Dict[str, Any], None] = None,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    real_field_centers: Any = None,
    real_field_method: str = "hk_density",
) -> StagedInputs:
    from .staging_ccpy import stage_from_ccpy as _impl

    return _impl(
        driver,
        mf,
        order=order,
        norb_frozen_core=norb_frozen_core,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        fcidump=fcidump,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
        real_field_centers=real_field_centers,
        real_field_method=real_field_method,
    )


def _dump_h5(staged: StagedInputs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(staged.ham.h0))
        gham.create_dataset("h1", data=staged.ham.h1)
        gham.create_dataset("chol", data=staged.ham.chol)
        if staged.ham.field_factors is not None:
            gham.create_dataset("field_factors", data=staged.ham.field_factors)
        if staged.ham.field_spin_coeffs is not None:
            gham.create_dataset("field_spin_coeffs", data=staged.ham.field_spin_coeffs)
        gham.create_dataset("nelec", data=np.array(staged.ham.nelec, dtype=np.int64))
        gham.attrs["norb"] = staged.ham.norb
        gham.attrs["chol_cut"] = staged.ham.chol_cut
        _dump_frozen(gham, staged.ham.frozen)
        gham.attrs["source_kind"] = staged.ham.source_kind
        gham.attrs["basis"] = staged.ham.basis

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        _dump_frozen(gtr, staged.trial.frozen)
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))


def _to_json_str(x: Any) -> str:
    # np scalar -> python scalar
    if isinstance(x, np.ndarray):
        x = x.item()
    # bytes like -> decode
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return bytes(x).decode("utf-8")
    return str(x)


def _load_h5(path: Path) -> StagedInputs:
    with h5py.File(path, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))
        if "frozen" in meta:
            meta["frozen"] = _freeze_from_meta_value(meta["frozen"])

        t_ham = _stage_begin("reading Hamiltonian from cache")
        gham: Any = f["ham"]
        ham = HamInput(
            h0=float(np.array(gham["h0"]).item()),
            h1=np.array(gham["h1"]),
            chol=np.array(gham["chol"]),
            nelec=(int(np.array(gham["nelec"])[0]), int(np.array(gham["nelec"])[1])),
            norb=int(gham.attrs["norb"]),
            chol_cut=float(gham.attrs["chol_cut"]),
            frozen=_load_frozen(gham),
            source_kind=str(gham.attrs["source_kind"]),
            basis=cast(HamBasis, str(gham.attrs["basis"])),
            field_factors=(np.array(gham["field_factors"]) if "field_factors" in gham else None),
            field_spin_coeffs=(
                np.array(gham["field_spin_coeffs"]) if "field_spin_coeffs" in gham else None
            ),
            field_metadata=meta.get("field_metadata"),
        )
        _stage_end(
            t_ham, "Hamiltonian loaded", details=f"norb={ham.norb} nchol={ham.chol.shape[0]}"
        )

        t_trial = _stage_begin("reading trial input from cache")
        gtr: Any = f["trial"]
        gdata = gtr["data"]
        trial_data = {k: np.array(gdata[k]) for k in gdata.keys()}
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data=trial_data,
            frozen=_load_frozen(gtr),
            source_kind=str(gtr.attrs["source_kind"]),
        )
        _stage_end(t_trial, "trial input loaded", details=f"kind={trial.kind}")

        return StagedInputs(ham=ham, trial=trial, meta=meta)


def build_ham_lno(
    obj: Any,
    *,
    frozen_orbitals: ArrayLike,
    chol_cut: float,
) -> HamInput:
    from pyscf import ao2mo, mcscf

    obj = StagedMfOrCc(obj, 0)
    mf = obj.mf.mf
    mol = mf.mol

    norb = obj.norb
    frozen = _normalize_frozen_list(frozen_orbitals, nmo=norb)
    basis_coeff = mf.mo_coeff

    nelec_frozen = 2 * np.sum(frozen < mol.nelec[0])
    nact = basis_coeff.shape[1] - frozen.size
    nelec_act = mol.nelectron - nelec_frozen
    mc = mcscf.CASSCF(mf, nact, nelec_act)
    mc.frozen = frozen  # type: ignore
    nelec = mc.nelecas  # type: ignore
    h1, h0 = mc.get_h1eff()  # type: ignore
    act = _active_orbital_indices(norb, frozen)
    e = np.asarray(ao2mo.kernel(mf.mol, mf.mo_coeff[:, act]))  # , compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)
    chol = chol.reshape((-1, nact, nact))

    ham = HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=nact,
        chol_cut=float(chol_cut),
        frozen=frozen,
        source_kind=obj.source,
        basis="restricted",
    )

    return ham
