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
    h1: Array  # (norb, norb)
    chol: Array  # (nchol, norb, norb)
    nelec: Tuple[int, int]
    norb: int
    chol_cut: float
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"
    basis: HamBasis  # "restricted" or "generalized"
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


def _build_kanamori_sign_full_real_fields_from_eri(
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

    def candidate_tensor(mode: Array, channel: str) -> Array:
        factor, spin_coeff, _ = _kanamori_field_coeffs(1.0, channel=channel)
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
                for a, b in ((pp_idx, qq_idx), (qq_idx, pp_idx), (pq_idx, pq_idx)):
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
                candidates = [
                    ("density_plus_charge", plus, "charge"),
                    ("density_minus_charge", minus, "charge"),
                    ("density_plus_spin", plus, "spin"),
                    ("density_minus_spin", minus, "spin"),
                    ("hund_preferred", bond, preferred_bond),
                    ("hund_other", bond, other_bond),
                ]
                matrix = np.stack(
                    [candidate_tensor(mode, channel).reshape(-1).real for _, mode, channel in candidates],
                    axis=1,
                )
                rhs = target_spin.reshape(-1).real
                coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
                spin_relerr = _relative_error(rhs, matrix @ coeffs)
                if spin_relerr > max(10.0 * float(chol_cut), 1.0e-10):
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
                    accepted_terms.append({"component": name, "coefficient": coeff, "channel": channel})
                residual_pair -= pair_extracted
                extracted_pair += pair_extracted
                kanamori_terms.append(
                    {
                        "kind": "pair_full",
                        "center": int(center_idx),
                        "orbitals": [int(orbitals[p]), int(orbitals[q])],
                        "preferred_decomposition": preferred_bond,
                        "decomposition": "spin_orbital_lstsq",
                        "spin_orbital_fit_relative_error": spin_relerr,
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
        "real_field_fit": "kanamori_sign_full",
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
    if real_centers:
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
                    "{'hk_density', 'local_exact', 'kanamori_sign', 'kanamori_sign_full'}, "
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
