from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

HamBasis = Literal["restricted", "unrestricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="unrestricted":
      h1:   (2, norb, norb)
      chol: (n_fields, 2, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)

    field_factors:
      Optional HS propagation coefficient per field. If omitted, every field uses
      the standard molecular convention, factor = 1j.

    field_spin_coeffs:
      Optional spin coupling coefficients with shape (n_fields, 2). The two
      columns multiply the alpha and beta one-body operators. If omitted, every
      field couples equally to alpha and beta.
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: HamBasis = "restricted"
    nchol: int | None = None
    field_factors: jax.Array | None = None
    field_spin_coeffs: jax.Array | None = None
    field_labels: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.basis not in ("restricted", "unrestricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")
        chol_shape = getattr(self.chol, "shape", None)
        if chol_shape is None:
            return

        n_chol_shape = int(chol_shape[0])
        nchol = self.nchol
        if nchol is None:
            object.__setattr__(self, "nchol", n_chol_shape)
        elif n_chol_shape not in (0, int(nchol)):
            raise ValueError(f"nchol={nchol} is inconsistent with chol.shape[0]={n_chol_shape}")
        if self.field_factors is not None and int(self.field_factors.shape[0]) != int(self.nchol):
            raise ValueError(
                f"field_factors length {self.field_factors.shape[0]} is inconsistent with "
                f"nchol={self.nchol}"
            )
        if self.field_spin_coeffs is not None and tuple(self.field_spin_coeffs.shape) != (
            int(self.nchol),
            2,
        ):
            raise ValueError(
                f"field_spin_coeffs must have shape ({self.nchol}, 2), got "
                f"{self.field_spin_coeffs.shape}"
            )
        if self.field_labels is not None and len(self.field_labels) != int(self.nchol):
            raise ValueError(
                f"field_labels length {len(self.field_labels)} is inconsistent with "
                f"nchol={self.nchol}"
            )

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol, self.field_factors, self.field_spin_coeffs)
        nchol = self.nchol
        assert nchol is not None
        aux = (self.basis, int(nchol), self.field_labels)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol, field_factors, field_spin_coeffs = children
        basis, nchol, field_labels = aux
        return cls(
            h0=h0,
            h1=h1,
            chol=chol,
            basis=basis,
            nchol=nchol,
            field_factors=field_factors,
            field_spin_coeffs=field_spin_coeffs,
            field_labels=field_labels,
        )


def n_fields(ham: HamChol) -> int:
    nchol = ham.nchol
    assert nchol is not None
    return int(nchol)


def slice_ham_level(ham: HamChol, *, norb_keep: int | None, nchol_keep: int | None) -> HamChol:
    """
    Build a HamChol view for measurement in MLMC:
      - slice orbitals as a prefix [:norb_keep]
      - slice chol as a prefix [:nchol_keep]
    """
    h0 = ham.h0
    h1 = ham.h1
    chol = ham.chol

    new_nchol = ham.nchol

    if norb_keep is not None:
        if ham.basis == "unrestricted":
            h1 = h1[:, :norb_keep, :norb_keep]
            chol = chol[:, :, :norb_keep, :norb_keep]
        else:
            h1 = h1[:norb_keep, :norb_keep]
            chol = chol[:, :norb_keep, :norb_keep]

    if nchol_keep is not None:
        chol = chol[:nchol_keep]
        ham_nchol = ham.nchol
        assert ham_nchol is not None
        new_nchol = min(int(ham_nchol), nchol_keep)

    field_factors = ham.field_factors
    field_spin_coeffs = ham.field_spin_coeffs
    field_labels = ham.field_labels
    if nchol_keep is not None:
        if field_factors is not None:
            field_factors = field_factors[:new_nchol]
        if field_spin_coeffs is not None:
            field_spin_coeffs = field_spin_coeffs[:new_nchol]
        if field_labels is not None:
            field_labels = field_labels[:new_nchol]

    return HamChol(
        h0=h0,
        h1=h1,
        chol=chol,
        basis=ham.basis,
        nchol=new_nchol,
        field_factors=field_factors,
        field_spin_coeffs=field_spin_coeffs,
        field_labels=field_labels,
    )
