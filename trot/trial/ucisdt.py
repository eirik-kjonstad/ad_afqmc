from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdtTrial:
    """
    Unrestricted CISDT trial in an MO basis where the reference
    determinant occupies the first nocc[0] alpha and nocc[1] beta orbitals.

    Arrays:
      mo_coeff_a: (norb, norb)
      mo_coeff_b: (norb, norb)                                  beta MO rotation matrix
      c1a : (nocc[0], nvir[0])                                  alpha singles
      c1b : (nocc[1], nvir[1])                                  beta singles
      c2aa: (nocc[0], nvir[0], nocc[0], nvir[0])               alpha-alpha doubles
      c2ab: (nocc[0], nvir[0], nocc[1], nvir[1])               alpha-beta doubles
      c2bb: (nocc[1], nvir[1], nocc[1], nvir[1])               beta-beta doubles
      c3aaa: (nocc[0], nvir[0], nocc[0], nvir[0], nocc[0], nvir[0])  AAA triples
      c3aab: (nocc[0], nvir[0], nocc[0], nvir[0], nocc[1], nvir[1])  AAB triples
      c3abb: (nocc[0], nvir[0], nocc[1], nvir[1], nocc[1], nvir[1])  ABB triples
      c3bbb: (nocc[1], nvir[1], nocc[1], nvir[1], nocc[1], nvir[1])  BBB triples
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    c2aa: jax.Array
    c2ab: jax.Array
    c2bb: jax.Array
    c3aaa: jax.Array
    c3aab: jax.Array
    c3abb: jax.Array
    c3bbb: jax.Array

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_b.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.c1a.shape[0]), int(self.c1b.shape[0]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.c1a.shape[1]), int(self.c1b.shape[1]))

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.c2aa,
            self.c2ab,
            self.c2bb,
            self.c3aaa,
            self.c3aab,
            self.c3abb,
            self.c3bbb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            mo_coeff_a,
            mo_coeff_b,
            c1a,
            c1b,
            c2aa,
            c2ab,
            c2bb,
            c3aaa,
            c3aab,
            c3abb,
            c3bbb,
        ) = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            c1a=c1a,
            c1b=c1b,
            c2aa=c2aa,
            c2ab=c2ab,
            c2bb=c2bb,
            c3aaa=c3aaa,
            c3aab=c3aab,
            c3abb=c3abb,
            c3bbb=c3bbb,
        )


def get_rdm1(trial_data: UcisdtTrial) -> jax.Array:
    c0 = jnp.array(
        1.0,
        dtype=jnp.result_type(
            trial_data.c1a,
            trial_data.c1b,
            trial_data.c2aa,
            trial_data.c2ab,
            trial_data.c2bb,
            trial_data.c3aaa,
            trial_data.c3aab,
            trial_data.c3abb,
            trial_data.c3bbb,
        ),
    )
    dm_a, dm_b = _cisdt_1rdm_uhf(
        c0,
        trial_data.c1a,
        trial_data.c1b,
        trial_data.c2aa,
        trial_data.c2ab,
        trial_data.c2bb,
        trial_data.c3aaa,
        trial_data.c3aab,
        trial_data.c3abb,
        trial_data.c3bbb,
    )
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b
    dm_a = c_a @ dm_a @ c_a.conj().T
    dm_b = c_b @ dm_b @ c_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def _cisdt_1rdm_uhf(
    c0: jax.Array,
    c1a: jax.Array,
    c1b: jax.Array,
    c2aa: jax.Array,
    c2ab: jax.Array,
    c2bb: jax.Array,
    c3aaa: jax.Array,
    c3aab: jax.Array,
    c3abb: jax.Array,
    c3bbb: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return spin-separated UCISDT 1-RDMs in alpha and beta MO coordinates."""
    nocc_a = c1a.shape[0]
    nocc_b = c1b.shape[0]

    norm = (
        jnp.conj(c0) * c0
        + jnp.sum(jnp.conj(c1a) * c1a)
        + jnp.sum(jnp.conj(c1b) * c1b)
        + 0.25 * jnp.sum(jnp.conj(c2aa) * c2aa)
        + jnp.sum(jnp.conj(c2ab) * c2ab)
        + 0.25 * jnp.sum(jnp.conj(c2bb) * c2bb)
        + (1.0 / 36.0) * jnp.sum(jnp.conj(c3aaa) * c3aaa)
        + 0.25 * jnp.sum(jnp.conj(c3aab) * c3aab)
        + 0.25 * jnp.sum(jnp.conj(c3abb) * c3abb)
        + (1.0 / 36.0) * jnp.sum(jnp.conj(c3bbb) * c3bbb)
    )

    oo_a = (
        norm * jnp.eye(nocc_a, dtype=norm.dtype)
        - jnp.einsum("ia,ja->ij", jnp.conj(c1a), c1a, optimize="optimal")
        - 0.5 * jnp.einsum("iakb,jakb->ij", jnp.conj(c2aa), c2aa, optimize="optimal")
        - jnp.einsum("iakb,jakb->ij", jnp.conj(c2ab), c2ab, optimize="optimal")
        - (1.0 / 12.0)
        * jnp.einsum("iakblc,jakblc->ij", jnp.conj(c3aaa), c3aaa, optimize="optimal")
        - 0.5 * jnp.einsum("iakblc,jakblc->ij", jnp.conj(c3aab), c3aab, optimize="optimal")
        - 0.25 * jnp.einsum("iakblc,jakblc->ij", jnp.conj(c3abb), c3abb, optimize="optimal")
    )
    vv_a = (
        jnp.einsum("ia,ib->ab", jnp.conj(c1a), c1a, optimize="optimal")
        + 0.5 * jnp.einsum("iajc,ibjc->ab", jnp.conj(c2aa), c2aa, optimize="optimal")
        + jnp.einsum("iajc,ibjc->ab", jnp.conj(c2ab), c2ab, optimize="optimal")
        + (1.0 / 12.0)
        * jnp.einsum("iajckd,ibjckd->ab", jnp.conj(c3aaa), c3aaa, optimize="optimal")
        + 0.5 * jnp.einsum("iajckd,ibjckd->ab", jnp.conj(c3aab), c3aab, optimize="optimal")
        + 0.25 * jnp.einsum("iakcld,ibkcld->ab", jnp.conj(c3abb), c3abb, optimize="optimal")
    )
    ov_a = (
        jnp.conj(c0) * c1a
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1a), c2aa, optimize="optimal")
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1b), c2ab, optimize="optimal")
        + 0.25 * jnp.einsum("jbkc,iajbkc->ia", jnp.conj(c2aa), c3aaa, optimize="optimal")
        + jnp.einsum("jbkc,iajbkc->ia", jnp.conj(c2ab), c3aab, optimize="optimal")
        + 0.25 * jnp.einsum("jbkc,iajbkc->ia", jnp.conj(c2bb), c3abb, optimize="optimal")
    )

    oo_b = (
        norm * jnp.eye(nocc_b, dtype=norm.dtype)
        - jnp.einsum("ia,ja->ij", jnp.conj(c1b), c1b, optimize="optimal")
        - 0.5 * jnp.einsum("iakb,jakb->ij", jnp.conj(c2bb), c2bb, optimize="optimal")
        - jnp.einsum("kaib,kajb->ij", jnp.conj(c2ab), c2ab, optimize="optimal")
        - (1.0 / 12.0)
        * jnp.einsum("iakblc,jakblc->ij", jnp.conj(c3bbb), c3bbb, optimize="optimal")
        - 0.5 * jnp.einsum("kaiblc,kajblc->ij", jnp.conj(c3abb), c3abb, optimize="optimal")
        - 0.25 * jnp.einsum("kalbic,kalbjc->ij", jnp.conj(c3aab), c3aab, optimize="optimal")
    )
    vv_b = (
        jnp.einsum("ia,ib->ab", jnp.conj(c1b), c1b, optimize="optimal")
        + 0.5 * jnp.einsum("iajc,ibjc->ab", jnp.conj(c2bb), c2bb, optimize="optimal")
        + jnp.einsum("icja,icjb->ab", jnp.conj(c2ab), c2ab, optimize="optimal")
        + (1.0 / 12.0)
        * jnp.einsum("iajckd,ibjckd->ab", jnp.conj(c3bbb), c3bbb, optimize="optimal")
        + 0.5 * jnp.einsum("kciajd,kcibjd->ab", jnp.conj(c3abb), c3abb, optimize="optimal")
        + 0.25 * jnp.einsum("kcldia,kcldib->ab", jnp.conj(c3aab), c3aab, optimize="optimal")
    )
    ov_b = (
        jnp.conj(c0) * c1b
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1b), c2bb, optimize="optimal")
        + jnp.einsum("jb,jbia->ia", jnp.conj(c1a), c2ab, optimize="optimal")
        + 0.25 * jnp.einsum("jbkc,iajbkc->ia", jnp.conj(c2bb), c3bbb, optimize="optimal")
        + jnp.einsum("jbkc,jbkcia->ia", jnp.conj(c2ab), c3abb, optimize="optimal")
        + 0.25 * jnp.einsum("jbkc,jbkcia->ia", jnp.conj(c2aa), c3aab, optimize="optimal")
    )

    def assemble(oo: jax.Array, ov: jax.Array, vv: jax.Array) -> jax.Array:
        nocc, nvir = ov.shape
        nmo = nocc + nvir
        dm = jnp.zeros((nmo, nmo), dtype=jnp.result_type(oo, ov, vv))
        dm = dm.at[:nocc, :nocc].set(oo)
        dm = dm.at[:nocc, nocc:].set(ov)
        dm = dm.at[nocc:, :nocc].set(jnp.conj(ov.T))
        dm = dm.at[nocc:, nocc:].set(vv)
        return dm / norm

    return assemble(oo_a, ov_a, vv_a), assemble(oo_b, ov_b, vv_b)


def overlap_r(walker: jax.Array, trial_data: UcisdtTrial) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return overlap_u((walker[:, :n_elec_0], walker[:, :n_elec_1]), trial_data)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UcisdtTrial) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c_b = trial_data.mo_coeff_b

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]
    wob = wb[:n_ob, :]

    g_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    g_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    g_a_vir = g_a[:, n_oa:]
    g_b_vir = g_b[:, n_ob:]

    o0 = jnp.linalg.det(woa) * jnp.linalg.det(wob)
    o1 = jnp.einsum("ia,ia", trial_data.c1a, g_a_vir) + jnp.einsum("ia,ia", trial_data.c1b, g_b_vir)
    o2 = (
        0.5 * jnp.einsum("iajb,ia,jb", trial_data.c2aa, g_a_vir, g_a_vir)
        + 0.5 * jnp.einsum("iajb,ia,jb", trial_data.c2bb, g_b_vir, g_b_vir)
        + jnp.einsum("iajb,ia,jb", trial_data.c2ab, g_a_vir, g_b_vir)
    )
    o3 = (
        (1 / 6) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3aaa, g_a_vir, g_a_vir, g_a_vir)
        + (1 / 6) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3bbb, g_b_vir, g_b_vir, g_b_vir)
        + (1 / 2) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3aab, g_a_vir, g_a_vir, g_b_vir)
        + (1 / 2) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3abb, g_a_vir, g_b_vir, g_b_vir)
    )
    return (1.0 + o1 + o2 + o3) * o0


def make_ucisdt_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)
    if wk == "generalized":
        raise NotImplementedError("UCISDT does not support generalized walkers.")
    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


def make_ucisdt_trial_data(data: dict, sys: System) -> UcisdtTrial:
    return UcisdtTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=jnp.asarray(data["ci1a"]),
        c1b=jnp.asarray(data["ci1b"]),
        c2aa=jnp.asarray(data["ci2aa"]),
        c2ab=jnp.asarray(data["ci2ab"]),
        c2bb=jnp.asarray(data["ci2bb"]),
        c3aaa=jnp.asarray(data["ci3aaa"]),
        c3aab=jnp.asarray(data["ci3aab"]),
        c3abb=jnp.asarray(data["ci3abb"]),
        c3bbb=jnp.asarray(data["ci3bbb"]),
    )
