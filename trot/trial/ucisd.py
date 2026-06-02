from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdTrial:
    """
    Unrestricted CISD trial in an MO basis where the reference
    determinant occupies the first nocc[0] alpha and nocc[1] beta orbitals.

    Arrays:
      mo_coeff_b: (norb, nocc[1])
      c1a : (nocc[0], nvir[0])                      singles coefficients c_{i,alpha a,alpha}
      c1b : (nocc[1], nvir[1])                      singles coefficients c_{i,beta  a,beta }
      c2aa: (nocc[0], nvir[0], nocc[0], nvir[0])    doubles coefficients c_{i,alpha a,alpha j,alpha b,alpha}
      c2ab: (nocc[0], nvir[0], nocc[1], nvir[1])    doubles coefficients c_{i,alpha a,alpha j,beta  b,beta }
      c2bb: (nocc[1], nvir[1], nocc[1], nvir[1])    doubles coefficients c_{i,beta  a,beta  j,beta  b,beta }
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    c2aa: jax.Array
    c2ab: jax.Array
    c2bb: jax.Array

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
        ) = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            c1a=c1a,
            c1b=c1b,
            c2aa=c2aa,
            c2ab=c2ab,
            c2bb=c2bb,
        )


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def _cisd_1rdm_uhf(
    c0: jax.Array,
    c1a: jax.Array,
    c1b: jax.Array,
    c2aa: jax.Array,
    c2ab: jax.Array,
    c2bb: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return spin-separated UCISD 1-RDMs in the alpha and beta MO bases.

    The CI doubles in ``UcisdTrial`` are stored as ``c2[i, a, j, b]``.
    """
    nocc_a = c1a.shape[0]
    nocc_b = c1b.shape[0]

    c0_abs2 = jnp.conj(c0) * c0
    norm = (
        c0_abs2
        + jnp.sum(jnp.conj(c1a) * c1a)
        + jnp.sum(jnp.conj(c1b) * c1b)
        + 0.25 * jnp.sum(jnp.conj(c2aa) * c2aa)
        + jnp.sum(jnp.conj(c2ab) * c2ab)
        + 0.25 * jnp.sum(jnp.conj(c2bb) * c2bb)
    )

    oo_a = (
        norm * jnp.eye(nocc_a, dtype=norm.dtype)
        - jnp.einsum("ia,ja->ij", jnp.conj(c1a), c1a, optimize="optimal")
        - 0.5 * jnp.einsum("iakb,jakb->ij", jnp.conj(c2aa), c2aa, optimize="optimal")
        - jnp.einsum("iakb,jakb->ij", jnp.conj(c2ab), c2ab, optimize="optimal")
    )
    vv_a = (
        jnp.einsum("ia,ib->ab", jnp.conj(c1a), c1a, optimize="optimal")
        + 0.5 * jnp.einsum("iajc,ibjc->ab", jnp.conj(c2aa), c2aa, optimize="optimal")
        + jnp.einsum("iajc,ibjc->ab", jnp.conj(c2ab), c2ab, optimize="optimal")
    )
    ov_a = (
        jnp.conj(c0) * c1a
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1a), c2aa, optimize="optimal")
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1b), c2ab, optimize="optimal")
    )

    oo_b = (
        norm * jnp.eye(nocc_b, dtype=norm.dtype)
        - jnp.einsum("ia,ja->ij", jnp.conj(c1b), c1b, optimize="optimal")
        - 0.5 * jnp.einsum("iakb,jakb->ij", jnp.conj(c2bb), c2bb, optimize="optimal")
        - jnp.einsum("kaib,kajb->ij", jnp.conj(c2ab), c2ab, optimize="optimal")
    )
    vv_b = (
        jnp.einsum("ia,ib->ab", jnp.conj(c1b), c1b, optimize="optimal")
        + 0.5 * jnp.einsum("iajc,ibjc->ab", jnp.conj(c2bb), c2bb, optimize="optimal")
        + jnp.einsum("icja,icjb->ab", jnp.conj(c2ab), c2ab, optimize="optimal")
    )
    ov_b = (
        jnp.conj(c0) * c1b
        + jnp.einsum("jb,iajb->ia", jnp.conj(c1b), c2bb, optimize="optimal")
        + jnp.einsum("jb,jbia->ia", jnp.conj(c1a), c2ab, optimize="optimal")
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


def get_rdm1(trial_data: UcisdTrial) -> jax.Array:
    dm_a, dm_b = _cisd_1rdm_uhf(
        jnp.array(
            1.0,
            dtype=jnp.result_type(
                trial_data.c1a,
                trial_data.c1b,
                trial_data.c2aa,
                trial_data.c2ab,
                trial_data.c2bb,
            ),
        ),
        trial_data.c1a,
        trial_data.c1b,
        trial_data.c2aa,
        trial_data.c2ab,
        trial_data.c2bb,
    )
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b
    # CI amplitudes define the beta density in beta-MO coordinates.  Propagation
    # Cholesky vectors are in the alpha-MO Hamiltonian basis, so rotate the
    # density in the dual direction to the measurement kernels' Cb^H L Cb.
    dm_a = c_a @ dm_a @ c_a.conj().T
    dm_b = c_b @ dm_b @ c_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)  # (2, norb, norb)


def overlap_r(walker: jax.Array, trial_data: UcisdTrial) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return overlap_u((walker[:, :n_elec_0], walker[:, :n_elec_1]), trial_data)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UcisdTrial) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    g_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    g_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    g_a = g_a[:, n_oa:]
    g_b = g_b[:, n_ob:]
    o0 = jnp.linalg.det(woa) * jnp.linalg.det(wob)
    o1 = jnp.einsum("ia,ia", c1a, g_a) + jnp.einsum("ia,ia", c1b, g_b)
    o2 = (
        0.5 * jnp.einsum("iajb, ia, jb", c2aa, g_a, g_a)
        + 0.5 * jnp.einsum("iajb, ia, jb", c2bb, g_b, g_b)
        + jnp.einsum("iajb, ia, jb", c2ab, g_a, g_b)
    )
    return (1.0 + o1 + o2) * o0


def overlap_g(walker: jax.Array, trial_data: UcisdTrial) -> jax.Array:
    n_oa, n_ob = trial_data.nocc
    norb = trial_data.norb
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b

    _, ci1A, ci2AA = n_oa, c1a, c2aa
    noccB, ci1B, ci2BB = n_ob, c1b, c2bb
    ci2AB = c2ab

    w = jnp.vstack(
        [
            walker[:norb],
            c_b.T @ walker[norb:, :],
        ]
    )  # put walker_dn in the basis of alpha reference

    Atrial, Btrial = (
        c_a[:, :n_oa],
        c_b[:, :n_ob],
    )
    bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
    o0 = jnp.linalg.det(bra.T.conj() @ walker)

    bra = jnp.block(
        [
            [Atrial, 0 * Btrial],
            [
                0 * Atrial,
                (c_b.T @ c_b)[:, :noccB],
            ],
        ]
    )

    g = (w @ jnp.linalg.inv(bra.T.conj() @ w) @ bra.T.conj()).T

    g_aa = g[:n_oa, n_oa:norb]
    g_bb = g[norb : norb + n_ob, norb + n_ob :]
    g_ab = g[:n_oa, norb + n_ob :]
    g_ba = g[norb : norb + n_ob, n_oa:norb]

    o1 = jnp.einsum("ia,ia", ci1A, g_aa) + jnp.einsum(
        "ia,ia",
        ci1B,
        g_bb,
    )

    # AA
    o2 = jnp.einsum("iajb, ia, jb", ci2AA, g_aa, g_aa)

    # BB
    o2 += jnp.einsum("iajb, ia, jb", ci2BB, g_bb, g_bb)

    # AB
    o2 += 2.0 * jnp.einsum("iajb, ia, jb", ci2AB, g_aa, g_bb)
    o2 -= 2.0 * jnp.einsum("iajb, ib, ja", ci2AB, g_ab, g_ba)

    return (1.0 + o1 + 0.5 * o2) * o0


def make_ucisd_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        overlap_fn = overlap_r
        get_rdm1_fn = get_rdm1
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        get_rdm1_fn = get_rdm1
    elif wk == "generalized":
        overlap_fn = overlap_g
        get_rdm1_fn = get_rdm1
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
    return TrialOps(
        overlap=overlap_fn,
        get_rdm1=get_rdm1_fn,
    )


def make_ucisd_trial_data(data: dict, sys: System) -> UcisdTrial:
    return UcisdTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=jnp.asarray(data["ci1a"]),
        c1b=jnp.asarray(data["ci1b"]),
        c2aa=jnp.asarray(data["ci2aa"]),
        c2ab=jnp.asarray(data["ci2ab"]),
        c2bb=jnp.asarray(data["ci2bb"]),
    )
