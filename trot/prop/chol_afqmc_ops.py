from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..ham.chol import HamChol
from .utils import taylor_expm_action

# contains low level details of AFQMC chol propagation


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholAfqmcCtx:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half: jax.Array  # (n,n) or (ns,ns)
    mf_shifts: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    chol_flat: jax.Array  # (n_fields, n*n)
    field_factors: jax.Array  # (n_fields,)
    field_spin_coeffs: jax.Array | None  # (n_fields, 2)
    norb: int

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat,
            self.field_factors,
            self.field_spin_coeffs,
        ), (self.norb,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            dt,
            sqrt_dt,
            exp_h1_half,
            mf_shifts,
            h0_prop,
            chol_flat,
            field_factors,
            field_spin_coeffs,
        ) = children
        (norb,) = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
            field_factors=field_factors,
            field_spin_coeffs=field_spin_coeffs,
            norb=norb,
        )


class TrotterOps(NamedTuple):
    apply_trotter: Callable[[Any, jax.Array, CholAfqmcCtx, int], Any]  # (w, field, ctx, n_terms)->w


def _as_total_rdm1_restricted(dm: jax.Array) -> jax.Array:
    if dm.ndim == 3 and dm.shape[0] == 2:
        return dm[0] + dm[1]
    return dm


def _get_dm(rdm1: jax.Array, ham_basis: str) -> jax.Array:
    match ham_basis:
        case "restricted":
            dm = _as_total_rdm1_restricted(rdm1)
        case "unrestricted":
            if rdm1.ndim != 3 or rdm1.shape[0] != 2:
                raise ValueError("unrestricted Hamiltonian basis requires rdm1 shape (2, norb, norb).")
            dm = rdm1
        case "generalized":
            if rdm1.ndim == 3 and rdm1.shape[0] == 2:
                z_ab = jnp.zeros_like(rdm1[0])
                z_ba = jnp.zeros_like(rdm1[1])
                dm = jnp.block([[rdm1[0], z_ab], [z_ba, rdm1[1]]])
            else:
                dm = rdm1
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_basis}")
    return dm


def _mf_shifts(ham_data: HamChol, rdm1: jax.Array) -> jax.Array:
    field_factors = _field_factors(ham_data)
    spin_coeffs = ham_data.field_spin_coeffs
    if spin_coeffs is not None:
        if ham_data.basis != "restricted":
            raise NotImplementedError("spin-resolved field coefficients require restricted basis.")
        if rdm1.ndim != 3 or rdm1.shape[0] != 2:
            raise ValueError(
                "spin-resolved field coefficients require rdm1 with shape (2, norb, norb)."
            )
        spin_contractions = jnp.einsum("gij,sji->gs", ham_data.chol, rdm1, optimize="optimal")
        return field_factors * jnp.einsum(
            "gs,gs->g", spin_coeffs, spin_contractions, optimize="optimal"
        )
    if ham_data.basis == "unrestricted":
        dm = _get_dm(rdm1, ham_data.basis)
        return field_factors * jnp.einsum("gsij,sji->g", ham_data.chol, dm, optimize="optimal")
    dm = _get_dm(rdm1, ham_data.basis)
    return field_factors * jnp.einsum("gij,ji->g", ham_data.chol, dm, optimize="optimal")


def _build_exp_h1_half_from_h1(h1: jax.Array, dt: jax.Array) -> jax.Array:
    if h1.ndim == 3:
        return jax.vmap(lambda block: jax.scipy.linalg.expm(-0.5 * dt * block))(h1)
    return jax.scipy.linalg.expm(-0.5 * dt * h1)


def _make_vhs_split_flat(*, chol_flat: jax.Array, x: jax.Array, n: int) -> jax.Array:
    # chol_flat: (n_fields, n*n) real
    v_re = jnp.real(x) @ chol_flat  # (n*n,)
    v_im = jnp.imag(x) @ chol_flat  # (n*n,)
    return lax.complex(v_re, v_im).reshape(n, n)


def _make_vhs_spin_resolved_flat(
    *,
    chol_flat: jax.Array,
    field_factors: jax.Array,
    field_spin_coeffs: jax.Array,
    x: jax.Array,
    n: int,
) -> tuple[jax.Array, jax.Array]:
    coeff = x * field_factors
    alpha = _make_vhs_split_flat(
        chol_flat=chol_flat,
        x=coeff * field_spin_coeffs[:, 0],
        n=n,
    )
    beta = _make_vhs_split_flat(
        chol_flat=chol_flat,
        x=coeff * field_spin_coeffs[:, 1],
        n=n,
    )
    return alpha, beta


def _field_factors(ham_data: HamChol) -> jax.Array:
    if ham_data.field_factors is not None:
        return ham_data.field_factors
    n_chol = ham_data.nchol
    n_fields = int(n_chol) if n_chol is not None else int(ham_data.chol.shape[0])
    return jnp.full((n_fields,), 1.0j, dtype=jnp.complex128)


def _get_h1_eff(ham_data: HamChol, mf: jax.Array) -> jax.Array:
    field_factors = _field_factors(ham_data)
    spin_coeffs = ham_data.field_spin_coeffs
    if spin_coeffs is not None:
        if ham_data.basis != "restricted":
            raise NotImplementedError("spin-resolved field coefficients require restricted basis.")
        k_chol = (
            field_factors[:, None, None, None]
            * spin_coeffs[:, :, None, None]
            * ham_data.chol[:, None, :, :]
        )
        v0m = 0.5 * jnp.stack(
            [
                jnp.einsum("gik,gkj->ij", k_chol[:, 0], k_chol[:, 0], optimize="optimal"),
                jnp.einsum("gik,gkj->ij", k_chol[:, 1], k_chol[:, 1], optimize="optimal"),
            ],
            axis=0,
        )
        v1m = jnp.stack(
            [
                jnp.einsum("g,gik->ik", mf, k_chol[:, 0], optimize="optimal"),
                jnp.einsum("g,gik->ik", mf, k_chol[:, 1], optimize="optimal"),
            ],
            axis=0,
        )
        return ham_data.h1[None, :, :] + v0m - v1m

    match ham_data.basis:
        case "restricted" | "generalized":
            k_chol = field_factors[:, None, None] * ham_data.chol
            v0m = 0.5 * jnp.einsum("gik,gkj->ij", k_chol, k_chol, optimize="optimal")
            v1m = jnp.einsum("g,gik->ik", mf, k_chol, optimize="optimal")
            h1_eff = ham_data.h1 + v0m - v1m
        case "unrestricted":
            k_chol = field_factors[:, None, None, None] * ham_data.chol
            v0m = 0.5 * jnp.einsum("gsik,gskj->sij", k_chol, k_chol, optimize="optimal")
            v1m = jnp.einsum("g,gsik->sik", mf, k_chol, optimize="optimal")
            h1_eff = ham_data.h1 + v0m - v1m
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_data.basis}")

    return h1_eff


def _build_prop_ctx(
    ham_data: HamChol,
    rdm1: jax.Array,
    dt: float,
    chol_flat_precision: jnp.dtype = jnp.float64,
) -> CholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)

    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2)
    h1_eff = _get_h1_eff(ham_data, mf)

    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], -1).astype(chol_flat_precision)
    field_factors = _field_factors(ham_data)
    field_spin_coeffs = ham_data.field_spin_coeffs
    norb = ham_data.chol.shape[2] if ham_data.basis == "unrestricted" else ham_data.chol.shape[1]
    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        field_factors=field_factors,
        field_spin_coeffs=field_spin_coeffs,
        norb=norb,
    )


def _apply_one_body_half_array(w: jax.Array, prop_ctx: CholAfqmcCtx) -> jax.Array:
    return prop_ctx.exp_h1_half @ w


def _apply_one_body_half_unrestricted(
    w_ud: Tuple[jax.Array, jax.Array], prop_ctx: CholAfqmcCtx
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    e = prop_ctx.exp_h1_half
    if e.ndim == 3 and e.shape[0] == 2:
        return (e[0] @ wu, e[1] @ wd)
    return (e @ wu, e @ wd)


def _apply_one_body_half_generalized_from_restricted(
    w: jax.Array, prop_ctx: CholAfqmcCtx
) -> jax.Array:
    e = prop_ctx.exp_h1_half
    norb = w.shape[0] // 2
    top = e @ w[:norb, :]
    bot = e @ w[norb:, :]
    return jnp.vstack([top, bot])


def _apply_two_body_array(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    vhs = make_vhs(field, prop_ctx).astype(w.dtype)
    a = prop_ctx.sqrt_dt.astype(w.dtype)
    return taylor_expm_action(a, vhs, w, n_terms)


def _apply_two_body_unrestricted(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    vhs = make_vhs(field, prop_ctx)
    a = prop_ctx.sqrt_dt.astype(wu.dtype)
    if isinstance(vhs, tuple):
        vhs_a, vhs_b = vhs
        return (
            taylor_expm_action(a, vhs_a.astype(wu.dtype), wu, n_terms),
            taylor_expm_action(a, vhs_b.astype(wd.dtype), wd, n_terms),
        )
    vhs = vhs.astype(wu.dtype)
    return (
        taylor_expm_action(a, vhs, wu, n_terms),
        taylor_expm_action(a, vhs, wd, n_terms),
    )


def _apply_two_body_generalized_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    vhs = make_vhs(field, prop_ctx).astype(w.dtype)
    a = prop_ctx.sqrt_dt.astype(w.dtype)
    norb = w.shape[0] // 2
    top = taylor_expm_action(a, vhs, w[:norb, :], n_terms)
    bot = taylor_expm_action(a, vhs, w[norb:, :], n_terms)
    return jnp.vstack([top, bot])


def _apply_trotter_r(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    w1 = _apply_one_body_half_array(w, prop_ctx)
    w2 = _apply_two_body_array(w1, field, prop_ctx, n_terms, make_vhs=make_vhs)
    return _apply_one_body_half_array(w2, prop_ctx)


def _apply_trotter_u(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    w1 = _apply_one_body_half_unrestricted(w_ud, prop_ctx)
    w2 = _apply_two_body_unrestricted(w1, field, prop_ctx, n_terms, make_vhs=make_vhs)
    a = _apply_one_body_half_unrestricted(w2, prop_ctx)
    return a


def _apply_trotter_g_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    w1 = _apply_one_body_half_generalized_from_restricted(w, prop_ctx)
    w2 = _apply_two_body_generalized_from_restricted(
        w1, field, prop_ctx, n_terms, make_vhs=make_vhs
    )
    return _apply_one_body_half_generalized_from_restricted(w2, prop_ctx)


def make_trotter_ops(ham_basis: str, walker_kind: str, mixed_precision: bool = False) -> TrotterOps:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)

    walker_kind = walker_kind.lower()

    if mixed_precision:
        vhs_complex_dtype = jnp.complex64
    else:
        vhs_complex_dtype = jnp.complex128

    def make_vhs(field: jax.Array, ctx: CholAfqmcCtx) -> jax.Array:
        x = field.astype(vhs_complex_dtype) * ctx.field_factors.astype(vhs_complex_dtype)
        return _make_vhs_split_flat(
            chol_flat=ctx.chol_flat,
            x=x,
            n=ctx.norb,
        )

    def make_vhs_unrestricted(field: jax.Array, ctx: CholAfqmcCtx) -> jax.Array:
        if ham_basis == "unrestricted":
            n = ctx.norb
            chol_flat = ctx.chol_flat.reshape(ctx.chol_flat.shape[0], 2, n * n)
            x = field.astype(vhs_complex_dtype) * ctx.field_factors.astype(vhs_complex_dtype)
            alpha = _make_vhs_split_flat(chol_flat=chol_flat[:, 0], x=x, n=n)
            beta = _make_vhs_split_flat(chol_flat=chol_flat[:, 1], x=x, n=n)
            return alpha, beta
        if ctx.field_spin_coeffs is None:
            return make_vhs(field, ctx)
        return _make_vhs_spin_resolved_flat(
            chol_flat=ctx.chol_flat,
            field_factors=ctx.field_factors.astype(vhs_complex_dtype),
            field_spin_coeffs=ctx.field_spin_coeffs.astype(vhs_complex_dtype),
            x=field.astype(vhs_complex_dtype),
            n=ctx.norb,
        )

    if walker_kind not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if ham_basis not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown ham_basis: {ham_basis}")

    match ham_basis, walker_kind:
        case "restricted", "restricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_r(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "unrestricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs_unrestricted: _apply_trotter_u(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "unrestricted", "unrestricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs_unrestricted: _apply_trotter_u(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "generalized":
            apply_trotter = (
                lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_g_from_restricted(
                    w, f, ctx, n_terms, make_vhs=mv
                )
            )
        case "generalized", "generalized":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_r(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case _:
            raise NotImplementedError(
                f"Not implemented for ham_basis={ham_basis} and walker_kind={walker_kind}"
            )

    return TrotterOps(apply_trotter)
