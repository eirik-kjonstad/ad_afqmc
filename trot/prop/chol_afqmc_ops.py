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
    force_bias_scales: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    chol_flat: jax.Array  # (n_fields, n*n)
    norb: int

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.force_bias_scales,
            self.h0_prop,
            self.chol_flat,
        ), (self.norb,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, sqrt_dt, exp_h1_half, mf_shifts, force_bias_scales, h0_prop, chol_flat = children
        (norb,) = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            force_bias_scales=force_bias_scales,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
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
        case "generalized":
            dm = rdm1
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_basis}")
    return dm


def _mf_shifts(ham_data: HamChol, rdm1: jax.Array) -> jax.Array:
    dm = _get_dm(rdm1, ham_data.basis)
    return 1.0j * jnp.einsum("gij,ji->g", ham_data.chol, dm, optimize="optimal")


def _spin_l_contractions(ham_data: HamChol, rdm1: jax.Array) -> tuple[jax.Array, jax.Array]:
    if ham_data.basis != "restricted":
        raise ValueError("Spin HS decomposition requires a restricted-basis Cholesky Hamiltonian.")
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError(
            "Spin HS decomposition requires a spin-resolved RDM1 with shape (2, norb, norb)."
        )

    dm_a, dm_b = rdm1[0], rdm1[1]
    l_a = jnp.einsum("gij,ji->g", ham_data.chol, dm_a, optimize="optimal")
    l_b = jnp.einsum("gij,ji->g", ham_data.chol, dm_b, optimize="optimal")
    return l_a, l_b


def _mf_shifts_spin(ham_data: HamChol, rdm1: jax.Array) -> jax.Array:
    l_a, l_b = _spin_l_contractions(ham_data, rdm1)
    rt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(l_a).dtype))
    return jnp.concatenate(
        [
            1.0j * rt2 * l_a,
            1.0j * rt2 * l_b,
            -(l_a - l_b),
        ]
    )


def _force_bias_scales_charge(mf: jax.Array) -> jax.Array:
    return 1.0j * jnp.ones_like(mf)


def _force_bias_scales_spin(chol: jax.Array) -> jax.Array:
    nchol = chol.shape[0]
    dtype = jnp.result_type(chol, 1.0j)
    rt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(chol).dtype))
    ones = jnp.ones((nchol,), dtype=dtype)
    return jnp.concatenate([1.0j * rt2 * ones, 1.0j * rt2 * ones, -ones])


def _build_exp_h1_half_from_h1(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jax.scipy.linalg.expm(-0.5 * dt * h1)


def _make_vhs_split_flat(*, chol_flat: jax.Array, x: jax.Array, n: int) -> jax.Array:
    # chol_flat: (n_fields, n*n) real
    v_re = jnp.real(x) @ chol_flat  # (n*n,)
    v_im = jnp.imag(x) @ chol_flat  # (n*n,)
    return lax.complex(v_re, v_im).reshape(n, n)


def _get_h1_eff(ham_data: HamChol, mf: jax.Array) -> jax.Array:
    match ham_data.basis:
        case "restricted" | "generalized":
            v0m = 0.5 * jnp.einsum("gik,gkj->ij", ham_data.chol, ham_data.chol, optimize="optimal")
            mf_r = (1.0j * mf).real
            v1m = jnp.einsum("g,gik->ik", mf_r, ham_data.chol, optimize="optimal")
            h1_eff = ham_data.h1 - v0m - v1m
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_data.basis}")

    return h1_eff


def _build_prop_ctx(
    ham_data: HamChol,
    rdm1: jax.Array,
    dt: float,
    chol_flat_precision: jnp.dtype = jnp.float64,
    hs_decomposition: str = "charge",
) -> CholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)

    hs_decomposition = hs_decomposition.lower()
    mf_charge = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf_charge**2)
    h1_eff = _get_h1_eff(ham_data, mf_charge)

    match hs_decomposition:
        case "charge":
            mf = mf_charge
            force_bias_scales = _force_bias_scales_charge(mf)
        case "spin":
            mf = _mf_shifts_spin(ham_data, rdm1)
            force_bias_scales = _force_bias_scales_spin(ham_data.chol)
        case _:
            raise ValueError(f"Unknown HS decomposition: {hs_decomposition}")

    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], -1).astype(chol_flat_precision)
    norb = ham_data.chol.shape[1]
    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        force_bias_scales=force_bias_scales,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        norb=norb,
    )


def _apply_one_body_half_array(w: jax.Array, prop_ctx: CholAfqmcCtx) -> jax.Array:
    return prop_ctx.exp_h1_half @ w


def _apply_one_body_half_unrestricted(
    w_ud: Tuple[jax.Array, jax.Array], prop_ctx: CholAfqmcCtx
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    e = prop_ctx.exp_h1_half
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
    a = (1.0j * prop_ctx.sqrt_dt).astype(w.dtype)
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
    vhs = make_vhs(field, prop_ctx).astype(wu.dtype)
    a = (1.0j * prop_ctx.sqrt_dt).astype(wu.dtype)
    return (
        taylor_expm_action(a, vhs, wu, n_terms),
        taylor_expm_action(a, vhs, wd, n_terms),
    )


def _apply_two_body_unrestricted_spin(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    vhs_complex_dtype: jnp.dtype,
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    nchol = prop_ctx.chol_flat.shape[0]
    x_a = field[:nchol]
    x_b = field[nchol : 2 * nchol]
    x_s = field[2 * nchol :]

    rt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(field).dtype))
    vhs_a = _make_vhs_split_flat(
        chol_flat=prop_ctx.chol_flat,
        x=(rt2 * x_a + 1.0j * x_s).astype(vhs_complex_dtype),
        n=prop_ctx.norb,
    ).astype(wu.dtype)
    vhs_b = _make_vhs_split_flat(
        chol_flat=prop_ctx.chol_flat,
        x=(rt2 * x_b - 1.0j * x_s).astype(vhs_complex_dtype),
        n=prop_ctx.norb,
    ).astype(wd.dtype)

    a = (1.0j * prop_ctx.sqrt_dt).astype(wu.dtype)
    return (
        taylor_expm_action(a, vhs_a, wu, n_terms),
        taylor_expm_action(a, vhs_b, wd, n_terms),
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
    a = (1.0j * prop_ctx.sqrt_dt).astype(w.dtype)
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


def _apply_trotter_u_spin(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    vhs_complex_dtype: jnp.dtype,
) -> Tuple[jax.Array, jax.Array]:
    w1 = _apply_one_body_half_unrestricted(w_ud, prop_ctx)
    w2 = _apply_two_body_unrestricted_spin(
        w1, field, prop_ctx, n_terms, vhs_complex_dtype=vhs_complex_dtype
    )
    return _apply_one_body_half_unrestricted(w2, prop_ctx)


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


def make_trotter_ops(
    ham_basis: str,
    walker_kind: str,
    mixed_precision: bool = False,
    hs_decomposition: str = "charge",
) -> TrotterOps:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)

    walker_kind = walker_kind.lower()
    hs_decomposition = hs_decomposition.lower()

    if mixed_precision:
        vhs_complex_dtype = jnp.complex64
    else:
        vhs_complex_dtype = jnp.complex128

    def make_vhs(field: jax.Array, ctx: CholAfqmcCtx) -> jax.Array:
        return _make_vhs_split_flat(
            chol_flat=ctx.chol_flat,
            x=field.astype(vhs_complex_dtype),
            n=ctx.norb,
        )

    if walker_kind not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if ham_basis not in ("restricted", "generalized"):
        raise ValueError(f"unknown ham_basis: {ham_basis}")

    if hs_decomposition == "spin":
        if (ham_basis, walker_kind) != ("restricted", "unrestricted"):
            raise NotImplementedError(
                "Spin HS decomposition is only implemented for restricted Hamiltonians "
                "with unrestricted walkers."
            )
        return TrotterOps(
            lambda w, f, ctx, n_terms, dtype=vhs_complex_dtype: _apply_trotter_u_spin(
                w, f, ctx, n_terms, vhs_complex_dtype=dtype
            )
        )

    if hs_decomposition != "charge":
        raise ValueError(f"Unknown HS decomposition: {hs_decomposition}")

    match ham_basis, walker_kind:
        case "restricted", "restricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_r(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "unrestricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_u(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "generalized":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: (
                _apply_trotter_g_from_restricted(w, f, ctx, n_terms, make_vhs=mv)
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
