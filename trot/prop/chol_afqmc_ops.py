from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..ham.chol import HamChol
from .utils import taylor_expm_action

# contains low level details of AFQMC chol propagation

CholDecomposition = Literal["charge", "spin", "spin_null"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholAfqmcCtx:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half: jax.Array  # (n,n) or (ns,ns)
    mf_shifts: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    chol_flat: jax.Array  # (n_fields, n*n)
    norb: int
    decomposition: CholDecomposition = "charge"
    spin_decomposition_lambda: float = 1.0
    spin_null_eta: float = 0.0

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat,
        ), (self.norb, self.decomposition, self.spin_decomposition_lambda, self.spin_null_eta)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, sqrt_dt, exp_h1_half, mf_shifts, h0_prop, chol_flat = children
        norb, decomposition, spin_decomposition_lambda, spin_null_eta = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
            norb=norb,
            decomposition=decomposition,
            spin_decomposition_lambda=spin_decomposition_lambda,
            spin_null_eta=spin_null_eta,
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


def _spin_mf_shifts(
    ham_data: HamChol,
    rdm1: jax.Array,
    *,
    spin_decomposition_lambda: float = 1.0,
) -> jax.Array:
    if ham_data.basis != "restricted":
        raise ValueError("Spin decomposition currently requires a restricted Hamiltonian basis.")
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError(
            "Spin decomposition requires a spin-block rdm1 with shape (2, norb, norb)."
        )
    if not 0.0 <= spin_decomposition_lambda <= 1.0:
        raise ValueError("spin_decomposition_lambda must be between 0 and 1.")

    dm_a, dm_b = rdm1[0], rdm1[1]
    tr_a = jnp.einsum("gij,ji->g", ham_data.chol, dm_a, optimize="optimal")
    tr_b = jnp.einsum("gij,ji->g", ham_data.chol, dm_b, optimize="optimal")
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=tr_a.real.dtype))
    mf_a = 1.0j * sqrt2 * tr_a
    mf_b = 1.0j * sqrt2 * tr_b
    mf_s = -(tr_a - tr_b)
    spin_mf = jnp.concatenate([mf_a, mf_b, mf_s], axis=0)

    if spin_decomposition_lambda >= 1.0:
        return spin_mf

    lam = jnp.asarray(spin_decomposition_lambda, dtype=jnp.real(tr_a).dtype)
    charge_mf = 1.0j * (tr_a + tr_b)
    return jnp.concatenate(
        [
            jnp.sqrt(1.0 - lam) * charge_mf,
            jnp.sqrt(lam) * spin_mf,
        ],
        axis=0,
    )


def _spin_null_mf_shifts(
    ham_data: HamChol,
    rdm1: jax.Array,
    *,
    spin_null_eta: float = 0.0,
) -> jax.Array:
    if ham_data.basis != "restricted":
        raise ValueError("spin_null decomposition currently requires a restricted Hamiltonian basis.")
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError(
            "spin_null decomposition requires a spin-block rdm1 with shape (2, norb, norb)."
        )
    if spin_null_eta < 0.0:
        raise ValueError("spin_null_eta must be non-negative.")

    dm_a, dm_b = rdm1[0], rdm1[1]
    tr_a = jnp.einsum("gij,ji->g", ham_data.chol, dm_a, optimize="optimal")
    tr_b = jnp.einsum("gij,ji->g", ham_data.chol, dm_b, optimize="optimal")
    eta = jnp.asarray(spin_null_eta, dtype=jnp.real(tr_a).dtype)
    charge_mf = 1.0j * (tr_a + tr_b)
    minus_mf = eta * 1.0j * (tr_a - tr_b)
    null_mf = -eta * (tr_a - tr_b)
    return jnp.concatenate([charge_mf, minus_mf, null_mf], axis=0)


def _build_exp_h1_half_from_h1(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jax.scipy.linalg.expm(-0.5 * dt * h1)


def _make_vhs_split_flat(*, chol_flat: jax.Array, x: jax.Array, n: int) -> jax.Array:
    # chol_flat: (n_fields, n*n) real
    v_re = jnp.real(x) @ chol_flat  # (n*n,)
    v_im = jnp.imag(x) @ chol_flat  # (n*n,)
    return lax.complex(v_re, v_im).reshape(n, n)


def _make_spin_vhs_split_flat(
    *, chol_flat: jax.Array, field: jax.Array, n: int
) -> tuple[jax.Array, jax.Array]:
    n_chol = field.shape[0] // 3
    chol = chol_flat[:n_chol]
    field_a = field[:n_chol]
    field_b = field[n_chol : 2 * n_chol]
    field_s = field[2 * n_chol :]
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(field).dtype))

    coeff_a = 1.0j * sqrt2 * field_a - field_s
    coeff_b = 1.0j * sqrt2 * field_b + field_s
    return (
        _make_vhs_split_flat(chol_flat=chol, x=coeff_a, n=n),
        _make_vhs_split_flat(chol_flat=chol, x=coeff_b, n=n),
    )


def _make_spin_vhs_split_flat_with_lambda(
    *,
    chol_flat: jax.Array,
    field: jax.Array,
    n: int,
    spin_decomposition_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    if spin_decomposition_lambda >= 1.0:
        return _make_spin_vhs_split_flat(chol_flat=chol_flat, field=field, n=n)

    n_chol = field.shape[0] // 4
    chol = chol_flat[:n_chol]
    field_c = field[:n_chol]
    field_a = field[n_chol : 2 * n_chol]
    field_b = field[2 * n_chol : 3 * n_chol]
    field_s = field[3 * n_chol :]

    dtype = jnp.real(field).dtype
    lam = jnp.asarray(spin_decomposition_lambda, dtype=dtype)
    sqrt_charge = jnp.sqrt(1.0 - lam)
    sqrt_spin = jnp.sqrt(lam)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=dtype))

    coeff_charge = 1.0j * sqrt_charge * field_c
    coeff_a = coeff_charge + sqrt_spin * (1.0j * sqrt2 * field_a - field_s)
    coeff_b = coeff_charge + sqrt_spin * (1.0j * sqrt2 * field_b + field_s)
    return (
        _make_vhs_split_flat(chol_flat=chol, x=coeff_a, n=n),
        _make_vhs_split_flat(chol_flat=chol, x=coeff_b, n=n),
    )


def _make_spin_null_vhs_split_flat(
    *,
    chol_flat: jax.Array,
    field: jax.Array,
    n: int,
    spin_null_eta: float,
) -> tuple[jax.Array, jax.Array]:
    n_chol = field.shape[0] // 3
    chol = chol_flat[:n_chol]
    field_c = field[:n_chol]
    field_m = field[n_chol : 2 * n_chol]
    field_n = field[2 * n_chol :]

    eta = jnp.asarray(spin_null_eta, dtype=jnp.real(field).dtype)
    coeff_charge = 1.0j * field_c
    coeff_minus = 1.0j * eta * field_m
    coeff_null = eta * field_n
    coeff_a = coeff_charge + coeff_minus - coeff_null
    coeff_b = coeff_charge - coeff_minus + coeff_null
    return (
        _make_vhs_split_flat(chol_flat=chol, x=coeff_a, n=n),
        _make_vhs_split_flat(chol_flat=chol, x=coeff_b, n=n),
    )


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
    decomposition: CholDecomposition = "charge",
    spin_decomposition_lambda: float = 1.0,
    spin_null_eta: float = 0.0,
) -> CholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)
    if not 0.0 <= spin_decomposition_lambda <= 1.0:
        raise ValueError("spin_decomposition_lambda must be between 0 and 1.")
    if spin_null_eta < 0.0:
        raise ValueError("spin_null_eta must be non-negative.")

    if decomposition == "charge":
        mf = _mf_shifts(ham_data, rdm1)
        charge_mf = mf
    elif decomposition == "spin":
        mf = _spin_mf_shifts(
            ham_data,
            rdm1,
            spin_decomposition_lambda=spin_decomposition_lambda,
        )
        charge_mf = _mf_shifts(ham_data, rdm1)
    elif decomposition == "spin_null":
        mf = _spin_null_mf_shifts(ham_data, rdm1, spin_null_eta=spin_null_eta)
        charge_mf = _mf_shifts(ham_data, rdm1)
    else:
        raise ValueError(f"Unknown Cholesky decomposition: {decomposition!r}")

    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2)
    h1_eff = _get_h1_eff(ham_data, charge_mf)

    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    chol_flat_base = ham_data.chol.reshape(ham_data.chol.shape[0], -1)
    if decomposition == "spin":
        n_copies = 3 if spin_decomposition_lambda >= 1.0 else 4
        chol_flat_base = jnp.concatenate([chol_flat_base] * n_copies, axis=0)
    elif decomposition == "spin_null":
        chol_flat_base = jnp.concatenate([chol_flat_base] * 3, axis=0)
    chol_flat = chol_flat_base.astype(chol_flat_precision)
    norb = ham_data.chol.shape[1]
    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        norb=norb,
        decomposition=decomposition,
        spin_decomposition_lambda=float(spin_decomposition_lambda),
        spin_null_eta=float(spin_null_eta),
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
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], tuple[jax.Array, jax.Array]],
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    vhs_a, vhs_b = make_vhs(field, prop_ctx)
    a = prop_ctx.sqrt_dt.astype(wu.dtype)
    return (
        taylor_expm_action(a, vhs_a.astype(wu.dtype), wu, n_terms),
        taylor_expm_action(a, vhs_b.astype(wd.dtype), wd, n_terms),
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


def _apply_two_body_generalized_spin_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], tuple[jax.Array, jax.Array]],
) -> jax.Array:
    vhs_a, vhs_b = make_vhs(field, prop_ctx)
    a = prop_ctx.sqrt_dt.astype(w.dtype)
    norb = w.shape[0] // 2
    top = taylor_expm_action(a, vhs_a.astype(w.dtype), w[:norb, :], n_terms)
    bot = taylor_expm_action(a, vhs_b.astype(w.dtype), w[norb:, :], n_terms)
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
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], tuple[jax.Array, jax.Array]],
) -> Tuple[jax.Array, jax.Array]:
    w1 = _apply_one_body_half_unrestricted(w_ud, prop_ctx)
    w2 = _apply_two_body_unrestricted_spin(w1, field, prop_ctx, n_terms, make_vhs=make_vhs)
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


def _apply_trotter_g_spin_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], tuple[jax.Array, jax.Array]],
) -> jax.Array:
    w1 = _apply_one_body_half_generalized_from_restricted(w, prop_ctx)
    w2 = _apply_two_body_generalized_spin_from_restricted(
        w1, field, prop_ctx, n_terms, make_vhs=make_vhs
    )
    return _apply_one_body_half_generalized_from_restricted(w2, prop_ctx)


def make_trotter_ops(
    ham_basis: str,
    walker_kind: str,
    mixed_precision: bool = False,
    decomposition: CholDecomposition = "charge",
    spin_decomposition_lambda: float = 1.0,
    spin_null_eta: float = 0.0,
) -> TrotterOps:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)
    if not 0.0 <= spin_decomposition_lambda <= 1.0:
        raise ValueError("spin_decomposition_lambda must be between 0 and 1.")
    if spin_null_eta < 0.0:
        raise ValueError("spin_null_eta must be non-negative.")

    walker_kind = walker_kind.lower()

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

    def make_vhs_spin(field: jax.Array, ctx: CholAfqmcCtx) -> tuple[jax.Array, jax.Array]:
        if ctx.decomposition == "spin_null":
            return _make_spin_null_vhs_split_flat(
                chol_flat=ctx.chol_flat,
                field=field.astype(vhs_complex_dtype),
                n=ctx.norb,
                spin_null_eta=ctx.spin_null_eta,
            )
        return _make_spin_vhs_split_flat_with_lambda(
            chol_flat=ctx.chol_flat,
            field=field.astype(vhs_complex_dtype),
            n=ctx.norb,
            spin_decomposition_lambda=ctx.spin_decomposition_lambda,
        )

    if walker_kind not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if ham_basis not in ("restricted", "generalized"):
        raise ValueError(f"unknown ham_basis: {ham_basis}")

    if decomposition not in ("charge", "spin", "spin_null"):
        raise ValueError(f"unknown decomposition: {decomposition}")

    if decomposition in ("spin", "spin_null"):
        if ham_basis != "restricted":
            raise NotImplementedError(
                f"{decomposition} decomposition is only implemented for restricted Hamiltonians."
            )
        match walker_kind:
            case "unrestricted":
                apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs_spin: _apply_trotter_u_spin(
                    w, f, ctx, n_terms, make_vhs=mv
                )
            case "generalized":
                apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs_spin: _apply_trotter_g_spin_from_restricted(
                    w, f, ctx, n_terms, make_vhs=mv
                )
            case _:
                raise NotImplementedError(
                    f"{decomposition} decomposition currently supports unrestricted or generalized walkers."
                )
        return TrotterOps(apply_trotter)

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
