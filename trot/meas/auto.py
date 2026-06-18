# meas/auto.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, TrialOps, k_energy, k_force_bias
from ..core.system import System
from ..core.typing import trial_data
from ..ham.chol import HamChol


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AutoMeasCtx:
    """
    Small intermediates for auto-measurements.
    """

    h1_eff: jax.Array  # (n,n) or (ns,ns)
    eps: jax.Array  # scalar

    def tree_flatten(self):
        return (
            self.h1_eff,
            self.eps,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        h1_eff, eps = children
        return cls(
            h1_eff=h1_eff,
            eps=eps,
        )


def _v0_from_chol(chol: jax.Array) -> jax.Array:
    return 0.5 * jnp.einsum("gik,gjk->ij", chol, chol, optimize="optimal")


def _field_factors(ham_data: HamChol) -> jax.Array:
    if ham_data.field_factors is not None:
        return ham_data.field_factors
    n_chol = ham_data.nchol
    n_fields = int(n_chol) if n_chol is not None else int(ham_data.chol.shape[0])
    return jnp.full((n_fields,), 1.0j, dtype=jnp.complex128)


def _quadratic_coefficients(ham_data: HamChol) -> jax.Array:
    factors = _field_factors(ham_data)
    return -(factors * factors)


def _v0_from_ham(ham_data: HamChol) -> jax.Array:
    coeff = _quadratic_coefficients(ham_data)
    if ham_data.basis == "unrestricted":
        return 0.5 * jnp.einsum(
            "g,gsik,gskj->sij",
            coeff,
            ham_data.chol,
            ham_data.chol,
            optimize="optimal",
        )
    if ham_data.field_spin_coeffs is not None:
        spin_coeffs = ham_data.field_spin_coeffs
        v0_a = jnp.einsum(
            "g,gik,gjk->ij",
            coeff * spin_coeffs[:, 0] * spin_coeffs[:, 0],
            ham_data.chol,
            ham_data.chol,
            optimize="optimal",
        )
        v0_b = jnp.einsum(
            "g,gik,gjk->ij",
            coeff * spin_coeffs[:, 1] * spin_coeffs[:, 1],
            ham_data.chol,
            ham_data.chol,
            optimize="optimal",
        )
        return 0.5 * jnp.stack([v0_a, v0_b], axis=0)
    return 0.5 * jnp.einsum("g,gik,gjk->ij", coeff, ham_data.chol, ham_data.chol)


def build_meas_ctx(ham_data: HamChol, _trial_data: trial_data, eps: float = 1.0e-4) -> AutoMeasCtx:
    v0 = _v0_from_ham(ham_data)
    h1_eff = ham_data.h1 - v0
    return AutoMeasCtx(h1_eff=h1_eff, eps=jnp.asarray(eps))


def _matmul_block_diag_if_needed(mat: jax.Array, w: jax.Array) -> jax.Array:
    n = mat.shape[0]
    if w.shape[0] == n:
        return mat @ w
    if w.shape[0] == 2 * n:
        top = mat @ w[:n, :]
        bot = mat @ w[n:, :]
        return jnp.vstack([top, bot])
    raise ValueError(f"incompatible shapes: mat {mat.shape}, walker {w.shape}")


def _lin_rot_walker_array(w: jax.Array, mat: jax.Array, x: jax.Array) -> jax.Array:
    return w + x * _matmul_block_diag_if_needed(mat, w)


def _quad_rot_walker_array(w: jax.Array, mat: jax.Array, x: jax.Array) -> jax.Array:
    mw = _matmul_block_diag_if_needed(mat, w)
    mmw = _matmul_block_diag_if_needed(mat, mw)
    return w + x * mw + 0.5 * (x * x) * mmw


def _force_bias_from_overlap_array(
    w: jax.Array,
    ham_data: HamChol,
    overlap: Callable[[jax.Array, Any], jax.Array],
    trial_data: trial_data,
) -> jax.Array:
    """
    Force bias gamma:
      <T| chol_gamma |w> / <T|w>
    computed as d/dx_gamma <T| exp(sum x_gamma chol_gamma) |w> / <T|w>
    using vjp at x=0 to linear order in the rotated walker.
    """
    chol = ham_data.chol  # (n_fields, n, n) or (n_fields, ns, ns) depending on basis
    n_fields = chol.shape[0]

    def f(x_gamma: jax.Array) -> jax.Array:
        x_chol = jnp.einsum("gij,g->ij", chol, x_gamma, optimize="optimal")
        w1 = w + _matmul_block_diag_if_needed(x_chol, w)  # linearized exp
        return overlap(w1, trial_data)

    x0 = jnp.zeros((n_fields,), dtype=w.dtype)
    val, pullback = jax.vjp(f, x0)
    grad_x = pullback(jnp.asarray(1.0, dtype=val.dtype))[0]
    return grad_x / val


def force_bias_kernel_rw_rh(
    w: jax.Array,
    ham_data: HamChol,
    _meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _force_bias_from_overlap_array(w, ham_data, overlap, trial_data)


def force_bias_kernel_gw_rh(
    w: jax.Array,
    ham_data: HamChol,
    _meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _force_bias_from_overlap_array(w, ham_data, overlap, trial_data)


def force_bias_kernel_uw_rh(
    w: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    _meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
) -> jax.Array:
    wu, wd = w
    chol = ham_data.chol
    n_fields = chol.shape[0]
    spin_coeffs = ham_data.field_spin_coeffs

    def f(x_gamma: jax.Array) -> jax.Array:
        if ham_data.basis == "unrestricted":
            x_chol_a = jnp.einsum("gij,g->ij", chol[:, 0], x_gamma, optimize="optimal")
            x_chol_b = jnp.einsum("gij,g->ij", chol[:, 1], x_gamma, optimize="optimal")
            wu1 = wu + x_chol_a @ wu
            wd1 = wd + x_chol_b @ wd
        elif spin_coeffs is None:
            x_chol = jnp.einsum("gij,g->ij", chol, x_gamma, optimize="optimal")
            wu1 = wu + x_chol @ wu
            wd1 = wd + x_chol @ wd
        else:
            x_chol_a = jnp.einsum(
                "gij,g,g->ij", chol, x_gamma, spin_coeffs[:, 0], optimize="optimal"
            )
            x_chol_b = jnp.einsum(
                "gij,g,g->ij", chol, x_gamma, spin_coeffs[:, 1], optimize="optimal"
            )
            wu1 = wu + x_chol_a @ wu
            wd1 = wd + x_chol_b @ wd
        return overlap((wu1, wd1), trial_data)

    x0 = jnp.zeros((n_fields,), dtype=wu.dtype)
    val, pullback = jax.vjp(f, x0)
    grad_x = pullback(jnp.asarray(1.0, dtype=val.dtype))[0]
    return grad_x / val


def _energy_from_overlap_array(
    w: jax.Array,
    ham_data: HamChol,
    meas_ctx: AutoMeasCtx,
    overlap: Callable[[jax.Array, Any], jax.Array],
    trial_data: trial_data,
) -> jax.Array:
    """
    Local energy from overlap derivatives:
      E = ( d/dx <T|exp(x h1_eff)|w> + 1/2 * sum_g d^2/dx^2 <T|exp(x chol_g)|w> ) / <T|w> + h0
    where first derivative is AD (jvp) and second derivative is FD on the quadratic truncation.
    """
    h0 = ham_data.h0
    h1_eff = meas_ctx.h1_eff
    chol = ham_data.chol
    eps = meas_ctx.eps

    # one-body derivative via jvp at x=0
    def f1(x: jax.Array) -> jax.Array:
        w1 = _lin_rot_walker_array(w, h1_eff, x)
        return overlap(w1, trial_data)

    x0 = jnp.asarray(0.0)
    ovlp0, d_ovlp = jax.jvp(f1, (x0,), (jnp.asarray(1.0, dtype=x0.dtype),))

    # two-body second derivative sum via FD on quadratic truncation
    def weighted_d2_sum(x: jax.Array) -> jax.Array:
        acc0 = jnp.zeros((), dtype=ovlp0.dtype)
        coeff = _quadratic_coefficients(ham_data)

        def body(acc, inputs):
            chol_i, coeff_i = inputs
            wi = _quad_rot_walker_array(w, chol_i, x)
            second = (overlap(wi, trial_data) - ovlp0) / (0.5 * x * x)
            return acc + coeff_i * second, None

        acc, _ = lax.scan(body, acc0, (chol, coeff))
        return acc

    d2_sum = 0.5 * (weighted_d2_sum(+eps) + weighted_d2_sum(-eps))

    return (d_ovlp + 0.5 * d2_sum) / ovlp0 + h0


def energy_kernel_rw_rh(
    w: jax.Array,
    ham_data: HamChol,
    meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _energy_from_overlap_array(w, ham_data, meas_ctx, overlap, trial_data)


def energy_kernel_gw_rh(
    w: jax.Array,
    ham_data: HamChol,
    meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _energy_from_overlap_array(w, ham_data, meas_ctx, overlap, trial_data)


def energy_kernel_uw_rh(
    w: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: AutoMeasCtx,
    trial_data: trial_data,
    *,
    overlap,
) -> jax.Array:
    wu, wd = w
    h0 = ham_data.h0
    h1_eff = meas_ctx.h1_eff
    chol = ham_data.chol
    n_fields = chol.shape[0]
    eps = meas_ctx.eps

    # one-body derivative via jvp
    def f1(x: jax.Array) -> jax.Array:
        if h1_eff.ndim == 3 and h1_eff.shape[0] == 2:
            wu1 = wu + x * (h1_eff[0] @ wu)
            wd1 = wd + x * (h1_eff[1] @ wd)
        else:
            wu1 = wu + x * (h1_eff @ wu)
            wd1 = wd + x * (h1_eff @ wd)
        return overlap((wu1, wd1), trial_data)

    x0 = jnp.asarray(0.0)
    ovlp0, d_ovlp = jax.jvp(f1, (x0,), (jnp.asarray(1.0, dtype=x0.dtype),))

    def weighted_d2_sum(x: jax.Array) -> jax.Array:
        acc0 = jnp.zeros((), dtype=ovlp0.dtype)
        coeff = _quadratic_coefficients(ham_data)
        spin_coeffs = ham_data.field_spin_coeffs
        scan_spin = (
            jnp.ones((n_fields, 2), dtype=chol.dtype) if spin_coeffs is None else spin_coeffs
        )

        def body(acc, inputs):
            chol_i, coeff_i, spin_i = inputs
            if ham_data.basis == "unrestricted":
                chol_a = chol_i[0]
                chol_b = chol_i[1]
            else:
                chol_a = chol_i if spin_coeffs is None else spin_i[0] * chol_i
                chol_b = chol_i if spin_coeffs is None else spin_i[1] * chol_i
            wu1 = wu + x * (chol_a @ wu) + 0.5 * (x * x) * (chol_a @ (chol_a @ wu))
            wd1 = wd + x * (chol_b @ wd) + 0.5 * (x * x) * (chol_b @ (chol_b @ wd))
            second = (overlap((wu1, wd1), trial_data) - ovlp0) / (0.5 * x * x)
            return acc + coeff_i * second, None

        acc, _ = lax.scan(body, acc0, (chol, coeff, scan_spin))
        return acc

    d2_sum = 0.5 * (weighted_d2_sum(+eps) + weighted_d2_sum(-eps))

    return (d_ovlp + 0.5 * d2_sum) / ovlp0 + h0


def make_auto_meas_ops(
    sys: System,
    trial_ops_: TrialOps,
    *,
    eps: float = 1.0e-4,
) -> MeasOps:
    """
    Measurement ops that compute force bias and energy by differentiating overlaps.
    This reuses the trial overlap from `trial_ops_` and avoids trial-specific
    half-rotated formulas.

    Note: build_meas_ctx does NOT depend on trial_data for this implementation,
    but we keep the signature (ham, trial) for compatibility.
    """
    wk = sys.walker_kind.lower()
    overlap = trial_ops_.overlap

    def build_ctx(ham_data: HamChol, trial_data: Any) -> AutoMeasCtx:
        return build_meas_ctx(ham_data, trial_data, eps=eps)

    if wk == "restricted":
        fb = lambda walker, ham_data, meas_ctx, trial_data: force_bias_kernel_rw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        ene = lambda walker, ham_data, meas_ctx, trial_data: energy_kernel_rw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        return MeasOps(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={k_force_bias: fb, k_energy: ene},
        )

    if wk == "unrestricted":
        fb = lambda walker, ham_data, meas_ctx, trial_data: force_bias_kernel_uw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        ene = lambda walker, ham_data, meas_ctx, trial_data: energy_kernel_uw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        return MeasOps(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={
                k_force_bias: fb,
                k_energy: ene,
            },
        )

    if wk == "generalized":
        fb = lambda walker, ham_data, meas_ctx, trial_data: force_bias_kernel_gw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        ene = lambda walker, ham_data, meas_ctx, trial_data: energy_kernel_gw_rh(
            walker, ham_data, meas_ctx, trial_data, overlap=overlap
        )
        return MeasOps(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={
                k_force_bias: fb,
                k_energy: ene,
            },
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
