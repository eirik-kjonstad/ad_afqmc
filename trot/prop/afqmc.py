from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamBasis, HamChol
from ..sharding import shard_prop_state
from ..walkers import init_walkers
from .chol_afqmc_ops import CholAfqmcCtx, TrotterOps, _build_prop_ctx, make_trotter_ops
from .types import PropOps, PropState, QmcParamsBase


def init_prop_state(
    *,
    sys: System,
    ham_data: HamChol,
    trial_ops: TrialOps,
    trial_data: Any,
    meas_ops: MeasOps,
    params: QmcParamsBase,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
    rdm1: jax.Array | None = None,
    mesh: Mesh | None = None,
) -> PropState:
    """
    Initialize AFQMC propagation state.
    """
    n_walkers = params.n_walkers
    seed = params.seed
    key = jax.random.PRNGKey(int(seed))
    if initial_walkers is None:
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=n_walkers)

    overlaps = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        initial_walkers, trial_data
    )
    if bool(getattr(params, "global_phaseless_projection", False)):
        weights = jnp.ones((n_walkers,), dtype=jnp.result_type(overlaps, 1.0j))
    else:
        weights = jnp.ones((n_walkers,))

    e_est = None
    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        walker_0 = wk.take_walkers(initial_walkers, jnp.array([0]))
        e_samples = jnp.real(
            wk.vmap_chunked(e_kernel, n_chunks=1, in_axes=(0, None, None, None))(
                walker_0, ham_data, meas_ctx, trial_data
            )
        )
        e_est = jnp.mean(e_samples)

    pop_shift = e_est

    node_encounters = jnp.asarray(0)

    state = PropState(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=pop_shift,
        e_estimate=e_est,
        node_encounters=node_encounters,
    )
    return shard_prop_state(state, mesh)


def coefficient_space_global_phaseless_projection(
    preliminary_weights: jax.Array,
    overlaps: jax.Array,
    *,
    dt: float,
    budget_scale: float = 1.0,
    overlap_floor: float = 1.0e-12,
    gauge_fix: bool = False,
) -> jax.Array:
    """
    Apply the recommended coefficient-space global phaseless projection.

    The trust-region budget is ``budget_scale * dt * ||w_tilde / S||`` and the
    correction is ``budget / ||S|| * exp(i arg(sum(w_tilde))) * |S_k|^2``.
    """
    complex_dtype = jnp.result_type(preliminary_weights, overlaps, 1.0j)
    w_tilde = jnp.asarray(preliminary_weights, dtype=complex_dtype)
    s = jnp.asarray(overlaps, dtype=complex_dtype)
    real_dtype = jnp.result_type(jnp.real(w_tilde), jnp.real(s), 1.0)
    eps = jnp.asarray(overlap_floor, dtype=real_dtype)

    abs_s = jnp.abs(s)
    safe_abs_s = jnp.maximum(abs_s, eps)
    a_norm = jnp.sqrt(jnp.sum((jnp.abs(w_tilde) / safe_abs_s) ** 2))
    budget = (
        jnp.asarray(budget_scale, dtype=real_dtype) * jnp.asarray(dt, dtype=real_dtype) * a_norm
    )

    z0 = jnp.sum(w_tilde)
    abs_z0 = jnp.abs(z0)
    phase = jnp.where(abs_z0 > eps, z0 / abs_z0, jnp.asarray(1.0 + 0.0j, dtype=w_tilde.dtype))

    s_norm = jnp.sqrt(jnp.sum(abs_s**2))
    correction = (budget / jnp.where(s_norm > eps, s_norm, 1.0)) * phase * abs_s**2
    weights = jnp.where(s_norm > eps, w_tilde + correction, w_tilde)

    total = jnp.sum(weights)
    abs_total = jnp.abs(total)
    gauge = jnp.where(
        abs_total > eps,
        jnp.conj(total) / abs_total,
        jnp.asarray(1.0 + 0.0j, dtype=weights.dtype),
    )
    return jnp.where(gauge_fix, weights * gauge, weights)


def afqmc_step(
    state: PropState,
    *,
    params: QmcParamsBase,
    ham_data: HamChol,
    trial_data: Any,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: CholAfqmcCtx,
    meas_ctx: Any,
) -> PropState:

    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    fields = jax.random.normal(subkey, (nw, prop_ctx.chol_flat.shape[0]))

    fb_kernel = meas_ops.require_kernel(k_force_bias)
    force_bias = wk.vmap_chunked(
        fb_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data)
    field_shifts = -prop_ctx.sqrt_dt * (1.0j * force_bias - prop_ctx.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.sum(shifted_fields * prop_ctx.mf_shifts, axis=1)
    fb_term = jnp.sum(fields * field_shifts - 0.5 * field_shifts * field_shifts, axis=1)
    walkers_new = wk.vmap_chunked(
        trotter_ops.apply_trotter, n_chunks=params.n_chunks, in_axes=(0, 0, None, None)
    )(state.walkers, shifted_fields, prop_ctx, params.n_exp_terms)

    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        walkers_new, trial_data
    )
    ratio = overlaps_new / state.overlaps
    exponent = (
        -prop_ctx.sqrt_dt * shift_term
        + fb_term
        + prop_ctx.dt * (state.pop_control_ene_shift + prop_ctx.h0_prop)
    )
    imp_fun = jnp.exp(exponent) * ratio

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    if bool(getattr(params, "global_phaseless_projection", False)):
        overlap_floor = float(getattr(params, "global_phaseless_overlap_floor", 1.0e-12))
        preliminary_weights = state.weights * imp_fun
        invalid = (
            ~jnp.isfinite(preliminary_weights)
            | ~jnp.isfinite(overlaps_new)
            | (jnp.abs(overlaps_new) <= overlap_floor)
            | (jnp.abs(preliminary_weights) > w_cap)
        )
        preliminary_weights = jnp.where(invalid, 0.0 + 0.0j, preliminary_weights)
        node_encounters_new = state.node_encounters + jnp.sum(invalid)
        weights_new = coefficient_space_global_phaseless_projection(
            preliminary_weights,
            overlaps_new,
            dt=prop_ctx.dt,
            budget_scale=float(getattr(params, "global_phaseless_budget_scale", 1.0)),
            overlap_floor=overlap_floor,
            gauge_fix=bool(getattr(params, "global_phaseless_gauge_fix", False)),
        )
    else:
        theta = jnp.angle(jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio)
        imp_ph = jnp.abs(imp_fun) * jnp.cos(theta)

        imp_ph = jnp.where(~jnp.isfinite(imp_ph) | (imp_ph < w_floor), 0.0, imp_ph)
        node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
        imp_ph = jnp.where(imp_ph > w_cap, 0.0, imp_ph)

        weights_new = state.weights * imp_ph
        weights_new = jnp.where(weights_new > w_cap, 0.0, weights_new)

    damping = float(getattr(params, "pop_control_damping", 0.1))
    avg_w = jnp.clip(jnp.mean(jnp.abs(weights_new)), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    return PropState(
        walkers=walkers_new,
        weights=weights_new,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
    )


def make_prop_ops(ham_basis: HamBasis, walker_kind: str, mixed_precision=False) -> PropOps:
    trotter_ops = make_trotter_ops(ham_basis, walker_kind, mixed_precision=mixed_precision)

    def step(
        state: PropState,
        *,
        params: QmcParamsBase,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState:
        return afqmc_step(
            state,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,
        )

    def build_prop_ctx(ham_data: Any, rdm1: jax.Array, params: QmcParamsBase) -> CholAfqmcCtx:
        return _build_prop_ctx(
            ham_data,
            rdm1,
            params.dt,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
        )

    return PropOps(init_prop_state=init_prop_state, build_prop_ctx=build_prop_ctx, step=step)
