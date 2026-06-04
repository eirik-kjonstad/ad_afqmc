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


def _field_partition_diagnostics(
    *,
    shifted_fields: jax.Array,
    mf_shifts: jax.Array,
    sqrt_dt: jax.Array,
    field_factors: jax.Array,
) -> dict[str, jax.Array]:
    real_mask = jnp.isclose(jnp.imag(field_factors), 0.0)
    complex_mask = ~real_mask
    real_shift = jnp.where(real_mask[None, :], shifted_fields, 0.0)
    complex_shift = jnp.where(complex_mask[None, :], shifted_fields, 0.0)
    phase_real = jnp.imag(-sqrt_dt * jnp.sum(real_shift * mf_shifts[None, :], axis=1))
    phase_complex = jnp.imag(-sqrt_dt * jnp.sum(complex_shift * mf_shifts[None, :], axis=1))
    real_norm = jnp.sqrt(jnp.sum(jnp.abs(real_shift) ** 2, axis=1))
    complex_norm = jnp.sqrt(jnp.sum(jnp.abs(complex_shift) ** 2, axis=1))
    return {
        "phase_abs_real_field_mean": jnp.mean(jnp.abs(phase_real)),
        "phase_abs_real_field_max": jnp.max(jnp.abs(phase_real)),
        "phase_abs_complex_field_mean": jnp.mean(jnp.abs(phase_complex)),
        "phase_abs_complex_field_max": jnp.max(jnp.abs(phase_complex)),
        "shifted_field_norm_real_mean": jnp.mean(real_norm),
        "shifted_field_norm_real_max": jnp.max(real_norm),
        "shifted_field_norm_complex_mean": jnp.mean(complex_norm),
        "shifted_field_norm_complex_max": jnp.max(complex_norm),
        "n_real_fields": jnp.asarray(jnp.sum(real_mask), dtype=jnp.float64),
        "n_complex_fields": jnp.asarray(jnp.sum(complex_mask), dtype=jnp.float64),
    }


def _empty_field_partition_diagnostics(dtype: Any = jnp.float64) -> dict[str, jax.Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    return {
        "phase_abs_real_field_mean": zero,
        "phase_abs_real_field_max": zero,
        "phase_abs_complex_field_mean": zero,
        "phase_abs_complex_field_max": zero,
        "shifted_field_norm_real_mean": zero,
        "shifted_field_norm_real_max": zero,
        "shifted_field_norm_complex_mean": zero,
        "shifted_field_norm_complex_max": zero,
        "n_real_fields": zero,
        "n_complex_fields": zero,
        "theta_abs_mean": zero,
        "theta_abs_max": zero,
        "cos_mean": zero,
        "cos_min": zero,
    }


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
    weights = jnp.ones((n_walkers,))

    if initial_walkers is None:
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=n_walkers)

    overlaps = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        initial_walkers, trial_data
    )

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
        diagnostics=_empty_field_partition_diagnostics(),
    )
    return shard_prop_state(state, mesh)


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
    field_shifts = -prop_ctx.sqrt_dt * (prop_ctx.field_factors * force_bias - prop_ctx.mf_shifts)
    shifted_fields = fields - field_shifts
    diagnostics = None
    if state.diagnostics is not None:
        diagnostics = _field_partition_diagnostics(
            shifted_fields=shifted_fields,
            mf_shifts=prop_ctx.mf_shifts,
            sqrt_dt=prop_ctx.sqrt_dt,
            field_factors=prop_ctx.field_factors,
        )

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

    theta = jnp.angle(jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio)
    imp_ph = jnp.abs(imp_fun) * jnp.cos(theta)
    if diagnostics is not None:
        diagnostics = {
            **diagnostics,
            "theta_abs_mean": jnp.mean(jnp.abs(theta)),
            "theta_abs_max": jnp.max(jnp.abs(theta)),
            "cos_mean": jnp.mean(jnp.cos(theta)),
            "cos_min": jnp.min(jnp.cos(theta)),
        }

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    imp_ph = jnp.where(~jnp.isfinite(imp_ph) | (imp_ph < w_floor), 0.0, imp_ph)
    node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
    imp_ph = jnp.where(imp_ph > w_cap, 0.0, imp_ph)

    weights_new = state.weights * imp_ph
    weights_new = jnp.where(weights_new > w_cap, 0.0, weights_new)

    damping = float(getattr(params, "pop_control_damping", 0.1))
    avg_w = jnp.clip(jnp.mean(weights_new), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    return PropState(
        walkers=walkers_new,
        weights=weights_new,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
        diagnostics=diagnostics,
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
