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
from .chol_afqmc_ops import (
    CholAfqmcCtx,
    CholDecomposition,
    TrotterOps,
    _build_prop_ctx,
    make_trotter_ops,
)
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

    theta = jnp.angle(jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio)
    imp_ph = jnp.abs(imp_fun) * jnp.cos(theta)

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
    )


def _spin_channel_norm_diagnostics(
    name: str,
    values: jax.Array,
    prop_ctx: CholAfqmcCtx,
) -> dict[str, jax.Array]:
    dtype = jnp.real(values).dtype
    if prop_ctx.decomposition not in ("spin", "spin_null"):
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        return {
            f"{name}_charge_mean": nan,
            f"{name}_charge_max": nan,
            f"{name}_minus_mean": nan,
            f"{name}_minus_max": nan,
            f"{name}_null_mean": nan,
            f"{name}_null_max": nan,
            f"{name}_alpha_mean": nan,
            f"{name}_alpha_max": nan,
            f"{name}_beta_mean": nan,
            f"{name}_beta_max": nan,
            f"{name}_s_mean": nan,
            f"{name}_s_max": nan,
        }

    if prop_ctx.spin_decomposition_lambda >= 1.0:
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        n_chol = values.shape[1] // 3
        charge = jnp.full((values.shape[0],), nan, dtype=dtype)
        if prop_ctx.decomposition == "spin_null":
            charge = jnp.linalg.norm(values[:, :n_chol], axis=1)
            minus = jnp.linalg.norm(values[:, n_chol : 2 * n_chol], axis=1)
            null = jnp.linalg.norm(values[:, 2 * n_chol :], axis=1)
            alpha = jnp.full((values.shape[0],), nan, dtype=dtype)
            beta = jnp.full((values.shape[0],), nan, dtype=dtype)
            spin = jnp.full((values.shape[0],), nan, dtype=dtype)
        else:
            minus = jnp.full((values.shape[0],), nan, dtype=dtype)
            null = jnp.full((values.shape[0],), nan, dtype=dtype)
            alpha = jnp.linalg.norm(values[:, :n_chol], axis=1)
            beta = jnp.linalg.norm(values[:, n_chol : 2 * n_chol], axis=1)
            spin = jnp.linalg.norm(values[:, 2 * n_chol :], axis=1)
    else:
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        n_chol = values.shape[1] // 4
        charge = jnp.linalg.norm(values[:, :n_chol], axis=1)
        minus = jnp.full((values.shape[0],), nan, dtype=dtype)
        null = jnp.full((values.shape[0],), nan, dtype=dtype)
        alpha = jnp.linalg.norm(values[:, n_chol : 2 * n_chol], axis=1)
        beta = jnp.linalg.norm(values[:, 2 * n_chol : 3 * n_chol], axis=1)
        spin = jnp.linalg.norm(values[:, 3 * n_chol :], axis=1)

    return {
        f"{name}_charge_mean": jnp.mean(charge),
        f"{name}_charge_max": jnp.max(charge),
        f"{name}_minus_mean": jnp.mean(minus),
        f"{name}_minus_max": jnp.max(minus),
        f"{name}_null_mean": jnp.mean(null),
        f"{name}_null_max": jnp.max(null),
        f"{name}_alpha_mean": jnp.mean(alpha),
        f"{name}_alpha_max": jnp.max(alpha),
        f"{name}_beta_mean": jnp.mean(beta),
        f"{name}_beta_max": jnp.max(beta),
        f"{name}_s_mean": jnp.mean(spin),
        f"{name}_s_max": jnp.max(spin),
    }


def afqmc_step_with_diagnostics(
    state: PropState,
    *,
    params: QmcParamsBase,
    ham_data: HamChol,
    trial_data: Any,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: CholAfqmcCtx,
    meas_ctx: Any,
) -> tuple[PropState, dict[str, jax.Array]]:

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

    phase_factor = jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio
    theta = jnp.angle(phase_factor)
    cos_theta = jnp.cos(theta)
    imp_raw = jnp.abs(imp_fun) * cos_theta

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    nonfinite = ~jnp.isfinite(imp_raw)
    floor_kill = (~nonfinite) & (imp_raw < w_floor)
    imp_ph = jnp.where(nonfinite | floor_kill, 0.0, imp_raw)
    cap_kill = imp_ph > w_cap
    node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
    imp_ph = jnp.where(cap_kill, 0.0, imp_ph)

    weights_unbounded = state.weights * imp_ph
    weight_cap_kill = weights_unbounded > w_cap
    weights_new = jnp.where(weight_cap_kill, 0.0, weights_unbounded)

    damping = float(getattr(params, "pop_control_damping", 0.1))
    avg_w = jnp.clip(jnp.mean(weights_new), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    new_state = PropState(
        walkers=walkers_new,
        weights=weights_new,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
    )

    force_bias_norm = jnp.linalg.norm(force_bias, axis=1)
    field_shift_norm = jnp.linalg.norm(field_shifts, axis=1)
    shifted_field_norm = jnp.linalg.norm(shifted_fields, axis=1)
    abs_ratio = jnp.abs(ratio)
    abs_phase_factor = jnp.abs(phase_factor)

    diagnostics = {
        "n_cos_nonpositive": jnp.sum(cos_theta <= 0.0),
        "n_floor": jnp.sum(floor_kill),
        "n_nonfinite": jnp.sum(nonfinite),
        "n_imp_cap": jnp.sum(cap_kill),
        "n_weight_cap": jnp.sum(weight_cap_kill),
        "theta_abs_mean": jnp.mean(jnp.abs(theta)),
        "theta_abs_max": jnp.max(jnp.abs(theta)),
        "cos_mean": jnp.mean(cos_theta),
        "cos_min": jnp.min(cos_theta),
        "imp_raw_mean": jnp.mean(imp_raw),
        "imp_raw_min": jnp.min(imp_raw),
        "imp_raw_max": jnp.max(imp_raw),
        "abs_ratio_mean": jnp.mean(abs_ratio),
        "abs_ratio_max": jnp.max(abs_ratio),
        "abs_phase_factor_mean": jnp.mean(abs_phase_factor),
        "abs_phase_factor_max": jnp.max(abs_phase_factor),
        "force_bias_norm_mean": jnp.mean(force_bias_norm),
        "force_bias_norm_max": jnp.max(force_bias_norm),
        "field_shift_norm_mean": jnp.mean(field_shift_norm),
        "field_shift_norm_max": jnp.max(field_shift_norm),
        "shifted_field_norm_mean": jnp.mean(shifted_field_norm),
        "shifted_field_norm_max": jnp.max(shifted_field_norm),
        "weight_sum_before": jnp.sum(state.weights),
        "weight_sum_after": jnp.sum(weights_new),
        "overlap_abs_mean_before": jnp.mean(jnp.abs(state.overlaps)),
        "overlap_abs_mean_after": jnp.mean(jnp.abs(overlaps_new)),
    }
    diagnostics.update(_spin_channel_norm_diagnostics("force_bias_norm", force_bias, prop_ctx))
    diagnostics.update(_spin_channel_norm_diagnostics("field_shift_norm", field_shifts, prop_ctx))
    diagnostics.update(
        _spin_channel_norm_diagnostics("shifted_field_norm", shifted_fields, prop_ctx)
    )
    return new_state, diagnostics


def make_prop_ops(
    ham_basis: HamBasis,
    walker_kind: str,
    mixed_precision=False,
    decomposition: CholDecomposition = "charge",
    spin_decomposition_lambda: float = 1.0,
    spin_null_eta: float = 0.0,
) -> PropOps:
    trotter_ops = make_trotter_ops(
        ham_basis,
        walker_kind,
        mixed_precision=mixed_precision,
        decomposition=decomposition,
        spin_decomposition_lambda=spin_decomposition_lambda,
        spin_null_eta=spin_null_eta,
    )

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

    def step_diagnostics(
        state: PropState,
        *,
        params: QmcParamsBase,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> tuple[PropState, dict[str, jax.Array]]:
        return afqmc_step_with_diagnostics(
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
            decomposition=decomposition,
            spin_decomposition_lambda=spin_decomposition_lambda,
            spin_null_eta=spin_null_eta,
        )

    return PropOps(
        init_prop_state=init_prop_state,
        build_prop_ctx=build_prop_ctx,
        step=step,
        step_diagnostics=step_diagnostics,
    )
