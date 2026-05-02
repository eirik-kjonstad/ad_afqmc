from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, k_force_bias, k_force_bias_charge_sz
from ..core.system import System
from ..ham.chol import HamBasis, HamChol
from ..sharding import shard_prop_state
from ..walkers import init_walkers
from .chol_afqmc_ops import CholAfqmcCtx, TrotterOps, _build_prop_ctx, make_trotter_ops
from .types import PropOps, PropState, QmcParamsBase


class AfqmcStepDiagnostics(NamedTuple):
    fields: jax.Array
    force_bias: jax.Array
    field_shifts: jax.Array
    shifted_fields: jax.Array
    shift_term: jax.Array
    fb_term: jax.Array
    overlap_ratio: jax.Array
    exponent: jax.Array
    weight_multiplier: jax.Array
    theta: jax.Array
    node_mask: jax.Array


def _charge_sz_couplings(dtype: jnp.dtype) -> jax.Array:
    inv_sqrt2 = 1.0 / jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    return jnp.asarray([1.0j, 1.0j, 1.0j * inv_sqrt2, inv_sqrt2])


def _force_bias_kernel_name(meas_ops: MeasOps, hs_decomposition: str) -> str:
    if hs_decomposition != "charge_sz":
        return k_force_bias
    if meas_ops.has_kernel(k_force_bias_charge_sz):
        return k_force_bias_charge_sz
    raise ValueError(
        "charge_sz HS decomposition requires a four-channel force_bias_charge_sz kernel."
    )


def _draw_fields(key: jax.Array, *, n_walkers: int, n_chol: int, hs_decomposition: str) -> jax.Array:
    if hs_decomposition == "charge_sz":
        return jax.random.normal(key, (n_walkers, n_chol, 4))
    return jax.random.normal(key, (n_walkers, n_chol))


def _make_field_shifts(force_bias: jax.Array, prop_ctx: CholAfqmcCtx) -> jax.Array:
    if prop_ctx.hs_decomposition == "charge_sz":
        if force_bias.ndim != 3 or force_bias.shape[-1] != 4:
            raise ValueError(
                "charge_sz force bias must have shape (n_walkers, n_chol, 4)."
            )
        couplings = _charge_sz_couplings(prop_ctx.sqrt_dt.dtype)
        return -prop_ctx.sqrt_dt * (couplings * force_bias - prop_ctx.mf_shifts)
    return -prop_ctx.sqrt_dt * (1.0j * force_bias - prop_ctx.mf_shifts)


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
    n_chol = prop_ctx.chol_flat.shape[0]
    fields = _draw_fields(
        subkey,
        n_walkers=nw,
        n_chol=n_chol,
        hs_decomposition=prop_ctx.hs_decomposition,
    )

    fb_name = _force_bias_kernel_name(meas_ops, prop_ctx.hs_decomposition)
    fb_kernel = meas_ops.require_kernel(fb_name)
    force_bias = wk.vmap_chunked(
        fb_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data)
    field_shifts = _make_field_shifts(force_bias, prop_ctx)
    shifted_fields = fields - field_shifts

    shift_term = jnp.sum(shifted_fields * prop_ctx.mf_shifts, axis=tuple(range(1, fields.ndim)))
    fb_term = jnp.sum(
        fields * field_shifts - 0.5 * field_shifts * field_shifts,
        axis=tuple(range(1, fields.ndim)),
    )
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


def afqmc_step_diagnostics(
    state: PropState,
    *,
    params: QmcParamsBase,
    ham_data: HamChol,
    trial_data: Any,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: CholAfqmcCtx,
    meas_ctx: Any,
) -> AfqmcStepDiagnostics:
    """
    Evaluate one AFQMC step without updating state and return debug terms.
    """
    _, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    n_chol = prop_ctx.chol_flat.shape[0]
    fields = _draw_fields(
        subkey,
        n_walkers=nw,
        n_chol=n_chol,
        hs_decomposition=prop_ctx.hs_decomposition,
    )

    fb_kernel = meas_ops.require_kernel(_force_bias_kernel_name(meas_ops, prop_ctx.hs_decomposition))
    force_bias = wk.vmap_chunked(
        fb_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, meas_ctx, trial_data)
    field_shifts = _make_field_shifts(force_bias, prop_ctx)
    shifted_fields = fields - field_shifts

    reduce_axes = tuple(range(1, fields.ndim))
    shift_term = jnp.sum(shifted_fields * prop_ctx.mf_shifts, axis=reduce_axes)
    fb_term = jnp.sum(fields * field_shifts - 0.5 * field_shifts * field_shifts, axis=reduce_axes)
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
    weight_multiplier = jnp.abs(imp_fun) * jnp.cos(theta)
    node_mask = ~jnp.isfinite(weight_multiplier) | (
        weight_multiplier < float(getattr(params, "weight_floor", 1.0e-3))
    )

    return AfqmcStepDiagnostics(
        fields=fields,
        force_bias=force_bias,
        field_shifts=field_shifts,
        shifted_fields=shifted_fields,
        shift_term=shift_term,
        fb_term=fb_term,
        overlap_ratio=ratio,
        exponent=exponent,
        weight_multiplier=weight_multiplier,
        theta=theta,
        node_mask=node_mask,
    )


def make_prop_ops(
    ham_basis: HamBasis,
    walker_kind: str,
    mixed_precision=False,
    hs_decomposition: str = "charge",
) -> PropOps:
    trotter_ops = make_trotter_ops(
        ham_basis,
        walker_kind,
        mixed_precision=mixed_precision,
        hs_decomposition=hs_decomposition,
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

    def build_prop_ctx(ham_data: Any, rdm1: jax.Array, params: QmcParamsBase) -> CholAfqmcCtx:
        return _build_prop_ctx(
            ham_data,
            rdm1,
            params.dt,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
            hs_decomposition=hs_decomposition,
        )

    return PropOps(init_prop_state=init_prop_state, build_prop_ctx=build_prop_ctx, step=step)
