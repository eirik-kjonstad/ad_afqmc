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


def _charge_spin_pivot_masks(ham_data: HamChol) -> tuple[jax.Array, jax.Array] | None:
    if ham_data.basis != "charge_spin" or ham_data.pivot_channels is None:
        return None
    channels = tuple(ham_data.pivot_channels)
    if "charge" not in channels and "spin" not in channels:
        return None
    charge_mask = jnp.asarray([channel == "charge" for channel in channels])
    spin_mask = jnp.asarray([channel == "spin" for channel in channels])
    return charge_mask, spin_mask


def _empty_charge_spin_pivot_diagnostics(ham_data: HamChol) -> dict[str, jax.Array] | None:
    masks = _charge_spin_pivot_masks(ham_data)
    if masks is None:
        return None
    zero = jnp.asarray(0.0)
    one = jnp.asarray(1.0)
    return {
        "force_bias_norm_charge_pivot_mean": zero,
        "force_bias_norm_charge_pivot_max": zero,
        "force_bias_norm_spin_pivot_mean": zero,
        "force_bias_norm_spin_pivot_max": zero,
        "force_bias_norm_per_pivot_charge_pivot_mean": zero,
        "force_bias_norm_per_pivot_charge_pivot_max": zero,
        "force_bias_norm_per_pivot_spin_pivot_mean": zero,
        "force_bias_norm_per_pivot_spin_pivot_max": zero,
        "field_shift_norm_charge_pivot_mean": zero,
        "field_shift_norm_charge_pivot_max": zero,
        "field_shift_norm_spin_pivot_mean": zero,
        "field_shift_norm_spin_pivot_max": zero,
        "field_shift_norm_per_pivot_charge_pivot_mean": zero,
        "field_shift_norm_per_pivot_charge_pivot_max": zero,
        "field_shift_norm_per_pivot_spin_pivot_mean": zero,
        "field_shift_norm_per_pivot_spin_pivot_max": zero,
        "shifted_field_norm_charge_pivot_mean": zero,
        "shifted_field_norm_charge_pivot_max": zero,
        "shifted_field_norm_spin_pivot_mean": zero,
        "shifted_field_norm_spin_pivot_max": zero,
        "shifted_field_norm_per_pivot_charge_pivot_mean": zero,
        "shifted_field_norm_per_pivot_charge_pivot_max": zero,
        "shifted_field_norm_per_pivot_spin_pivot_mean": zero,
        "shifted_field_norm_per_pivot_spin_pivot_max": zero,
        "field_phase_abs_charge_pivot_mean": zero,
        "field_phase_abs_charge_pivot_max": zero,
        "field_phase_abs_spin_pivot_mean": zero,
        "field_phase_abs_spin_pivot_max": zero,
        "spin_pivot_field_shift_cap": zero,
        "spin_pivot_field_shift_cap_n_applied": zero,
        "spin_pivot_field_shift_cap_fraction": zero,
        "spin_pivot_field_shift_cap_scale_min": one,
        "spin_pivot_field_shift_cap_excess_mean": zero,
        "spin_pivot_field_shift_cap_excess_max": zero,
        "field_shift_uncapped_norm_spin_pivot_mean": zero,
        "field_shift_uncapped_norm_spin_pivot_max": zero,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_mean": zero,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_max": zero,
        "n_floor": zero,
        "n_nonfinite": zero,
        "n_imp_cap": zero,
        "n_weight_cap": zero,
        "n_node_encounters": zero,
        "abs_ratio_mean": zero,
        "abs_ratio_max": zero,
        "imp_raw_mean": zero,
        "imp_raw_min": zero,
        "imp_raw_max": zero,
    }


def _masked_vector_norm(values: jax.Array, mask: jax.Array) -> jax.Array:
    masked = jnp.where(mask[None, :], values, 0.0)
    return jnp.sqrt(jnp.sum(jnp.abs(masked) ** 2, axis=1))


def _mean_max(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.mean(values), jnp.max(values)


def _per_pivot(values: jax.Array, mask: jax.Array) -> jax.Array:
    n_pivots = jnp.sum(mask)
    denom = jnp.sqrt(jnp.asarray(n_pivots, dtype=values.dtype))
    return jnp.where(n_pivots > 0, values / denom, 0.0)


def _empty_spin_pivot_field_shift_cap_diagnostics(dtype: Any = jnp.float64) -> dict[str, jax.Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    return {
        "spin_pivot_field_shift_cap": zero,
        "spin_pivot_field_shift_cap_n_applied": zero,
        "spin_pivot_field_shift_cap_fraction": zero,
        "spin_pivot_field_shift_cap_scale_min": one,
        "spin_pivot_field_shift_cap_excess_mean": zero,
        "spin_pivot_field_shift_cap_excess_max": zero,
        "field_shift_uncapped_norm_spin_pivot_mean": zero,
        "field_shift_uncapped_norm_spin_pivot_max": zero,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_mean": zero,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_max": zero,
    }


def _cap_spin_pivot_field_shifts(
    field_shifts: jax.Array,
    spin_mask: jax.Array,
    cap: float | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dtype = field_shifts.real.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    spin_norm = _masked_vector_norm(field_shifts, spin_mask)
    uncapped_pp = _per_pivot(spin_norm, spin_mask)
    uncapped_pp_mean, uncapped_pp_max = _mean_max(uncapped_pp)
    uncapped_mean, uncapped_max = _mean_max(spin_norm)

    if cap is None or cap <= 0.0:
        diagnostics = {
            "spin_pivot_field_shift_cap": zero,
            "spin_pivot_field_shift_cap_n_applied": zero,
            "spin_pivot_field_shift_cap_fraction": zero,
            "spin_pivot_field_shift_cap_scale_min": one,
            "spin_pivot_field_shift_cap_excess_mean": zero,
            "spin_pivot_field_shift_cap_excess_max": zero,
            "field_shift_uncapped_norm_spin_pivot_mean": uncapped_mean,
            "field_shift_uncapped_norm_spin_pivot_max": uncapped_max,
            "field_shift_uncapped_norm_per_pivot_spin_pivot_mean": uncapped_pp_mean,
            "field_shift_uncapped_norm_per_pivot_spin_pivot_max": uncapped_pp_max,
        }
        return field_shifts, diagnostics

    cap_value = jnp.asarray(cap, dtype=dtype)
    has_spin_pivots = jnp.sum(spin_mask) > 0
    over_cap = has_spin_pivots & (spin_norm > cap_value)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    scale = jnp.where(over_cap, cap_value / jnp.maximum(spin_norm, tiny), 1.0)
    capped = jnp.where(
        spin_mask[None, :],
        field_shifts * scale[:, None],
        field_shifts,
    )
    n_applied = jnp.asarray(jnp.sum(over_cap), dtype=dtype)
    excess = jnp.where(over_cap, spin_norm - cap_value, 0.0)
    diagnostics = {
        "spin_pivot_field_shift_cap": cap_value,
        "spin_pivot_field_shift_cap_n_applied": n_applied,
        "spin_pivot_field_shift_cap_fraction": n_applied
        / jnp.asarray(field_shifts.shape[0], dtype=dtype),
        "spin_pivot_field_shift_cap_scale_min": jnp.min(scale),
        "spin_pivot_field_shift_cap_excess_mean": jnp.mean(excess),
        "spin_pivot_field_shift_cap_excess_max": jnp.max(excess),
        "field_shift_uncapped_norm_spin_pivot_mean": uncapped_mean,
        "field_shift_uncapped_norm_spin_pivot_max": uncapped_max,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_mean": uncapped_pp_mean,
        "field_shift_uncapped_norm_per_pivot_spin_pivot_max": uncapped_pp_max,
    }
    return capped, diagnostics


def _apply_spin_pivot_field_shift_cap(
    ham_data: HamChol,
    field_shifts: jax.Array,
    cap: float | None,
) -> tuple[jax.Array, dict[str, jax.Array] | None]:
    masks = _charge_spin_pivot_masks(ham_data)
    if masks is None:
        return field_shifts, None
    _, spin_mask = masks
    return _cap_spin_pivot_field_shifts(field_shifts, spin_mask, cap)


def _charge_spin_pivot_diagnostics(
    ham_data: HamChol,
    *,
    force_bias: jax.Array,
    field_shifts: jax.Array,
    shifted_fields: jax.Array,
    mf_shifts: jax.Array,
    sqrt_dt: jax.Array,
) -> dict[str, jax.Array] | None:
    masks = _charge_spin_pivot_masks(ham_data)
    if masks is None:
        return None
    charge_mask, spin_mask = masks

    fb_charge = _masked_vector_norm(force_bias, charge_mask)
    fb_spin = _masked_vector_norm(force_bias, spin_mask)
    shift_charge = _masked_vector_norm(field_shifts, charge_mask)
    shift_spin = _masked_vector_norm(field_shifts, spin_mask)
    shifted_charge = _masked_vector_norm(shifted_fields, charge_mask)
    shifted_spin = _masked_vector_norm(shifted_fields, spin_mask)

    fb_charge_mean, fb_charge_max = _mean_max(fb_charge)
    fb_spin_mean, fb_spin_max = _mean_max(fb_spin)
    fb_charge_pp_mean, fb_charge_pp_max = _mean_max(_per_pivot(fb_charge, charge_mask))
    fb_spin_pp_mean, fb_spin_pp_max = _mean_max(_per_pivot(fb_spin, spin_mask))
    shift_charge_mean, shift_charge_max = _mean_max(shift_charge)
    shift_spin_mean, shift_spin_max = _mean_max(shift_spin)
    shift_charge_pp_mean, shift_charge_pp_max = _mean_max(_per_pivot(shift_charge, charge_mask))
    shift_spin_pp_mean, shift_spin_pp_max = _mean_max(_per_pivot(shift_spin, spin_mask))
    shifted_charge_mean, shifted_charge_max = _mean_max(shifted_charge)
    shifted_spin_mean, shifted_spin_max = _mean_max(shifted_spin)
    shifted_charge_pp_mean, shifted_charge_pp_max = _mean_max(
        _per_pivot(shifted_charge, charge_mask)
    )
    shifted_spin_pp_mean, shifted_spin_pp_max = _mean_max(_per_pivot(shifted_spin, spin_mask))

    charge_phase = jnp.imag(
        -sqrt_dt
        * jnp.sum(
            jnp.where(charge_mask[None, :], shifted_fields, 0.0) * mf_shifts[None, :],
            axis=1,
        )
    )
    spin_phase = jnp.imag(
        -sqrt_dt
        * jnp.sum(
            jnp.where(spin_mask[None, :], shifted_fields, 0.0) * mf_shifts[None, :],
            axis=1,
        )
    )
    phase_charge_mean, phase_charge_max = _mean_max(jnp.abs(charge_phase))
    phase_spin_mean, phase_spin_max = _mean_max(jnp.abs(spin_phase))

    return {
        "force_bias_norm_charge_pivot_mean": fb_charge_mean,
        "force_bias_norm_charge_pivot_max": fb_charge_max,
        "force_bias_norm_spin_pivot_mean": fb_spin_mean,
        "force_bias_norm_spin_pivot_max": fb_spin_max,
        "force_bias_norm_per_pivot_charge_pivot_mean": fb_charge_pp_mean,
        "force_bias_norm_per_pivot_charge_pivot_max": fb_charge_pp_max,
        "force_bias_norm_per_pivot_spin_pivot_mean": fb_spin_pp_mean,
        "force_bias_norm_per_pivot_spin_pivot_max": fb_spin_pp_max,
        "field_shift_norm_charge_pivot_mean": shift_charge_mean,
        "field_shift_norm_charge_pivot_max": shift_charge_max,
        "field_shift_norm_spin_pivot_mean": shift_spin_mean,
        "field_shift_norm_spin_pivot_max": shift_spin_max,
        "field_shift_norm_per_pivot_charge_pivot_mean": shift_charge_pp_mean,
        "field_shift_norm_per_pivot_charge_pivot_max": shift_charge_pp_max,
        "field_shift_norm_per_pivot_spin_pivot_mean": shift_spin_pp_mean,
        "field_shift_norm_per_pivot_spin_pivot_max": shift_spin_pp_max,
        "shifted_field_norm_charge_pivot_mean": shifted_charge_mean,
        "shifted_field_norm_charge_pivot_max": shifted_charge_max,
        "shifted_field_norm_spin_pivot_mean": shifted_spin_mean,
        "shifted_field_norm_spin_pivot_max": shifted_spin_max,
        "shifted_field_norm_per_pivot_charge_pivot_mean": shifted_charge_pp_mean,
        "shifted_field_norm_per_pivot_charge_pivot_max": shifted_charge_pp_max,
        "shifted_field_norm_per_pivot_spin_pivot_mean": shifted_spin_pp_mean,
        "shifted_field_norm_per_pivot_spin_pivot_max": shifted_spin_pp_max,
        "field_phase_abs_charge_pivot_mean": phase_charge_mean,
        "field_phase_abs_charge_pivot_max": phase_charge_max,
        "field_phase_abs_spin_pivot_mean": phase_spin_mean,
        "field_phase_abs_spin_pivot_max": phase_spin_max,
    }


def _phaseless_event_diagnostics(
    *,
    ratio: jax.Array,
    imp_ph_raw: jax.Array,
    imp_after_imp_cap: jax.Array,
    weights_before: jax.Array,
    w_floor: float,
    w_cap: float,
) -> dict[str, jax.Array]:
    finite = jnp.isfinite(imp_ph_raw)
    weights_raw = weights_before * imp_after_imp_cap
    abs_ratio = jnp.abs(ratio)
    dtype = imp_ph_raw.dtype
    return {
        "n_floor": jnp.asarray(jnp.sum(finite & (imp_ph_raw < w_floor)), dtype=dtype),
        "n_nonfinite": jnp.asarray(jnp.sum(~finite), dtype=dtype),
        "n_imp_cap": jnp.asarray(jnp.sum(finite & (imp_ph_raw > w_cap)), dtype=dtype),
        "n_weight_cap": jnp.asarray(jnp.sum(weights_raw > w_cap), dtype=dtype),
        "n_node_encounters": jnp.asarray(jnp.sum(imp_after_imp_cap <= 0.0), dtype=dtype),
        "abs_ratio_mean": jnp.mean(abs_ratio),
        "abs_ratio_max": jnp.max(abs_ratio),
        "imp_raw_mean": jnp.mean(jnp.where(finite, imp_ph_raw, 0.0)),
        "imp_raw_min": jnp.min(jnp.where(finite, imp_ph_raw, 0.0)),
        "imp_raw_max": jnp.max(jnp.where(finite, imp_ph_raw, 0.0)),
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
    diagnostics = _empty_charge_spin_pivot_diagnostics(ham_data)

    state = PropState(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=pop_shift,
        e_estimate=e_est,
        node_encounters=node_encounters,
        diagnostics=diagnostics,
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
    field_shifts, cap_diagnostics = _apply_spin_pivot_field_shift_cap(
        ham_data,
        field_shifts,
        getattr(params, "spin_pivot_field_shift_cap", None),
    )
    shifted_fields = fields - field_shifts
    diagnostics = _charge_spin_pivot_diagnostics(
        ham_data,
        force_bias=force_bias,
        field_shifts=field_shifts,
        shifted_fields=shifted_fields,
        mf_shifts=prop_ctx.mf_shifts,
        sqrt_dt=prop_ctx.sqrt_dt,
    )
    if diagnostics is not None and cap_diagnostics is not None:
        diagnostics = {**diagnostics, **cap_diagnostics}

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
    imp_ph_raw = jnp.abs(imp_fun) * jnp.cos(theta)

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    imp_ph = jnp.where(~jnp.isfinite(imp_ph_raw) | (imp_ph_raw < w_floor), 0.0, imp_ph_raw)
    node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
    imp_ph = jnp.where(imp_ph > w_cap, 0.0, imp_ph)
    if diagnostics is not None:
        diagnostics = {
            **diagnostics,
            **_phaseless_event_diagnostics(
                ratio=ratio,
                imp_ph_raw=imp_ph_raw,
                imp_after_imp_cap=imp_ph,
                weights_before=state.weights,
                w_floor=w_floor,
                w_cap=w_cap,
            ),
        }

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
