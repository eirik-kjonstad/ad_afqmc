from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpinDecompMeasCtx:
    base_ctx: Any

    def tree_flatten(self):
        return (self.base_ctx,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (base_ctx,) = children
        return cls(base_ctx=base_ctx)


def _force_bias_spin_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    trial_data: Any,
    *,
    overlap,
) -> jax.Array:
    wu, wd = walker
    chol = ham_data.chol
    n_chol = chol.shape[0]

    def f(x: jax.Array) -> jax.Array:
        x_a = x[:n_chol]
        x_b = x[n_chol:]
        v_a = jnp.einsum("gij,g->ij", chol, x_a, optimize="optimal")
        v_b = jnp.einsum("gij,g->ij", chol, x_b, optimize="optimal")
        return overlap((wu + v_a @ wu, wd + v_b @ wd), trial_data)

    x0 = jnp.zeros((2 * n_chol,), dtype=jnp.result_type(wu, wd, jnp.complex64))
    val, pullback = jax.vjp(f, x0)
    grad = pullback(jnp.asarray(1.0, dtype=val.dtype))[0] / val
    fb_a = grad[:n_chol]
    fb_b = grad[n_chol:]
    return jnp.concatenate([fb_a, fb_b, fb_a - fb_b], axis=0)


def wrap_spin_decomp_meas_ops(
    base_ops: MeasOps,
    *,
    sys: System,
    trial_kind: str,
) -> MeasOps:
    if sys.walker_kind != "unrestricted":
        raise NotImplementedError("Spin decomposition currently requires unrestricted walkers.")
    if trial_kind.lower() not in {"uhf", "ucisd", "ucisdt", "ucisdtq"}:
        raise NotImplementedError(
            "Spin decomposition currently supports UHF-family trials: "
            "uhf, ucisd, ucisdt, ucisdtq."
        )

    overlap = base_ops.overlap

    def build_ctx(ham_data: HamChol, trial_data: Any) -> SpinDecompMeasCtx:
        if ham_data.basis != "restricted":
            raise NotImplementedError(
                "Spin decomposition requires a restricted-basis Cholesky Hamiltonian."
            )
        return SpinDecompMeasCtx(base_ctx=base_ops.build_meas_ctx(ham_data, trial_data))

    def force_bias(walker, ham_data, meas_ctx: SpinDecompMeasCtx, trial_data):
        return _force_bias_spin_uw(walker, ham_data, trial_data, overlap=overlap)

    kernels = {}
    for name, kernel in base_ops.kernels.items():
        if name == k_force_bias:
            kernels[name] = force_bias
        else:
            kernels[name] = lambda walker, ham_data, meas_ctx, trial_data, kernel=kernel: kernel(
                walker, ham_data, meas_ctx.base_ctx, trial_data
            )

    observables = {
        name: (
            lambda walker, ham_data, meas_ctx, trial_data, kernel=kernel: kernel(
                walker, ham_data, meas_ctx.base_ctx, trial_data
            )
        )
        for name, kernel in base_ops.observables.items()
    }

    return MeasOps(
        overlap=overlap,
        build_meas_ctx=build_ctx,
        kernels=kernels,
        observables=observables,
    )
