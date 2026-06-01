import jax
import jax.numpy as jnp
import pytest

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops


def _make_small_ham(*, norb=4, n_fields=3, h0=0.0, seed=0):
    key = jax.random.PRNGKey(seed)

    a = jax.random.normal(key, (norb, norb))
    h1 = 0.1 * (a + a.T)

    key, sub = jax.random.split(key)
    chol = 0.05 * jax.random.normal(sub, (n_fields, norb, norb))

    return HamChol(basis="restricted", h0=jnp.asarray(h0), h1=h1, chol=chol)


def test_build_prop_ctx_shapes_and_nfields():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)

    dm = jnp.zeros((norb, norb))
    dt = 0.2
    ctx = _build_prop_ctx(ham, dm, dt)

    assert ctx.mf_shifts.shape == (n_fields,)
    assert ctx.exp_h1_half.shape == (norb, norb)
    assert ctx.dt.shape == ()
    assert ctx.sqrt_dt.shape == ()
    assert ctx.h0_prop.shape == ()


def test_build_prop_ctx_spin_decomposition_shapes_and_charge_mf_invariants():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.3)

    key = jax.random.PRNGKey(11)
    dm = 0.1 * jax.random.normal(key, (2, norb, norb))
    dt = 0.2

    ctx_charge = _build_prop_ctx(ham, dm, dt)
    ctx_spin = _build_prop_ctx(ham, dm, dt, hs_decomposition="spin")

    l_a = jnp.einsum("gij,ji->g", ham.chol, dm[0], optimize="optimal")
    l_b = jnp.einsum("gij,ji->g", ham.chol, dm[1], optimize="optimal")
    rt2 = jnp.sqrt(jnp.asarray(2.0, dtype=l_a.dtype))
    mf_spin_ref = jnp.concatenate([1.0j * rt2 * l_a, 1.0j * rt2 * l_b, -(l_a - l_b)])
    scales_ref = jnp.concatenate(
        [
            1.0j * rt2 * jnp.ones((n_fields,), dtype=ctx_spin.force_bias_scales.dtype),
            1.0j * rt2 * jnp.ones((n_fields,), dtype=ctx_spin.force_bias_scales.dtype),
            -jnp.ones((n_fields,), dtype=ctx_spin.force_bias_scales.dtype),
        ]
    )

    assert ctx_spin.mf_shifts.shape == (3 * n_fields,)
    assert ctx_spin.force_bias_scales.shape == (3 * n_fields,)
    assert ctx_spin.chol_flat.shape == ctx_charge.chol_flat.shape
    assert jnp.allclose(ctx_spin.mf_shifts, mf_spin_ref)
    assert jnp.allclose(ctx_spin.force_bias_scales, scales_ref)
    assert jnp.allclose(jnp.sum(ctx_spin.mf_shifts**2), jnp.sum(ctx_charge.mf_shifts**2))
    assert jnp.allclose(ctx_spin.h0_prop, ctx_charge.h0_prop)
    assert jnp.allclose(ctx_spin.exp_h1_half, ctx_charge.exp_h1_half)


def test_make_trotter_ops_spin_decomposition_is_unrestricted_only():
    ops = make_trotter_ops("restricted", "unrestricted", hs_decomposition="spin")
    assert ops.apply_trotter_split is not None
    assert ops.apply_trotter_a is not None
    assert ops.apply_trotter_b is not None
    assert ops.apply_trotter_s is not None

    with pytest.raises(NotImplementedError):
        make_trotter_ops("restricted", "restricted", hs_decomposition="spin")


if __name__ == "__main__":
    pytest.main([__file__])
