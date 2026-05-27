import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.prop.utils import taylor_expm_action


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
    assert ctx.force_bias_scales.shape == (n_fields,)


def test_build_prop_ctx_spin_shapes_and_mean_field_constant():
    norb, n_fields = 4, 5
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)
    dm_a = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))
    dm_b = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
    dm = jnp.stack([dm_a, dm_b], axis=0)

    charge_ctx = _build_prop_ctx(ham, dm, 0.2)
    spin_ctx = _build_prop_ctx(ham, dm, 0.2, hs_decomposition="spin")

    assert spin_ctx.mf_shifts.shape == (3 * n_fields,)
    assert spin_ctx.force_bias_scales.shape == (3 * n_fields,)
    assert spin_ctx.chol_flat.shape == (n_fields, norb * norb)
    assert jnp.allclose(jnp.sum(spin_ctx.mf_shifts**2), jnp.sum(charge_ctx.mf_shifts**2))
    assert jnp.allclose(spin_ctx.h0_prop, charge_ctx.h0_prop)


def test_spin_trotter_applies_opposite_spin_channel_signs():
    norb, n_fields = 3, 2
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)
    ham = HamChol(
        basis="restricted",
        h0=ham.h0,
        h1=jnp.zeros_like(ham.h1),
        chol=ham.chol,
    )
    dm = jnp.zeros((2, norb, norb))
    dt = 0.05
    ctx = _build_prop_ctx(ham, dm, dt, hs_decomposition="spin")
    ctx = replace(ctx, exp_h1_half=jnp.eye(norb, dtype=ctx.exp_h1_half.dtype))
    ops = make_trotter_ops("restricted", "unrestricted", hs_decomposition="spin")

    wu = jnp.eye(norb, 1, dtype=jnp.complex128)
    wd = 2.0 * jnp.eye(norb, 1, dtype=jnp.complex128)
    field_s = jnp.array([0.2, -0.3])
    field = jnp.concatenate([jnp.zeros(n_fields), jnp.zeros(n_fields), field_s])

    out_u, out_d = ops.apply_trotter((wu, wd), field, ctx, 8)
    spin_mat = jnp.einsum("gij,g->ij", ham.chol, field_s, optimize="optimal")

    exp_u = taylor_expm_action(jnp.sqrt(jnp.asarray(dt)), -spin_mat, wu, 8)
    exp_d = taylor_expm_action(jnp.sqrt(jnp.asarray(dt)), spin_mat, wd, 8)

    assert jnp.allclose(out_u, exp_u)
    assert jnp.allclose(out_d, exp_d)


if __name__ == "__main__":
    pytest.main([__file__])
