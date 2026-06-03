import jax
import jax.numpy as jnp
import pytest

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import _build_prop_ctx


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


def test_build_spin_prop_ctx_shapes_and_matches_charge_scalars():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=1.5)

    dm_a = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    dm_b = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    dm = jnp.stack([dm_a, dm_b], axis=0)
    dt = 0.2

    ctx_charge = _build_prop_ctx(ham, dm, dt)
    ctx_spin = _build_prop_ctx(ham, dm, dt, decomposition="spin")

    assert ctx_spin.mf_shifts.shape == (3 * n_fields,)
    assert ctx_spin.chol_flat.shape == (3 * n_fields, norb * norb)
    assert ctx_spin.exp_h1_half.shape == (norb, norb)
    assert jnp.allclose(ctx_spin.h0_prop, ctx_charge.h0_prop)
    assert jnp.allclose(ctx_spin.exp_h1_half, ctx_charge.exp_h1_half)


def test_build_interpolated_spin_prop_ctx_shapes_and_matches_charge_scalars():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=1.5)

    dm_a = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    dm_b = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    dm = jnp.stack([dm_a, dm_b], axis=0)
    dt = 0.2
    lam = 0.25

    ctx_charge = _build_prop_ctx(ham, dm, dt)
    ctx_spin = _build_prop_ctx(
        ham,
        dm,
        dt,
        decomposition="spin",
        spin_decomposition_lambda=lam,
    )

    assert ctx_spin.spin_decomposition_lambda == lam
    assert ctx_spin.mf_shifts.shape == (4 * n_fields,)
    assert ctx_spin.chol_flat.shape == (4 * n_fields, norb * norb)
    assert ctx_spin.exp_h1_half.shape == (norb, norb)
    assert jnp.allclose(ctx_spin.h0_prop, ctx_charge.h0_prop)
    assert jnp.allclose(ctx_spin.exp_h1_half, ctx_charge.exp_h1_half)


def test_build_spin_null_prop_ctx_shapes_and_matches_charge_scalars():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=1.5)

    dm_a = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    dm_b = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    dm = jnp.stack([dm_a, dm_b], axis=0)
    dt = 0.2
    eta = 0.75

    ctx_charge = _build_prop_ctx(ham, dm, dt)
    ctx_null = _build_prop_ctx(
        ham,
        dm,
        dt,
        decomposition="spin_null",
        spin_null_eta=eta,
    )

    assert ctx_null.spin_null_eta == eta
    assert ctx_null.mf_shifts.shape == (3 * n_fields,)
    assert ctx_null.chol_flat.shape == (3 * n_fields, norb * norb)
    assert ctx_null.exp_h1_half.shape == (norb, norb)
    assert jnp.allclose(ctx_null.h0_prop, ctx_charge.h0_prop)
    assert jnp.allclose(ctx_null.exp_h1_half, ctx_charge.exp_h1_half)


if __name__ == "__main__":
    pytest.main([__file__])
