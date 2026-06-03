import jax
import jax.numpy as jnp
import pytest

from trot.ham.chol import HamChol
from trot.meas.auto import make_auto_meas_ops
from trot.core.system import System
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops


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


def test_charge_spin_prop_ctx_shapes_and_spin_counterterms():
    norb, n_fields = 4, 5
    key = jax.random.PRNGKey(11)
    h1 = 0.1 * jax.random.normal(key, (2, norb, norb))
    h1 = h1 + jnp.swapaxes(h1, -1, -2)
    key, sub = jax.random.split(key)
    chol = 0.03 * jax.random.normal(sub, (n_fields, 2, norb, norb))
    ham = HamChol(basis="charge_spin", h0=jnp.asarray(0.2), h1=h1, chol=chol)
    dm = jnp.stack([jnp.eye(norb), 0.5 * jnp.eye(norb)])

    ctx = _build_prop_ctx(ham, dm, 0.01)

    assert ctx.mf_shifts.shape == (n_fields,)
    assert ctx.exp_h1_half.shape == (2, norb, norb)
    assert ctx.chol_flat.shape == (n_fields, 2 * norb * norb)
    assert ctx.norb == norb


def test_charge_spin_trotter_applies_distinct_spin_fields():
    norb, n_fields = 3, 2
    h1 = jnp.zeros((2, norb, norb))
    chol = jnp.zeros((n_fields, 2, norb, norb))
    chol = chol.at[0, 0, 0, 0].set(1.0)
    chol = chol.at[0, 1, 1, 1].set(2.0)
    ham = HamChol(basis="charge_spin", h0=jnp.asarray(0.0), h1=h1, chol=chol)
    dm = jnp.zeros((2, norb, norb))
    ctx = _build_prop_ctx(ham, dm, 0.04)
    ops = make_trotter_ops("charge_spin", "unrestricted")

    wu = jnp.eye(norb, 1, dtype=jnp.complex128)
    wd = jnp.eye(norb, 1, k=-1, dtype=jnp.complex128)
    out_u, out_d = ops.apply_trotter((wu, wd), jnp.array([0.2, 0.0]), ctx, 6)

    assert not jnp.allclose(out_u, wu)
    assert not jnp.allclose(out_d, wd)
    assert not jnp.allclose(out_u[0, 0], out_d[1, 0])


def test_charge_spin_auto_force_bias_shape():
    norb, n_fields = 4, 3
    sys = System(norb=norb, nelec=(2, 1), walker_kind="unrestricted")
    trial = UhfTrial(jnp.eye(norb, 2), jnp.eye(norb, 1))
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_auto_meas_ops(sys, trial_ops_=trial_ops)
    chol = 0.02 * jnp.ones((n_fields, 2, norb, norb))
    ham = HamChol(
        basis="charge_spin",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((2, norb, norb)),
        chol=chol,
    )
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    walker = (jnp.eye(norb, 2), jnp.eye(norb, 1))

    force_bias = meas_ops.require_kernel("force_bias")(walker, ham, meas_ctx, trial)

    assert force_bias.shape == (n_fields,)


def test_charge_spin_channel_reconstructs_spin_resolved_cholesky_products():
    norb, n_fields = 3, 4
    key = jax.random.PRNGKey(7)
    l_alpha = jax.random.normal(key, (n_fields, norb, norb))
    key, sub = jax.random.split(key)
    l_beta = jax.random.normal(sub, (n_fields, norb, norb))

    l0 = 0.5 * (l_alpha + l_beta)
    lz = 0.5 * (l_alpha - l_beta)

    rec_alpha = l0 + lz
    rec_beta = l0 - lz
    v_ab = jnp.einsum("gpq,grs->pqrs", rec_alpha, rec_beta, optimize="optimal")
    v_ab_ref = jnp.einsum("gpq,grs->pqrs", l_alpha, l_beta, optimize="optimal")

    assert jnp.allclose(rec_alpha, l_alpha)
    assert jnp.allclose(rec_beta, l_beta)
    assert jnp.allclose(v_ab, v_ab_ref)


if __name__ == "__main__":
    pytest.main([__file__])
