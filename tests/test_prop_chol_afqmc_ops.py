import jax
import jax.numpy as jnp
import pytest

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import (
    _build_prop_ctx,
    _force_bias_charge_sz_from_halves,
    _make_charge_sz_vhs_split_flat,
)


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


def test_charge_sz_build_prop_ctx_shapes_and_charge_h1_correction():
    norb, n_fields = 4, 3
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)

    dm = jnp.zeros((2, norb, norb))
    dt = 0.2
    ctx = _build_prop_ctx(ham, dm, dt, hs_decomposition="charge_sz")

    assert ctx.hs_decomposition == "charge_sz"
    assert ctx.mf_shifts.shape == (n_fields, 4)
    assert ctx.chol_flat.shape == (n_fields, norb * norb)

    charge_v0 = 0.5 * jnp.einsum("gik,gkj->ij", ham.chol, ham.chol, optimize="optimal")
    exp_charge = jax.scipy.linalg.expm(-0.5 * dt * (ham.h1 - charge_v0))
    assert jnp.allclose(ctx.exp_h1_half, exp_charge)


def test_charge_sz_mean_field_bookkeeping_is_unshifted():
    norb, n_fields = 4, 3
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=1.25)
    dm = jnp.stack(
        [
            jnp.diag(jnp.asarray([1.0, 0.8, 0.2, 0.0])),
            jnp.diag(jnp.asarray([1.0, 0.5, 0.1, 0.0])),
        ]
    )
    dt = 0.05

    ctx = _build_prop_ctx(ham, dm, dt, hs_decomposition="charge_sz")
    ctx_zero_dm = _build_prop_ctx(
        ham,
        jnp.zeros_like(dm),
        dt,
        hs_decomposition="charge_sz",
    )

    assert jnp.allclose(ctx.mf_shifts, jnp.zeros((n_fields, 4), dtype=ctx.mf_shifts.dtype))
    assert jnp.allclose(ctx.h0_prop, -ham.h0)
    assert jnp.allclose(ctx.exp_h1_half, ctx_zero_dm.exp_h1_half)


def test_charge_sz_potential_construction():
    chol = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-0.5, 0.25], [0.75, -1.25]],
        ]
    )
    field = jnp.asarray([[0.3, -0.7, 1.2, 0.4], [1.1, -0.9, 0.8, -1.4]])
    dt = jnp.asarray(0.08)
    v_up, v_dn = _make_charge_sz_vhs_split_flat(
        chol_flat=chol.reshape(chol.shape[0], -1),
        field=field,
        sqrt_dt=jnp.sqrt(dt),
        n=2,
    )

    scale = jnp.sqrt(dt)
    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    expected_up = jnp.einsum(
        "g,gij->ij",
        scale * (1.0j * field[:, 0] + inv_sqrt2 * (1.0j * field[:, 2] + field[:, 3])),
        chol,
    )
    expected_dn = jnp.einsum(
        "g,gij->ij",
        scale * (1.0j * field[:, 1] + inv_sqrt2 * (1.0j * field[:, 2] - field[:, 3])),
        chol,
    )
    assert jnp.allclose(v_up, expected_up)
    assert jnp.allclose(v_dn, expected_dn)


def test_charge_sz_force_bias_sum_and_difference():
    chol = jnp.asarray(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-1.5, 0.25], [2.0, -0.75]],
        ]
    )
    green_up = jnp.asarray([[0.1, 0.2], [0.3, -0.4]])
    green_dn = jnp.asarray([[-0.7, 0.5], [0.6, 0.9]])

    fb = _force_bias_charge_sz_from_halves(chol, chol, green_up, green_dn)

    fb_u = jnp.einsum("gij,ij->g", chol, green_up, optimize="optimal")
    fb_d = jnp.einsum("gij,ij->g", chol, green_dn, optimize="optimal")
    expected = jnp.stack([fb_u, fb_d, fb_u + fb_d, fb_u - fb_d], axis=-1)
    assert jnp.allclose(fb, expected)


def test_charge_sz_full_decomposition_matches_charge_square():
    u = 3.5
    occupations = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    n_up = occupations[:, 0]
    n_dn = occupations[:, 1]
    a = jnp.sqrt(u) * n_up
    b = jnp.sqrt(u) * n_dn

    lhs = 0.5 * a**2 + 0.5 * b**2 + 0.25 * (a + b) ** 2 - 0.25 * (a - b) ** 2
    rhs = 0.5 * (a + b) ** 2
    assert jnp.allclose(lhs, rhs)


if __name__ == "__main__":
    pytest.main([__file__])
