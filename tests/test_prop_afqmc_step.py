import jax
import jax.numpy as jnp
import pytest

from trot.core.ops import MeasOps
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.prop.afqmc import afqmc_step, coefficient_space_global_phaseless_projection
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.prop.types import PropState, QmcParams
from trot import testing


def _make_dummy_meas_ops():
    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(walker, trial_data):
        return jnp.asarray(1.0 + 0.0j)

    def force_bias_kernel(walker, ham_data, meas_ctx, trial_data):
        n_fields = ham_data.chol.shape[0]
        return jnp.zeros((n_fields,), dtype=walker.dtype)

    return MeasOps(
        overlap=overlap,
        build_meas_ctx=build_meas_ctx,
        kernels={"force_bias": force_bias_kernel},
        observables={},
    )


def test_weight_update_matches_h0_prop_and_pop_control_update():
    norb, nocc, nw, n_fields = 4, 2, 8, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(1.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    params = QmcParams(
        dt=0.2,
        n_chunks=2,
        n_exp_terms=4,
        pop_control_damping=0.1,
    )

    testing.make_dummy_trial_ops()
    meas_ops = _make_dummy_meas_ops()
    trial_data = {"rdm1": jnp.zeros((norb, norb))}

    walkers = jnp.ones((nw, norb, nocc), dtype=jnp.complex64)
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    trotter_ops = make_trotter_ops(ham.basis, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
    out = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    expected_w = jnp.exp(-jnp.asarray(params.dt)) * jnp.ones((nw,))
    assert jnp.allclose(out.weights, expected_w)
    assert jnp.allclose(out.pop_control_ene_shift, jnp.asarray(0.1))
    assert jnp.allclose(out.e_estimate, jnp.asarray(0.0))


def test_step_matches_manual_walker_propagation_and_is_chunk_invariant():

    norb, nocc, nw, n_fields = 5, 2, 6, 4
    key = jax.random.PRNGKey(42)

    a = jax.random.normal(key, (norb, norb))
    h1 = 0.05 * (a + a.T)
    key, sub = jax.random.split(key)
    chol = 0.02 * jax.random.normal(sub, (n_fields, norb, norb))

    ham = HamChol(basis="restricted", h0=jnp.asarray(0.0), h1=h1, chol=chol)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    params1 = QmcParams(dt=0.1, n_chunks=1, n_exp_terms=6)
    params2 = QmcParams(dt=0.1, n_chunks=3, n_exp_terms=6)

    testing.make_dummy_trial_ops()
    meas_ops = _make_dummy_meas_ops()
    trial_data = {"rdm1": jnp.zeros((norb, norb))}

    key, sub = jax.random.split(key)
    walkers = jax.random.normal(sub, (nw, norb, nocc)).astype(
        jnp.complex64
    ) + 1.0j * jax.random.normal(sub, (nw, norb, nocc)).astype(jnp.complex64)
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    trotter_ops = make_trotter_ops(ham.basis, sys.walker_kind)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params1.dt)
    out1 = afqmc_step(
        state,
        params=params1,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    key_next, subkey = jax.random.split(state.rng_key)
    fields = jax.random.normal(subkey, (nw, n_fields)).astype(jnp.complex64)

    ops = make_trotter_ops(ham.basis, "restricted")
    ctx = _build_prop_ctx(ham, trial_data["rdm1"], params1.dt)

    def trotter(w, f):
        return ops.apply_trotter(w, f, ctx, params1.n_exp_terms)

    expected_walkers = jax.vmap(trotter)(walkers, fields)

    assert jnp.allclose(out1.walkers, expected_walkers)
    assert jnp.allclose(out1.overlaps, jnp.ones((nw,), dtype=jnp.complex64))
    assert jnp.all(out1.rng_key == key_next)

    out2 = afqmc_step(
        state,
        params=params2,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    assert jnp.allclose(out2.walkers, out1.walkers)
    assert jnp.allclose(out2.weights, out1.weights)
    assert jnp.allclose(out2.overlaps, out1.overlaps)
    assert jnp.all(out2.rng_key == out1.rng_key)


def test_coefficient_space_global_projection_matches_closed_form():
    w_tilde = jnp.asarray([1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 0.5j])
    overlaps = jnp.asarray([2.0 + 0.0j, 1.0 - 1.0j, 0.5j])
    dt = 0.2
    r = 0.7

    out = coefficient_space_global_phaseless_projection(
        w_tilde,
        overlaps,
        dt=dt,
        budget_scale=r,
        gauge_fix=False,
    )

    a_norm = jnp.sqrt(jnp.sum(jnp.abs(w_tilde / overlaps) ** 2))
    budget = r * dt * a_norm
    phase = jnp.sum(w_tilde) / jnp.abs(jnp.sum(w_tilde))
    s_norm = jnp.sqrt(jnp.sum(jnp.abs(overlaps) ** 2))
    expected = w_tilde + (budget / s_norm) * phase * jnp.abs(overlaps) ** 2

    assert jnp.allclose(out, expected)


def test_global_projection_step_uses_complex_importance_without_local_cosine():
    norb, nocc, nw, n_fields = 4, 2, 5, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    params = QmcParams(
        dt=0.2,
        n_chunks=1,
        n_exp_terms=4,
        pop_control_damping=0.1,
        global_phaseless_projection=True,
        global_phaseless_budget_scale=0.5,
        global_phaseless_gauge_fix=False,
    )

    meas_ops = _make_dummy_meas_ops()
    trial_data = {"rdm1": jnp.zeros((norb, norb))}
    walkers = jnp.ones((nw, norb, nocc), dtype=jnp.complex64)
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,), dtype=jnp.complex64),
        overlaps=1.0j * jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    trotter_ops = make_trotter_ops(ham.basis, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
    out = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    preliminary = -1.0j * jnp.ones((nw,), dtype=jnp.complex64)
    expected = coefficient_space_global_phaseless_projection(
        preliminary,
        jnp.ones((nw,), dtype=jnp.complex64),
        dt=params.dt,
        budget_scale=params.global_phaseless_budget_scale,
        gauge_fix=False,
    )

    assert jnp.allclose(out.weights, expected)
    assert jnp.all(jnp.abs(out.weights) > 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
