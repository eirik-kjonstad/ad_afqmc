import jax
import jax.numpy as jnp
import pytest

from trot.core.ops import MeasOps
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.prop.afqmc import afqmc_step, afqmc_step_diagnostics, _make_field_shifts
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


def _make_dummy_charge_sz_meas_ops():
    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(walker, trial_data):
        return jnp.asarray(1.0 + 0.0j)

    def force_bias_charge_sz_kernel(walker, ham_data, meas_ctx, trial_data):
        n_fields = ham_data.chol.shape[0]
        return jnp.zeros((n_fields, 4), dtype=walker[0].dtype)

    return MeasOps(
        overlap=overlap,
        build_meas_ctx=build_meas_ctx,
        kernels={"force_bias_charge_sz": force_bias_charge_sz_kernel},
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


def test_charge_sz_step_accepts_two_channel_force_bias_for_unrestricted_walkers():
    norb, nocc, nw, n_fields = 4, 2, 5, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    params = QmcParams(dt=0.1, n_chunks=1, n_exp_terms=4)
    trial_data = {"rdm1": jnp.zeros((2, norb, norb))}
    meas_ops = _make_dummy_charge_sz_meas_ops()
    walkers = (
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
    )
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    prop_ctx = _build_prop_ctx(
        ham,
        trial_data["rdm1"],
        params.dt,
        hs_decomposition="charge_sz",
    )
    trotter_ops = make_trotter_ops(ham.basis, "unrestricted", hs_decomposition="charge_sz")
    out = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ops.build_meas_ctx(ham, trial_data),
    )

    assert out.walkers[0].shape == walkers[0].shape
    assert out.walkers[1].shape == walkers[1].shape
    assert out.weights.shape == (nw,)


def test_charge_sz_field_shifts_use_four_channel_couplings():
    norb, n_fields = 3, 2
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    prop_ctx = _build_prop_ctx(
        ham,
        jnp.zeros((2, norb, norb)),
        0.2,
        hs_decomposition="charge_sz",
    )
    force_bias = jnp.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, -0.75, 1.25]],
            [[0.1, -0.2, 0.3, -0.4], [1.5, -1.0, 0.5, -0.25]],
        ]
    )

    shifts = _make_field_shifts(force_bias, prop_ctx)

    inv_sqrt2 = 1.0 / jnp.sqrt(jnp.asarray(2.0, dtype=prop_ctx.sqrt_dt.dtype))
    couplings = jnp.asarray([1.0j, 1.0j, 1.0j * inv_sqrt2, inv_sqrt2])
    expected = -prop_ctx.sqrt_dt * couplings * force_bias
    assert jnp.allclose(shifts, expected)


def test_charge_sz_step_diagnostics_matches_weight_update():
    norb, nocc, nw, n_fields = 4, 2, 5, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    params = QmcParams(dt=0.1, n_chunks=1, n_exp_terms=4)
    trial_data = {"rdm1": jnp.zeros((2, norb, norb))}
    meas_ops = _make_dummy_charge_sz_meas_ops()
    walkers = (
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
    )
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    prop_ctx = _build_prop_ctx(
        ham,
        trial_data["rdm1"],
        params.dt,
        hs_decomposition="charge_sz",
    )
    trotter_ops = make_trotter_ops(ham.basis, "unrestricted", hs_decomposition="charge_sz")
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)

    diag = afqmc_step_diagnostics(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )
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

    assert diag.fields.shape == (nw, n_fields, 4)
    assert diag.force_bias.shape == (nw, n_fields, 4)
    assert diag.field_shifts.shape == (nw, n_fields, 4)
    assert jnp.allclose(out.weights, diag.weight_multiplier)
    assert jnp.array_equal(diag.node_mask, jnp.zeros((nw,), dtype=bool))


def test_charge_sz_requires_four_channel_force_bias_kernel():
    norb, nocc, nw, n_fields = 4, 2, 5, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    params = QmcParams(dt=0.1, n_chunks=1, n_exp_terms=4)
    trial_data = {"rdm1": jnp.zeros((2, norb, norb))}
    walkers = (
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
        jnp.ones((nw, norb, nocc), dtype=jnp.complex64),
    )
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    prop_ctx = _build_prop_ctx(
        ham,
        trial_data["rdm1"],
        params.dt,
        hs_decomposition="charge_sz",
    )
    trotter_ops = make_trotter_ops(ham.basis, "unrestricted", hs_decomposition="charge_sz")
    meas_ops = _make_dummy_meas_ops()

    with pytest.raises(ValueError, match="force_bias_charge_sz"):
        afqmc_step(
            state,
            params=params,
            ham_data=ham,
            trial_data=trial_data,
            meas_ops=meas_ops,
            trotter_ops=trotter_ops,
            prop_ctx=prop_ctx,
            meas_ctx=meas_ops.build_meas_ctx(ham, trial_data),
        )


if __name__ == "__main__":
    pytest.main([__file__])
