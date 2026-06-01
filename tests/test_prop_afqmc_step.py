import jax
import jax.numpy as jnp
import pytest

from trot.core.ops import MeasOps
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.prop.afqmc import afqmc_step
from trot.prop.chol_afqmc_ops import CholAfqmcCtx, TrotterOps, _build_prop_ctx, make_trotter_ops
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


def _make_dummy_spin_meas_ops():
    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(walker, trial_data):
        return jnp.asarray(1.0 + 0.0j)

    def force_bias_kernel(walker, ham_data, meas_ctx, trial_data):
        n_fields = 3 * ham_data.chol.shape[0]
        wu, _ = walker
        return jnp.zeros((n_fields,), dtype=wu.dtype)

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


def test_spin_decomposition_step_runs_with_three_aux_fields():
    norb, nocc, nw, n_fields = 4, 1, 5, 3
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((norb, norb)),
        chol=jnp.zeros((n_fields, norb, norb)),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="unrestricted")
    params = QmcParams(dt=0.05, n_chunks=1, n_exp_terms=4)
    meas_ops = _make_dummy_spin_meas_ops()
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

    trotter_ops = make_trotter_ops(
        ham.basis, sys.walker_kind, hs_decomposition="spin"
    )
    prop_ctx = _build_prop_ctx(
        ham, trial_data["rdm1"], params.dt, hs_decomposition="spin"
    )
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

    assert prop_ctx.mf_shifts.shape == (3 * n_fields,)
    assert jnp.allclose(out.weights, jnp.ones((nw,)))
    assert jnp.asarray(out.node_encounters) == 0
    assert jnp.asarray(out.ab_cos_nodes) == 0
    assert jnp.asarray(out.s_sign_nodes) == 0
    assert jnp.asarray(out.floor_kills) == 0


def test_spin_decomposition_uses_three_conditional_importance_factors():
    nw = 1
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((1, 1)),
        chol=jnp.zeros((1, 1, 1)),
    )
    params = QmcParams(
        dt=1.0,
        n_chunks=1,
        n_exp_terms=1,
        weight_floor=0.0,
        weight_cap=1.0e12,
        pop_control_damping=0.0,
    )

    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(_walker, _trial_data):
        return jnp.asarray(1.0 + 0.0j)

    def force_bias_kernel(walker, _ham_data, _meas_ctx, _trial_data):
        wu, wd = walker
        alpha_marker = jnp.real(wu[0, 0])
        beta_marker = jnp.real(wd[0, 0])
        return jnp.asarray(
            [1.0, 2.0 + alpha_marker, 3.0 + beta_marker],
            dtype=wu.dtype,
        )

    def apply_a(walker, field, _ctx, _n_terms):
        _wu, wd = walker
        wu = jnp.asarray([[field[0]]], dtype=wd.dtype)
        return (wu, wd)

    def apply_b(walker, field, _ctx, _n_terms):
        wu, _wd = walker
        wd = jnp.asarray([[field[1]]], dtype=wu.dtype)
        return (wu, wd)

    def apply_s(walker, field, _ctx, _n_terms):
        wu, wd = walker
        return (wu + field[2], wd - field[2])

    meas_ops = MeasOps(
        overlap=overlap,
        build_meas_ctx=build_meas_ctx,
        kernels={"force_bias": force_bias_kernel},
        observables={},
    )
    trotter_ops = TrotterOps(
        apply_trotter=lambda walker, _field, _ctx, _n_terms: walker,
        apply_trotter_a=apply_a,
        apply_trotter_b=apply_b,
        apply_trotter_s=apply_s,
    )
    prop_ctx = CholAfqmcCtx(
        dt=jnp.asarray(1.0),
        sqrt_dt=jnp.asarray(1.0),
        exp_h1_half=jnp.eye(1),
        mf_shifts=jnp.zeros((3,), dtype=jnp.complex64),
        force_bias_scales=jnp.ones((3,), dtype=jnp.complex64),
        h0_prop=jnp.asarray(0.0),
        chol_flat=jnp.zeros((1, 1)),
        norb=1,
    )
    walkers = (
        jnp.zeros((nw, 1, 1), dtype=jnp.complex64),
        jnp.zeros((nw, 1, 1), dtype=jnp.complex64),
    )
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(19),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    out = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=None,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=None,
    )

    _, subkey = jax.random.split(state.rng_key)
    fields = jax.random.normal(subkey, (nw, 3))[0]

    field_shift_a = -jnp.asarray(1.0)
    shifted_a = fields[0] - field_shift_a
    field_shift_b = -(2.0 + shifted_a)
    shifted_b = fields[1] - field_shift_b
    field_shift_s = -(3.0 + shifted_b)

    fb_term_a = fields[0] * field_shift_a - 0.5 * field_shift_a * field_shift_a
    fb_term_b = fields[1] * field_shift_b - 0.5 * field_shift_b * field_shift_b
    fb_term_s = fields[2] * field_shift_s - 0.5 * field_shift_s * field_shift_s
    expected_weight = jnp.exp(fb_term_a) * jnp.exp(fb_term_b) * jnp.exp(fb_term_s)

    assert jnp.allclose(out.weights[0], expected_weight)
    assert jnp.allclose(out.walkers[0][0, 0, 0], shifted_a + fields[2] - field_shift_s)
    assert jnp.allclose(out.walkers[1][0, 0, 0], shifted_b - (fields[2] - field_shift_s))
    assert jnp.asarray(out.ab_cos_nodes) == 0
    assert jnp.asarray(out.s_sign_nodes) == 0
    assert jnp.asarray(out.floor_kills) == 0


def test_spin_decomposition_combines_alpha_beta_phaseless_projection():
    nw = 1
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((1, 1)),
        chol=jnp.zeros((1, 1, 1)),
    )
    params = QmcParams(
        dt=1.0,
        n_chunks=1,
        n_exp_terms=1,
        weight_floor=0.0,
        weight_cap=1.0e12,
        pop_control_damping=0.0,
    )
    theta = 0.75 * jnp.pi

    def build_meas_ctx(_ham, _trial):
        return None

    def overlap(walker, _trial_data):
        marker = jnp.real(walker[0][0, 0])
        phase = jnp.where(
            marker == 1.0,
            0.0,
            jnp.where(marker == 2.0, theta, jnp.where(marker == 3.0, 0.0, 0.0)),
        )
        return jnp.exp(1.0j * phase)

    def force_bias_kernel(walker, _ham_data, _meas_ctx, _trial_data):
        wu, _ = walker
        return jnp.zeros((3,), dtype=wu.dtype)

    def apply_a(_walker, _field, _ctx, _n_terms):
        return (
            jnp.asarray([[2.0 + 0.0j]], dtype=jnp.complex64),
            jnp.asarray([[0.0 + 0.0j]], dtype=jnp.complex64),
        )

    def apply_b(_walker, _field, _ctx, _n_terms):
        return (
            jnp.asarray([[3.0 + 0.0j]], dtype=jnp.complex64),
            jnp.asarray([[0.0 + 0.0j]], dtype=jnp.complex64),
        )

    def apply_s(walker, _field, _ctx, _n_terms):
        return walker

    meas_ops = MeasOps(
        overlap=overlap,
        build_meas_ctx=build_meas_ctx,
        kernels={"force_bias": force_bias_kernel},
        observables={},
    )
    trotter_ops = TrotterOps(
        apply_trotter=lambda walker, _field, _ctx, _n_terms: walker,
        apply_trotter_a=apply_a,
        apply_trotter_b=apply_b,
        apply_trotter_s=apply_s,
    )
    prop_ctx = CholAfqmcCtx(
        dt=jnp.asarray(1.0),
        sqrt_dt=jnp.asarray(1.0),
        exp_h1_half=jnp.eye(1),
        mf_shifts=jnp.zeros((3,), dtype=jnp.complex64),
        force_bias_scales=jnp.ones((3,), dtype=jnp.complex64),
        h0_prop=jnp.asarray(0.0),
        chol_flat=jnp.zeros((1, 1)),
        norb=1,
    )
    state = PropState(
        walkers=(
            jnp.asarray([[[1.0 + 0.0j]]], dtype=jnp.complex64),
            jnp.asarray([[[0.0 + 0.0j]]], dtype=jnp.complex64),
        ),
        weights=jnp.ones((nw,)),
        overlaps=jnp.ones((nw,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(0),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    out = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=None,
        meas_ops=meas_ops,
        trotter_ops=trotter_ops,
        prop_ctx=prop_ctx,
        meas_ctx=None,
    )

    assert jnp.allclose(out.weights[0], 1.0)
    assert jnp.asarray(out.ab_cos_nodes) == 0
    assert jnp.asarray(out.floor_kills) == 0


if __name__ == "__main__":
    pytest.main([__file__])
