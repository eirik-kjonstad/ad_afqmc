from trot import config

config.configure_once()

import jax
import jax.numpy as jnp

from trot import testing
from trot.core.system import System
from trot.meas.ucisd import make_ucisd_meas_ops
from trot.meas.uhf import make_uhf_meas_ops
from trot.prop.blocks import block_uhf_bra_energy
from trot.prop.types import PropOps, PropState, QmcParams
from trot.setup import setup
from trot.staging import HamInput, StagedInputs, TrialInput
from trot.trial.ucisd import UcisdTrial, make_ucisd_trial_ops
from trot.trial.uhf import UhfTrial
from trot import walkers as wk


def _identity_prop_ops() -> PropOps:
    def step(state, **_kwargs):
        return state

    return PropOps(
        init_prop_state=lambda **_kwargs: None,
        build_prop_ctx=lambda *_args, **_kwargs: None,
        step=step,
    )


def _identity_sr(walkers, weights, _zeta, _walker_kind):
    return walkers, weights


def _make_ucisd_trial(key, norb: int, nup: int, ndn: int) -> UcisdTrial:
    nva = norb - nup
    nvb = norb - ndn
    k1, k2, k3, k4, k5, kb = jax.random.split(key, 6)
    return UcisdTrial(
        mo_coeff_a=jnp.eye(norb),
        mo_coeff_b=testing.rand_orthonormal_cols(kb, norb, norb),
        c1a=0.03 * jax.random.normal(k1, (nup, nva)),
        c1b=0.02 * jax.random.normal(k2, (ndn, nvb)),
        c2aa=0.01 * jax.random.normal(k3, (nup, nva, nup, nva)),
        c2ab=0.01 * jax.random.normal(k4, (nup, nva, ndn, nvb)),
        c2bb=0.01 * jax.random.normal(k5, (ndn, nvb, ndn, nvb)),
    )


def test_uhf_bra_block_energy_matches_manual_reweighted_formula():
    key = jax.random.PRNGKey(41)
    norb, nup, ndn, n_walkers = 4, 2, 1, 3
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = testing.make_random_ham_chol(key, norb=norb, n_chol=3)
    trial = _make_ucisd_trial(jax.random.fold_in(key, 1), norb, nup, ndn)
    trial_ops = make_ucisd_trial_ops(sys)
    meas_ops = make_ucisd_meas_ops(sys, mixed_precision=False)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    uhf_trial = UhfTrial(trial.mo_coeff_a[:, :nup], trial.mo_coeff_b[:, :ndn])
    uhf_meas_ops = make_uhf_meas_ops(sys)
    uhf_meas_ctx = uhf_meas_ops.build_meas_ctx(ham, uhf_trial)
    walker_list = [
        testing.make_walkers(jax.random.fold_in(key, 20 + i), sys) for i in range(n_walkers)
    ]
    walkers = (
        jnp.stack([walker[0] for walker in walker_list], axis=0),
        jnp.stack([walker[1] for walker in walker_list], axis=0),
    )
    weights = jnp.array([0.7, 1.2, 0.9])
    state = PropState(
        walkers=walkers,
        weights=weights,
        overlaps=jnp.ones((n_walkers,), dtype=jnp.complex64),
        rng_key=jax.random.PRNGKey(3),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    params = QmcParams(dt=0.01, n_walkers=n_walkers, n_chunks=1, n_prop_steps=0)

    _, obs = block_uhf_bra_energy(
        state,
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        meas_ctx=meas_ctx,
        prop_ops=_identity_prop_ops(),
        prop_ctx=None,
        uhf_trial_data=uhf_trial,
        uhf_meas_ops=uhf_meas_ops,
        uhf_meas_ctx=uhf_meas_ctx,
        sr_fn=_identity_sr,
    )

    walkers_orth = wk.orthonormalize(walkers, sys.walker_kind)
    corr_overlaps = jax.vmap(meas_ops.overlap, in_axes=(0, None))(walkers_orth, trial)
    uhf_overlaps = jax.vmap(uhf_meas_ops.overlap, in_axes=(0, None))(walkers_orth, uhf_trial)
    uhf_e_kernel = uhf_meas_ops.require_kernel("energy")
    uhf_energies = jax.vmap(uhf_e_kernel, in_axes=(0, None, None, None))(
        walkers_orth, ham, uhf_meas_ctx, uhf_trial
    )
    f = uhf_overlaps / corr_overlaps
    expected = jnp.sum(jnp.real(weights * uhf_energies * f)) / jnp.sum(jnp.real(weights * f))

    assert jnp.allclose(obs.scalars["energy"], expected)
    assert jnp.allclose(obs.scalars["weight"], jnp.sum(weights))


def test_setup_accepts_uhf_bra_energy_estimator_for_ucisd_unrestricted():
    ham = HamInput(
        h0=0.0,
        h1=jnp.zeros((2, 2)),
        chol=jnp.array([[[0.1, 0.0], [0.0, -0.2]]]),
        nelec=(1, 1),
        norb=2,
        chol_cut=1.0e-5,
        frozen=0,
        source_kind="mf",
        basis="restricted",
    )
    trial = TrialInput(
        kind="ucisd",
        data={
            "mo_coeff_a": jnp.eye(2),
            "mo_coeff_b": jnp.eye(2),
            "ci1a": jnp.zeros((1, 1)),
            "ci1b": jnp.zeros((1, 1)),
            "ci2aa": jnp.zeros((1, 1, 1, 1)),
            "ci2ab": jnp.zeros((1, 1, 1, 1)),
            "ci2bb": jnp.zeros((1, 1, 1, 1)),
        },
        frozen=0,
        source_kind="mf",
    )
    staged = StagedInputs(
        ham=ham,
        trial=trial,
        meta={"source_kind": "mf", "chol_cut": 1.0e-5},
    )

    job = setup(
        staged,
        walker_kind="unrestricted",
        energy_estimator="uhf_bra",
        mixed_precision=False,
        params=QmcParams(dt=0.01, n_walkers=1, n_blocks=1, n_prop_steps=0, n_eql_blocks=0),
    )
    state, _meas_ctx, _prop_ctx = job._prepare_runtime()

    assert job.energy_estimator == "uhf_bra"
    assert state.walkers[0].shape == (1, 2, 1)
