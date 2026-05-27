import jax.numpy as jnp
import jax

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.spin_decomp import _force_bias_spin_uw, wrap_spin_decomp_meas_ops
from trot.meas.uhf import make_uhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.types import PropState, QmcParams
from trot.setup import setup
from trot.staging import HamInput, StagedInputs, TrialInput
from trot.trial.uhf import UhfTrial, overlap_u


def test_spin_decomp_force_bias_matches_uhf_reference_contractions():
    chol = jnp.array(
        [
            [[1.0, 0.2, 0.0], [0.2, -0.3, 0.1], [0.0, 0.1, 0.4]],
            [[0.5, -0.1, 0.3], [-0.1, 0.7, 0.2], [0.3, 0.2, -0.2]],
        ]
    )
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((3, 3)),
        chol=chol,
    )
    mo_a = jnp.eye(3, 2)
    mo_b = jnp.eye(3, 1)
    trial = UhfTrial(mo_a, mo_b)
    walker = (mo_a.astype(jnp.complex128), mo_b.astype(jnp.complex128))

    fb = _force_bias_spin_uw(walker, ham, trial, overlap=overlap_u)

    expected_a = jnp.array([chol[0, 0, 0] + chol[0, 1, 1], chol[1, 0, 0] + chol[1, 1, 1]])
    expected_b = jnp.array([chol[0, 0, 0], chol[1, 0, 0]])
    expected = jnp.concatenate([expected_a, expected_b, expected_a - expected_b])

    assert jnp.allclose(fb, expected)


def test_spin_decomp_uhf_step_smoke():
    chol = jnp.array(
        [
            [[0.1, 0.02, 0.0], [0.02, -0.03, 0.01], [0.0, 0.01, 0.04]],
            [[0.05, -0.01, 0.03], [-0.01, 0.07, 0.02], [0.03, 0.02, -0.02]],
        ]
    )
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((3, 3)),
        chol=chol,
    )
    sys = System(norb=3, nelec=(2, 1), walker_kind="unrestricted")
    trial = UhfTrial(jnp.eye(3, 2), jnp.eye(3, 1))
    base_meas_ops = make_uhf_meas_ops(sys)
    meas_ops = wrap_spin_decomp_meas_ops(base_meas_ops, sys=sys, trial_kind="uhf")
    prop_ops = make_prop_ops("restricted", "unrestricted", hs_decomposition="spin")
    params = QmcParams(dt=0.01, n_walkers=2, n_chunks=1, n_exp_terms=4)
    rdm1 = jnp.stack(
        [
            jnp.diag(jnp.array([1.0, 1.0, 0.0])),
            jnp.diag(jnp.array([1.0, 0.0, 0.0])),
        ],
        axis=0,
    )
    prop_ctx = prop_ops.build_prop_ctx(ham, rdm1, params)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    walkers = (
        jnp.stack([trial.mo_coeff_a, trial.mo_coeff_a]).astype(jnp.complex128),
        jnp.stack([trial.mo_coeff_b, trial.mo_coeff_b]).astype(jnp.complex128),
    )
    overlaps = jnp.array([overlap_u((trial.mo_coeff_a, trial.mo_coeff_b), trial)] * 2)
    state = PropState(
        walkers=walkers,
        weights=jnp.ones((2,)),
        overlaps=overlaps,
        rng_key=jax.random.PRNGKey(7),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )

    out = prop_ops.step(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=None,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    assert out.weights.shape == (2,)
    assert out.walkers[0].shape == walkers[0].shape
    assert out.walkers[1].shape == walkers[1].shape


def test_setup_accepts_spin_decomposition_for_uhf_unrestricted():
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
        kind="uhf",
        data={"mo_a": jnp.eye(2), "mo_b": jnp.eye(2)},
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
        hs_decomposition="spin",
        mixed_precision=False,
        params=QmcParams(dt=0.01, n_walkers=1, n_blocks=1, n_prop_steps=1, n_eql_blocks=0),
    )
    state, meas_ctx, prop_ctx = job._prepare_runtime()

    assert job.hs_decomposition == "spin"
    assert prop_ctx.mf_shifts.shape == (3,)
    assert state.walkers[0].shape == (1, 2, 1)
    assert meas_ctx.base_ctx is not None
