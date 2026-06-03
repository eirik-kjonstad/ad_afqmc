from trot import config

config.configure_once()

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from pyscf import gto, scf

from trot import testing
from trot.afqmc import Afqmc
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.auto import make_auto_meas_ops
from trot.meas.uhf import (
    build_meas_ctx,
    energy_kernel_gw_rh,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_gw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    make_uhf_meas_ops,
)
from trot.prop.types import QmcParams
from trot.setup import setup
from trot.staging import HamInput, StagedInputs, TrialInput
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops


def _make_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ka, kb = jax.random.split(key)
    ca = testing.rand_orthonormal_cols(ka, norb, nup, dtype=dtype)
    cb = testing.rand_orthonormal_cols(kb, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)


def test_spin_auto_force_bias_matches_uhf_spin_channel_transform():
    key = jax.random.PRNGKey(12)
    norb, nup, ndn, n_chol = 6, 2, 1, 8
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = testing.make_random_ham_chol(key, norb=norb, n_chol=n_chol)
    trial = _make_uhf_trial(jax.random.fold_in(key, 1), norb, nup, ndn)
    walker = testing.make_walkers(jax.random.fold_in(key, 2), sys)

    trial_ops = make_uhf_trial_ops(sys)
    meas_auto_spin = make_auto_meas_ops(sys, trial_ops, decomposition="spin")
    ctx_auto_spin = meas_auto_spin.build_meas_ctx(ham, trial)
    fb_spin = meas_auto_spin.require_kernel(k_force_bias)(walker, ham, ctx_auto_spin, trial)

    meas_ctx = build_meas_ctx(ham, trial)
    wu, wd = walker
    mu = trial.mo_coeff_a.conj().T @ wu
    md = trial.mo_coeff_b.conj().T @ wd
    gu = jnp.linalg.solve(mu.T, wu.T)
    gd = jnp.linalg.solve(md.T, wd.T)
    fb_a = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_a, gu, optimize="optimal")
    fb_b = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_b, gd, optimize="optimal")
    sqrt2 = jnp.sqrt(jnp.asarray(2.0))
    expected = jnp.concatenate([sqrt2 * fb_a, sqrt2 * fb_b, 1.0j * (fb_a - fb_b)])

    assert fb_spin.shape == (3 * n_chol,)
    assert jnp.allclose(fb_spin, expected, rtol=5e-6, atol=5e-7), (fb_spin, expected)


def test_interpolated_spin_auto_force_bias_matches_scaled_channels():
    key = jax.random.PRNGKey(13)
    norb, nup, ndn, n_chol = 6, 2, 1, 8
    lam = 0.25
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = testing.make_random_ham_chol(key, norb=norb, n_chol=n_chol)
    trial = _make_uhf_trial(jax.random.fold_in(key, 1), norb, nup, ndn)
    walker = testing.make_walkers(jax.random.fold_in(key, 2), sys)

    trial_ops = make_uhf_trial_ops(sys)
    meas_auto_spin = make_auto_meas_ops(
        sys,
        trial_ops,
        decomposition="spin",
        spin_decomposition_lambda=lam,
    )
    ctx_auto_spin = meas_auto_spin.build_meas_ctx(ham, trial)
    fb_spin = meas_auto_spin.require_kernel(k_force_bias)(walker, ham, ctx_auto_spin, trial)

    meas_ctx = build_meas_ctx(ham, trial)
    wu, wd = walker
    mu = trial.mo_coeff_a.conj().T @ wu
    md = trial.mo_coeff_b.conj().T @ wd
    gu = jnp.linalg.solve(mu.T, wu.T)
    gd = jnp.linalg.solve(md.T, wd.T)
    fb_a = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_a, gu, optimize="optimal")
    fb_b = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_b, gd, optimize="optimal")
    sqrt2 = jnp.sqrt(jnp.asarray(2.0))
    sqrt_lam = jnp.sqrt(jnp.asarray(lam))
    sqrt_charge = jnp.sqrt(jnp.asarray(1.0 - lam))
    expected = jnp.concatenate(
        [
            sqrt_charge * (fb_a + fb_b),
            sqrt_lam * sqrt2 * fb_a,
            sqrt_lam * sqrt2 * fb_b,
            sqrt_lam * 1.0j * (fb_a - fb_b),
        ]
    )

    assert fb_spin.shape == (4 * n_chol,)
    assert jnp.allclose(fb_spin, expected, rtol=5e-6, atol=5e-7), (fb_spin, expected)


def test_setup_spin_decomposition_builds_three_field_uhf_job():
    norb, nup, ndn, n_chol = 4, 2, 1, 3
    rng = np.random.default_rng(4)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.standard_normal((n_chol, norb, norb)) * 0.05
    mo = np.eye(norb)

    staged = StagedInputs(
        ham=HamInput(
            h0=0.0,
            h1=h1,
            chol=chol,
            nelec=(nup, ndn),
            norb=norb,
            chol_cut=1.0e-5,
            frozen=0,
            source_kind="mf",
            basis="restricted",
        ),
        trial=TrialInput(
            kind="uhf",
            data={"mo_a": mo, "mo_b": mo},
            frozen=0,
            source_kind="mf",
        ),
        meta={"source_kind": "mf", "chol_cut": 1.0e-5, "frozen": 0},
    )

    job = setup(
        staged,
        walker_kind="unrestricted",
        mixed_precision=False,
        decomposition="spin",
        params=QmcParams(n_walkers=2, n_eql_blocks=0, n_blocks=0),
    )
    rdm1 = job.trial_ops.get_rdm1(job.trial_data)
    prop_ctx = job.prop_ops.build_prop_ctx(job.ham_data, rdm1, job.params)
    meas_ctx = job.meas_ops.build_meas_ctx(job.ham_data, job.trial_data)
    walker = (jnp.eye(norb, nup), jnp.eye(norb, ndn))
    fb = job.meas_ops.require_kernel(k_force_bias)(
        walker,
        job.ham_data,
        meas_ctx,
        job.trial_data,
    )

    assert job.decomposition == "spin"
    assert prop_ctx.mf_shifts.shape == (3 * n_chol,)
    assert prop_ctx.chol_flat.shape == (3 * n_chol, norb * norb)
    assert fb.shape == (3 * n_chol,)


def test_setup_interpolated_spin_decomposition_builds_four_field_uhf_job():
    norb, nup, ndn, n_chol = 4, 2, 1, 3
    lam = 0.5
    rng = np.random.default_rng(5)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.standard_normal((n_chol, norb, norb)) * 0.05
    mo = np.eye(norb)

    staged = StagedInputs(
        ham=HamInput(
            h0=0.0,
            h1=h1,
            chol=chol,
            nelec=(nup, ndn),
            norb=norb,
            chol_cut=1.0e-5,
            frozen=0,
            source_kind="mf",
            basis="restricted",
        ),
        trial=TrialInput(
            kind="uhf",
            data={"mo_a": mo, "mo_b": mo},
            frozen=0,
            source_kind="mf",
        ),
        meta={"source_kind": "mf", "chol_cut": 1.0e-5, "frozen": 0},
    )

    job = setup(
        staged,
        walker_kind="unrestricted",
        mixed_precision=False,
        decomposition="spin",
        spin_decomposition_lambda=lam,
        params=QmcParams(n_walkers=2, n_eql_blocks=0, n_blocks=0),
    )
    rdm1 = job.trial_ops.get_rdm1(job.trial_data)
    prop_ctx = job.prop_ops.build_prop_ctx(job.ham_data, rdm1, job.params)
    meas_ctx = job.meas_ops.build_meas_ctx(job.ham_data, job.trial_data)
    walker = (jnp.eye(norb, nup), jnp.eye(norb, ndn))
    fb = job.meas_ops.require_kernel(k_force_bias)(
        walker,
        job.ham_data,
        meas_ctx,
        job.trial_data,
    )

    assert job.decomposition == "spin"
    assert job.spin_decomposition_lambda == lam
    assert prop_ctx.mf_shifts.shape == (4 * n_chol,)
    assert prop_ctx.chol_flat.shape == (4 * n_chol, norb * norb)
    assert fb.shape == (4 * n_chol,)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)


def test_force_bias_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(jax.Array, wi)
        fbr = force_bias_kernel_rw_rh(wi, ham, ctx, trial)
        fbu = force_bias_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(fbr, fbu, atol=1e-12), (fbr, fbu)


def test_force_bias_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        fbg = force_bias_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(fbu, fbg, atol=1e-12), (fbu, fbg)


def test_energy_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(jax.Array, wi)
        er = energy_kernel_rw_rh(wi, ham, ctx, trial)
        eu = energy_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(er, eu, atol=1e-12), (er, eu)


def test_energy_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 1
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        eg = energy_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)


def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    return mf


def mf2():
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    return mf


mf = mf()  # type: ignore
mf2 = mf2()  # type: ignore


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf, "restricted", -75.75594187783527, 0.01213383697785241),
        (mf2, "unrestricted", -55.43066756011652, 0.00761980459817991),
        (mf2, "generalized", -55.43066756011653, 0.007619804598170696),
    ],
)
def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    myafqmc = Afqmc(mf)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
