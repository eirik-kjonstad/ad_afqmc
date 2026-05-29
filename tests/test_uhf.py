from trot import config

config.configure_once()

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from pyscf import gto, scf

from trot import testing
from trot.afqmc import Afqmc
from trot.core.ops import k_energy, k_force_bias
from trot.meas.uhf import (
    build_meas_ctx,
    energy_kernel_gw_rh,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_gw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    force_bias_kernel_uw_rh_spin,
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


def test_spin_decomposition_force_bias_components_uhf():
    norb = 6
    nup, ndn = 2, 1
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(12)
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

    meas_spin = make_uhf_meas_ops(sys, hs_decomposition="spin")
    fb_spin_selected = meas_spin.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        fb_charge = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        fb_spin = force_bias_kernel_uw_rh_spin(wi, ham, ctx, trial)
        fb_spin_from_ops = fb_spin_selected(wi, ham, ctx, trial)

        fb_a = fb_spin[:n_chol]
        fb_b = fb_spin[n_chol : 2 * n_chol]
        fb_s = fb_spin[2 * n_chol :]

        assert fb_spin.shape == (3 * n_chol,)
        assert jnp.allclose(fb_spin, fb_spin_from_ops, atol=1e-12)
        assert jnp.allclose(fb_charge, fb_a + fb_b, atol=1e-12)
        assert jnp.allclose(fb_s, fb_a - fb_b, atol=1e-12)


def test_spin_decomposition_force_bias_rejects_non_unrestricted_uhf():
    key = jax.random.PRNGKey(13)
    sys, *_ = testing.make_common_manual_only(
        key,
        "restricted",
        6,
        (2, 2),
        8,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(norb=6, nup=2, ndn=2),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    with pytest.raises(NotImplementedError):
        make_uhf_meas_ops(sys, hs_decomposition="spin")


def test_setup_wires_spin_decomposition_for_uhf():
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
        params=QmcParams(dt=0.01, n_walkers=1, n_blocks=1, n_prop_steps=0, n_eql_blocks=0),
    )
    rdm1 = job.trial_ops.get_rdm1(job.trial_data)
    prop_ctx = job.prop_ops.build_prop_ctx(job.ham_data, rdm1, job.params)
    meas_ctx = job.meas_ops.build_meas_ctx(job.ham_data, job.trial_data)
    walker = testing.make_walkers(jax.random.PRNGKey(14), job.sys)
    force_bias = job.meas_ops.require_kernel(k_force_bias)(
        walker, job.ham_data, meas_ctx, job.trial_data
    )

    assert prop_ctx.mf_shifts.shape == (3,)
    assert prop_ctx.force_bias_scales.shape == (3,)
    assert force_bias.shape == (3,)


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
