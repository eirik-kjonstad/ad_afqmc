import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace
from pyscf import ao2mo

from trot.core.ops import k_energy
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.auto import make_auto_meas_ops
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.runtime_layout import _build_restricted_prop_ctx_from_host
from trot.staging import (
    _factorize_symmetric_supermatrix,
    _build_hk_density_real_fields_from_eri,
    _build_kanamori_sign_real_fields_from_eri,
    _build_kanamori_sign_full_real_fields_from_eri,
    _build_local_exact_real_fields_from_eri,
    _normalize_real_field_centers,
    _reconstruct_packed_pair_from_fields,
)
from trot.trial.uhf import make_uhf_trial_data, make_uhf_trial_ops


def test_normalize_real_field_centers_accepts_ranges():
    centers = _normalize_real_field_centers("2:4,7:9", norb=10)
    assert centers == ((2, 3), (7, 8))


def test_hk_density_fit_extracts_real_spin_field_and_residual():
    norb = 2
    eri = np.zeros((norb, norb, norb, norb))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 1.5
    coeff = np.eye(norb)

    fit = _build_hk_density_real_fields_from_eri(
        eri,
        basis_coeff=coeff,
        centers=((0,),),
        chol_cut=1.0e-10,
    )

    assert fit.metadata["n_hk_real_fields"] == 1
    assert fit.chol.shape[1:] == (norb, norb)
    np.testing.assert_allclose(fit.chol[0], np.diag([2.0, 0.0]))
    np.testing.assert_allclose(fit.field_factors[0], 1.0)
    np.testing.assert_allclose(fit.field_spin_coeffs[0], [1.0, -1.0])
    np.testing.assert_allclose(fit.h1_shift, np.zeros((norb, norb)))
    assert fit.metadata["extracted_terms"][0]["U"] == 4.0
    frob = fit.metadata["frobenius"]
    np.testing.assert_allclose(frob["full_norm"], np.sqrt(4.0**2 + 1.5**2))
    np.testing.assert_allclose(frob["hk_onsite_norm"], 4.0)
    np.testing.assert_allclose(frob["center_block_norm"], 4.0)
    np.testing.assert_allclose(frob["hk_fraction_center_block_weight"], 1.0)


def test_local_exact_one_orbital_center_reproduces_onsite_u():
    norb = 1
    eri = np.zeros((norb, norb, norb, norb))
    eri[0, 0, 0, 0] = 4.0

    fit = _build_local_exact_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0,),),
        chol_cut=1.0e-12,
    )

    assert fit.metadata["real_field_fit"] == "local_exact"
    assert fit.metadata["n_local_real_fields"] == 1
    assert fit.metadata["n_local_complex_fields"] == 0
    np.testing.assert_allclose(fit.chol[0], np.asarray([[2.0]]), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(fit.field_factors[0], 1.0)
    np.testing.assert_allclose(fit.field_spin_coeffs[0], [1.0, -1.0])

    pair_ref = ao2mo.restore(4, eri, norb)
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair_ref, rtol=1.0e-12, atol=1.0e-12)
    assert fit.metadata["center_reports"][0]["packed_pair_reconstruction_relative_error"] < 1.0e-12


def test_local_exact_two_orbital_kanamori_like_block_reconstructs_exactly():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 0.50, 0.15],
            [0.60, 0.15, 3.6],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_local_exact_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)
    report = fit.metadata["center_reports"][0]
    assert report["packed_pair_reconstruction_relative_error"] < 1.0e-10
    assert report["parameters"]["onsite_U"][0]["U"] == 4.0
    assert len(report["onsite_real_terms"]) == 2
    assert fit.metadata["n_local_real_fields"] == 2
    assert fit.metadata["n_residual_real_fields"] == 0
    assert fit.metadata["n_residual_complex_fields"] == 0


def test_local_exact_extracts_onsite_real_before_local_residual_modes():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.25, 0.70],
            [0.25, 0.80, 0.10],
            [0.70, 0.10, 3.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_local_exact_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    np.testing.assert_allclose(fit.chol[0], np.diag([2.0, 0.0]), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        fit.chol[1], np.diag([0.0, np.sqrt(3.0)]), rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(fit.field_factors[:2], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(fit.field_spin_coeffs[:2], np.asarray([[1.0, -1.0], [1.0, -1.0]]))
    assert fit.metadata["n_local_real_fields"] == 2
    assert fit.metadata["n_local_complex_fields"] > 0

    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)
    assert fit.metadata["center_reports"][0]["packed_pair_reconstruction_relative_error"] < 1.0e-10


def test_local_exact_subtracts_only_same_center_block_and_leaves_cross_terms():
    norb = 2
    eri = np.zeros((norb, norb, norb, norb))
    eri[0, 0, 0, 0] = 2.0
    eri[0, 0, 1, 1] = 0.7
    eri[1, 1, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 1.1

    fit = _build_local_exact_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0,),),
        chol_cut=1.0e-12,
    )

    pair_ref = ao2mo.restore(4, eri, norb)
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair_ref, rtol=1.0e-10, atol=1.0e-10)
    assert fit.metadata["center_reports"][0]["residual_center_block_norm"] == 0.0
    assert fit.metadata["n_residual_complex_fields"] > 0


def test_kanamori_sign_falls_back_to_exact_charge_bond_for_positive_hund_j():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[0, 2] = pair[2, 0] = 0.8
    pair[1, 1] = 0.5
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_kanamori_sign_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    assert fit.metadata["real_field_fit"] == "kanamori_sign"
    assert fit.metadata["n_local_complex_fields"] > 0
    labels = fit.metadata["field_labels"]
    hund_label = next(label for label in labels if "hund0-1" in label)
    assert hund_label.endswith(":charge")
    hund_term = next(
        term
        for term in fit.metadata["center_reports"][0]["kanamori_terms"]
        if term["kind"] == "hund_J_bond"
    )
    assert hund_term["preferred_decomposition"] == "spin"
    assert hund_term["decomposition"] == "charge"
    assert fit.metadata["center_reports"][0]["spin_orbital_reconstruction_relative_error"] < 1.0e-10
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)


def test_kanamori_sign_uses_charge_bond_for_negative_hund_j():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[1, 1] = -0.5
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_kanamori_sign_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    labels = fit.metadata["field_labels"]
    hund_label = next(label for label in labels if "hund0-1" in label)
    assert hund_label.endswith(":charge")
    assert fit.metadata["center_reports"][0]["spin_orbital_reconstruction_relative_error"] < 1.0e-10
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)


def test_kanamori_sign_full_reconstructs_pair_block_in_spin_orbital_space():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[0, 2] = pair[2, 0] = 0.8
    pair[1, 1] = 0.5
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_kanamori_sign_full_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    report = fit.metadata["center_reports"][0]
    assert fit.metadata["real_field_fit"] == "kanamori_sign_full"
    assert report["spin_orbital_reconstruction_relative_error"] < 1.0e-10
    assert report["packed_pair_reconstruction_relative_error"] < 1.0e-10
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)


def test_residual_factorization_reconstructs_packed_pair_matrix():
    norb = 3
    pair = np.asarray(
        [
            [1.2, 0.1, -0.2, 0.0, 0.3, 0.2],
            [0.1, 0.7, 0.4, -0.1, 0.2, 0.0],
            [-0.2, 0.4, 1.4, 0.3, -0.2, 0.1],
            [0.0, -0.1, 0.3, 0.9, 0.2, -0.3],
            [0.3, 0.2, -0.2, 0.2, 1.1, 0.4],
            [0.2, 0.0, 0.1, -0.3, 0.4, 0.8],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)

    chol, factors, _spin_coeffs, _labels, diagnostics = _factorize_symmetric_supermatrix(
        eri,
        coeff=np.eye(norb),
        chol_cut=1.0e-12,
    )

    chol_pair = []
    for chol_i in chol:
        chol_pair.append([chol_i[m, n] for m in range(norb) for n in range(m + 1)])
    chol_pair = np.asarray(chol_pair)
    coeff = -(factors * factors)
    reconstructed = np.einsum("g,gi,gj->ij", coeff, chol_pair, chol_pair)

    np.testing.assert_allclose(reconstructed, pair, rtol=1.0e-10, atol=1.0e-10)
    assert diagnostics["residual_pair_reconstruction_relative_error"] < 1.0e-10


def test_hk_real_spin_field_local_energy_has_no_extra_one_body_shift():
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((1, 1)),
        chol=jnp.asarray([[[2.0]]]),
        field_factors=jnp.asarray([1.0 + 0.0j]),
        field_spin_coeffs=jnp.asarray([[1.0, -1.0]]),
    )
    sys = System(norb=1, nelec=(1, 1), walker_kind="unrestricted")
    trial_data = make_uhf_trial_data({"mo_a": np.eye(1), "mo_b": np.eye(1)}, sys)
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_auto_meas_ops(sys=sys, trial_ops_=trial_ops)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
    energy = meas_ops.require_kernel(k_energy)(
        (jnp.eye(1, dtype=jnp.complex128), jnp.eye(1, dtype=jnp.complex128)),
        ham,
        meas_ctx,
        trial_data,
    )

    np.testing.assert_allclose(np.asarray(energy), 4.0, rtol=1.0e-6, atol=1.0e-6)


def test_mixed_real_complex_prop_ctx_uses_spin_resolved_h1():
    chol = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.5]],
        ]
    )
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((2, 2)),
        chol=chol,
        field_factors=jnp.asarray([1.0 + 0.0j, 1.0j]),
        field_spin_coeffs=jnp.asarray([[1.0, -1.0], [1.0, 1.0]]),
    )
    rdm1 = jnp.stack([jnp.diag(jnp.asarray([1.0, 0.0])), jnp.diag(jnp.asarray([0.0, 1.0]))])

    ctx = _build_prop_ctx(ham, rdm1, 0.01)

    assert ctx.mf_shifts.shape == (2,)
    assert ctx.exp_h1_half.shape == (2, 2, 2)
    np.testing.assert_allclose(np.asarray(ctx.field_factors), np.asarray([1.0 + 0.0j, 1.0j]))


def test_host_prop_ctx_matches_mixed_field_device_builder():
    chol = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.5]],
        ]
    )
    h1 = np.zeros((2, 2))
    field_factors = np.asarray([1.0 + 0.0j, 1.0j])
    field_spin_coeffs = np.asarray([[1.0, -1.0], [1.0, 1.0]])
    rdm1 = jnp.stack([jnp.diag(jnp.asarray([1.0, 0.0])), jnp.diag(jnp.asarray([0.0, 1.0]))])
    staged = SimpleNamespace(
        ham=SimpleNamespace(
            h0=0.0,
            h1=h1,
            chol=chol,
            field_factors=field_factors,
            field_spin_coeffs=field_spin_coeffs,
        )
    )
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.asarray(h1),
        chol=jnp.asarray(chol),
        field_factors=jnp.asarray(field_factors),
        field_spin_coeffs=jnp.asarray(field_spin_coeffs),
    )

    host_ctx = _build_restricted_prop_ctx_from_host(
        staged, trial_rdm1=rdm1, dt=0.01, mixed_precision=False, mesh=None
    )
    device_ctx = _build_prop_ctx(ham, rdm1, 0.01)

    np.testing.assert_allclose(np.asarray(host_ctx.mf_shifts), np.asarray(device_ctx.mf_shifts))
    np.testing.assert_allclose(np.asarray(host_ctx.h0_prop), np.asarray(device_ctx.h0_prop))
    np.testing.assert_allclose(np.asarray(host_ctx.exp_h1_half), np.asarray(device_ctx.exp_h1_half))


def test_trotter_ops_apply_spin_resolved_fields_to_unrestricted_walkers():
    chol = jnp.asarray([[[1.0, 0.0], [0.0, 0.0]]])
    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((2, 2)),
        chol=chol,
        field_factors=jnp.asarray([1.0 + 0.0j]),
        field_spin_coeffs=jnp.asarray([[1.0, -1.0]]),
    )
    rdm1 = jnp.stack([jnp.diag(jnp.asarray([1.0, 0.0])), jnp.diag(jnp.asarray([1.0, 0.0]))])
    ctx = _build_prop_ctx(ham, rdm1, 0.01)
    ops = make_trotter_ops("restricted", "unrestricted")
    walker = (jnp.eye(2, 1, dtype=jnp.complex128), jnp.eye(2, 1, dtype=jnp.complex128))

    out_a, out_b = ops.apply_trotter(walker, jnp.asarray([0.2]), ctx, 4)

    assert out_a.shape == walker[0].shape
    assert out_b.shape == walker[1].shape
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))
