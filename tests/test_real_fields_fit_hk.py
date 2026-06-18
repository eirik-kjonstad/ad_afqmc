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
    _build_charge_spin_real_fields_from_eri,
    _build_kanamori_uj_real_fields_from_eri,
    _build_kanamori_real_fields_from_eri,
    _build_kanamori_sign_real_fields_from_eri,
    _build_kanamori_sign_full_real_fields_from_eri,
    _build_local_exact_real_fields_from_eri,
    _normalize_real_field_centers,
    analyze_hk_from_fcidump,
    build_model_fcidump_from_fcidump,
    fcidump_pair_spectrum,
    _reconstruct_packed_pair_from_fields,
    _factorize_uhf_charge_spin_blocked_supermatrix,
    _factorize_uhf_charge_spin_supermatrix,
    _stage_ham_input_from_fcidump,
    _stage_uhf_charge_spin_ham_input_from_fcidump,
    _stage_uhf_local_real_then_charge_spin_unrham_from_fcidump,
    _uhf_charge_spin_supermatrix_from_eri,
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


def test_hk_density_fit_can_extract_ligand_singleton_onsite_u():
    norb = 3
    eri = np.zeros((norb, norb, norb, norb))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 2.0
    eri[0, 0, 1, 1] = 0.7
    eri[1, 1, 0, 0] = 0.7

    fit = _build_hk_density_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0,), (1,)),
        chol_cut=1.0e-12,
    )

    assert fit.metadata["n_hk_real_fields"] == 2
    assert fit.metadata["n_residual_complex_fields"] > 0
    np.testing.assert_allclose(fit.chol[:2], np.asarray([np.diag([2.0, 0.0, 0.0]), np.diag([0.0, np.sqrt(2.0), 0.0])]))
    np.testing.assert_allclose(fit.field_factors[:2], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(
        fit.field_spin_coeffs[:2],
        np.asarray([[1.0, -1.0], [1.0, -1.0]]),
    )


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


def test_charge_spin_formal_fit_reconstructs_local_block_as_charge_sector():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_charge_spin_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    assert fit.metadata["real_field_fit"] == "charge_spin"
    assert fit.metadata["charge_spin_blocks"]["spin_pair_norm"] == 0.0
    assert fit.metadata["charge_spin_blocks"]["mixed_pair_norm"] == 0.0
    report = fit.metadata["center_reports"][0]
    assert report["n_spin_fields"] == 0
    assert report["n_mixed_charge_spin_fields"] == 0
    np.testing.assert_allclose(report["spin_orbital_reconstruction_relative_error"], 0.0, atol=1.0e-10)

    n_local = report["n_charge_fields"]
    got = _reconstruct_packed_pair_from_fields(
        fit.chol[:n_local],
        fit.field_factors[:n_local],
        fit.field_spin_coeffs[:n_local],
    )
    np.testing.assert_allclose(got, pair, rtol=1.0e-10, atol=1.0e-10)
    assert fit.metadata["n_residual_real_fields"] == 0
    assert fit.metadata["n_residual_complex_fields"] == 0


def test_charge_spin_staging_without_centers_decomposes_full_tensor():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    obj = SimpleNamespace(
        mf=SimpleNamespace(
            kind="uhf",
            afqmc_frozen=0,
            mo_coeff=np.stack([np.eye(norb), np.eye(norb)]),
        ),
        source="mf",
    )

    ham = _stage_ham_input_from_fcidump(
        obj,
        fcidump={
            "NORB": norb,
            "NELEC": 2,
            "MS2": 0,
            "ECORE": 0.0,
            "H1": np.zeros((norb, norb)),
            "H2": pair,
        },
        chol_cut=1.0e-12,
        verbose=False,
        real_field_centers=None,
        real_field_method="charge_spin",
    )

    meta = ham.field_metadata
    assert meta["real_field_fit"] == "charge_spin"
    assert meta["decomposition_scope"] == "full"
    assert meta["centers"] == []
    assert meta["n_local_real_fields"] == 0
    assert meta["n_local_complex_fields"] == 0
    assert meta["n_residual_real_fields"] == 0
    assert meta["n_residual_complex_fields"] == 0
    assert meta["n_full_real_fields"] + meta["n_full_complex_fields"] == ham.chol.shape[0]
    assert meta["n_full_charge_dominant_fields"] == ham.chol.shape[0]
    assert all(label.startswith("charge_spin_full_") for label in meta["field_labels"])

    got = _reconstruct_packed_pair_from_fields(
        ham.chol,
        ham.field_factors,
        ham.field_spin_coeffs,
    )
    np.testing.assert_allclose(got, pair, rtol=1.0e-10, atol=1.0e-10)
    assert meta["frobenius"]["full_pair_reconstruction_relative_error"] < 1.0e-10


def test_uhf_charge_spin_factorization_reconstructs_alpha_beta_blocks():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)
    theta_a = 0.17
    theta_b = -0.23
    ca = np.asarray(
        [
            [np.cos(theta_a), -np.sin(theta_a)],
            [np.sin(theta_a), np.cos(theta_a)],
        ]
    )
    cb = np.asarray(
        [
            [np.cos(theta_b), -np.sin(theta_b)],
            [np.sin(theta_b), np.cos(theta_b)],
        ]
    )

    cs, _diagnostics = _uhf_charge_spin_supermatrix_from_eri(
        eri,
        coeff_alpha=ca,
        coeff_beta=cb,
    )
    chol, factors, _labels, diagnostics = _factorize_uhf_charge_spin_supermatrix(
        cs,
        norb=norb,
        chol_cut=1.0e-12,
        label_prefix="test",
    )

    npair = norb * (norb + 1) // 2
    j_spin = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    transform = np.kron(j_spin, np.eye(npair))
    ref_ab = transform @ cs @ transform.T
    got_ab = np.zeros_like(ref_ab, dtype=np.complex128)
    coeff = -(factors * factors)
    for coeff_i, chol_i in zip(coeff, chol):
        alpha = chol_i[:norb, :norb]
        beta = chol_i[norb:, norb:]
        alpha_pair = np.asarray([alpha[m, n] for m in range(norb) for n in range(m + 1)])
        beta_pair = np.asarray([beta[m, n] for m in range(norb) for n in range(m + 1)])
        vec = np.concatenate([alpha_pair, beta_pair])
        got_ab += coeff_i * np.outer(vec, vec)

    np.testing.assert_allclose(got_ab, ref_ab, rtol=1.0e-10, atol=1.0e-10)
    assert diagnostics["test_pair_reconstruction_relative_error"] < 1.0e-10


def test_uhf_charge_spin_blocked_factorization_reconstructs_alpha_beta_blocks():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)
    ca = np.asarray(
        [
            [np.cos(0.17), -np.sin(0.17)],
            [np.sin(0.17), np.cos(0.17)],
        ]
    )
    cb = np.asarray(
        [
            [np.cos(-0.23), -np.sin(-0.23)],
            [np.sin(-0.23), np.cos(-0.23)],
        ]
    )
    cs, _diagnostics = _uhf_charge_spin_supermatrix_from_eri(
        eri,
        coeff_alpha=ca,
        coeff_beta=cb,
    )
    chol, factors, labels, diagnostics = _factorize_uhf_charge_spin_blocked_supermatrix(
        cs,
        norb=norb,
        chol_cut=1.0e-12,
        label_prefix="test",
    )

    npair = norb * (norb + 1) // 2
    transform = np.kron(np.asarray([[1.0, 1.0], [1.0, -1.0]]), np.eye(npair))
    ref_ab = transform @ cs @ transform.T
    got_ab = np.zeros_like(ref_ab, dtype=np.complex128)
    coeff = -(factors * factors)
    for coeff_i, chol_i in zip(coeff, chol):
        alpha = chol_i[:norb, :norb]
        beta = chol_i[norb:, norb:]
        alpha_pair = np.asarray([alpha[m, n] for m in range(norb) for n in range(m + 1)])
        beta_pair = np.asarray([beta[m, n] for m in range(norb) for n in range(m + 1)])
        vec = np.concatenate([alpha_pair, beta_pair])
        got_ab += coeff_i * np.outer(vec, vec)

    np.testing.assert_allclose(got_ab, ref_ab, rtol=1.0e-10, atol=1.0e-10)
    assert diagnostics["test_pair_reconstruction_relative_error"] < 1.0e-10
    assert any(label.startswith("test_complex:charge_") for label in labels)
    assert any(label.startswith("test_complex:spin_") for label in labels)
    assert any(label.startswith("test_real:mixed_residual_") for label in labels)


def test_uhf_charge_spin_supermatrix_stays_psd_for_spin_dependent_basis():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    eri = ao2mo.restore(1, pair, norb)
    theta_a = 0.17
    theta_b = -0.23
    ca = np.asarray(
        [
            [np.cos(theta_a), -np.sin(theta_a)],
            [np.sin(theta_a), np.cos(theta_a)],
        ]
    )
    cb = np.asarray(
        [
            [np.cos(theta_b), -np.sin(theta_b)],
            [np.sin(theta_b), np.cos(theta_b)],
        ]
    )

    cs, _diagnostics = _uhf_charge_spin_supermatrix_from_eri(
        eri,
        coeff_alpha=ca,
        coeff_beta=cb,
    )
    eigvals = np.linalg.eigvalsh(cs)

    assert np.min(eigvals) > -1.0e-10


def test_uhf_charge_spin_staging_without_centers_decomposes_full_kernel():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    theta_a = 0.17
    theta_b = -0.23
    ca = np.asarray(
        [
            [np.cos(theta_a), -np.sin(theta_a)],
            [np.sin(theta_a), np.cos(theta_a)],
        ]
    )
    cb = np.asarray(
        [
            [np.cos(theta_b), -np.sin(theta_b)],
            [np.sin(theta_b), np.cos(theta_b)],
        ]
    )
    obj = SimpleNamespace(
        mf=SimpleNamespace(
            kind="uhf",
            afqmc_frozen=0,
            mo_coeff=np.stack([ca, cb]),
        ),
        source="mf",
    )
    ham = _stage_uhf_charge_spin_ham_input_from_fcidump(
        obj,
        fcidump={
            "NORB": norb,
            "NELEC": 2,
            "MS2": 0,
            "ECORE": 0.0,
            "H1": np.zeros((norb, norb)),
            "H2": pair,
        },
        chol_cut=1.0e-12,
        verbose=False,
        real_field_centers=None,
    )

    meta = ham.field_metadata
    assert meta["decomposition_scope"] == "full"
    assert meta["centers"] == []
    assert meta["n_local_real_fields"] == 0
    assert meta["n_local_complex_fields"] == 0
    assert meta["n_residual_real_fields"] == 0
    assert meta["n_residual_complex_fields"] == 0
    assert meta["n_full_real_fields"] + meta["n_full_complex_fields"] == ham.chol.shape[0]
    assert meta["frobenius"]["full_pair_reconstruction_relative_error"] < 1.0e-10
    assert all(label.startswith("uhf_charge_spin_full_") for label in meta["field_labels"])


def test_uhf_charge_spin_unrham_matches_generalized_block_representation():
    norb = 2
    pair = np.asarray(
        [
            [4.0, 0.20, 0.60],
            [0.20, 3.0, 0.10],
            [0.60, 0.10, 2.0],
        ]
    )
    pair = 0.5 * (pair + pair.T)
    theta_a = 0.17
    theta_b = -0.23
    ca = np.asarray(
        [
            [np.cos(theta_a), -np.sin(theta_a)],
            [np.sin(theta_a), np.cos(theta_a)],
        ]
    )
    cb = np.asarray(
        [
            [np.cos(theta_b), -np.sin(theta_b)],
            [np.sin(theta_b), np.cos(theta_b)],
        ]
    )
    obj = SimpleNamespace(
        mf=SimpleNamespace(
            kind="uhf",
            afqmc_frozen=0,
            mo_coeff=np.stack([ca, cb]),
        ),
        source="mf",
    )
    fcidump = {
        "NORB": norb,
        "NELEC": 2,
        "MS2": 0,
        "ECORE": 0.0,
        "H1": np.diag([0.5, 0.7]),
        "H2": pair,
    }

    ham_g = _stage_uhf_charge_spin_ham_input_from_fcidump(
        obj,
        fcidump=fcidump,
        chol_cut=1.0e-12,
        verbose=False,
        real_field_centers=None,
    )
    ham_u = _stage_uhf_charge_spin_ham_input_from_fcidump(
        obj,
        fcidump=fcidump,
        chol_cut=1.0e-12,
        verbose=False,
        real_field_centers=None,
        unrestricted_ham=True,
    )

    assert ham_g.basis == "generalized"
    assert ham_u.basis == "unrestricted"
    assert ham_u.h1.shape == (2, norb, norb)
    assert ham_u.chol.shape == (ham_g.chol.shape[0], 2, norb, norb)
    np.testing.assert_allclose(ham_u.field_factors, ham_g.field_factors)
    np.testing.assert_allclose(ham_u.h1[0], ham_g.h1[:norb, :norb])
    np.testing.assert_allclose(ham_u.h1[1], ham_g.h1[norb:, norb:])
    np.testing.assert_allclose(ham_u.chol[:, 0], ham_g.chol[:, :norb, :norb])
    np.testing.assert_allclose(ham_u.chol[:, 1], ham_g.chol[:, norb:, norb:])
    assert ham_u.field_metadata["real_field_fit"] == "uhf_charge_spin_unrham"
    assert ham_u.field_metadata["basis"] == "uhf_alpha_beta_mo_unrestricted"


def test_uhf_local_real_then_charge_spin_unrham_extracts_onsite_real_field():
    norb = 1
    obj = SimpleNamespace(
        mf=SimpleNamespace(
            kind="uhf",
            afqmc_frozen=0,
            mo_coeff=np.stack([np.eye(norb), np.eye(norb)]),
        ),
        source="mf",
    )

    ham = _stage_uhf_local_real_then_charge_spin_unrham_from_fcidump(
        obj,
        fcidump={
            "NORB": norb,
            "NELEC": 2,
            "MS2": 0,
            "ECORE": 0.0,
            "H1": np.zeros((norb, norb)),
            "H2": np.asarray([[4.0]]),
        },
        chol_cut=1.0e-12,
        verbose=False,
        real_field_centers=((0,),),
    )

    assert ham.basis == "unrestricted"
    assert ham.field_metadata["real_field_fit"] == "uhf_local_real_then_charge_spin_unrham"
    assert ham.field_metadata["n_local_real_fields"] == 1
    assert ham.field_metadata["n_residual_real_fields"] == 0
    assert ham.field_metadata["n_residual_complex_fields"] == 0
    np.testing.assert_allclose(ham.field_factors, np.asarray([1.0 + 0.0j]))
    np.testing.assert_allclose(ham.chol[:, 0], np.asarray([[[2.0]]]))
    np.testing.assert_allclose(ham.chol[:, 1], np.asarray([[[-2.0]]]))


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


def test_kanamori_real_prefers_real_hund_identity_channels():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[0, 2] = pair[2, 0] = 0.8
    pair[1, 1] = 0.5
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_kanamori_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    report = fit.metadata["center_reports"][0]
    assert fit.metadata["real_field_fit"] == "kanamori_real"
    assert fit.metadata["n_local_complex_fields"] == 0
    assert report["spin_orbital_reconstruction_relative_error"] < 1.0e-10
    pair_term = next(term for term in report["kanamori_terms"] if term["kind"] == "pair_full")
    assert pair_term["decomposition"] == "real_spin_orbital_nnls"
    assert all(component["real_field"] for component in pair_term["components"])
    assert any(
        component["component"] == "hund_spin_real" and component["channel"] == "spin"
        for component in pair_term["components"]
    )
    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)


def test_kanamori_uj_extracts_u_and_j_but_leaves_uprime_in_residual():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[0, 2] = pair[2, 0] = 0.8
    pair[1, 1] = 0.5
    eri = ao2mo.restore(1, pair, norb)

    fit = _build_kanamori_uj_real_fields_from_eri(
        eri,
        basis_coeff=np.eye(norb),
        centers=((0, 1),),
        chol_cut=1.0e-12,
    )

    report = fit.metadata["center_reports"][0]
    assert fit.metadata["real_field_fit"] == "kanamori_uj"
    assert {term["kind"] for term in report["kanamori_terms"]} == {
        "onsite_U",
        "hund_J_pair",
    }
    n_local = fit.metadata["n_local_real_fields"] + fit.metadata["n_local_complex_fields"]
    local_pair = _reconstruct_packed_pair_from_fields(
        fit.chol[:n_local],
        fit.field_factors[:n_local],
        fit.field_spin_coeffs[:n_local],
    )
    expected_local = np.zeros_like(pair)
    expected_local[0, 0] = pair[0, 0]
    expected_local[1, 1] = pair[1, 1]
    expected_local[2, 2] = pair[2, 2]
    np.testing.assert_allclose(local_pair, expected_local, rtol=1.0e-10, atol=1.0e-10)

    pair_got = _reconstruct_packed_pair_from_fields(
        fit.chol,
        fit.field_factors,
        fit.field_spin_coeffs,
    )
    np.testing.assert_allclose(pair_got, pair, rtol=1.0e-10, atol=1.0e-10)
    assert report["parameters"]["interorbital_Uprime"][0]["Uprime"] == 0.8
    assert report["residual_center_block_norm"] > 0.0


def test_analyze_hk_from_fcidump_reports_method_weight_hierarchy():
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[2, 2] = 3.0
    pair[0, 2] = pair[2, 0] = 0.8
    pair[1, 1] = 0.5
    eri = ao2mo.restore(1, pair, norb)
    ctx = {"NORB": norb, "NELEC": 2, "H1": np.zeros((norb, norb)), "H2": eri}

    analysis = analyze_hk_from_fcidump(ctx, centers=((0, 1),))

    assert analysis["centers"] == [[0, 1]]
    methods = analysis["method_weight_fractions"]
    assert methods["hk_density"] < methods["kanamori_uj"]
    assert methods["kanamori_uj"] < methods["kanamori_sign_like"]
    assert methods["kanamori_sign_like"] == methods["local_exact"]
    assert analysis["local_weight_fractions"]["local_other"] == 0.0


def test_analyze_hk_from_fcidump_reports_ligand_real_field_diagnostics():
    norb = 3
    pair = np.zeros((6, 6))
    fe_pp = 0
    lig_pp = 2
    other_pp = 5
    pair[fe_pp, fe_pp] = 4.0
    pair[lig_pp, lig_pp] = 2.0
    pair[other_pp, other_pp] = 1.0
    pair[fe_pp, lig_pp] = pair[lig_pp, fe_pp] = 4.0
    eri = ao2mo.restore(1, pair, norb)
    ctx = {"NORB": norb, "NELEC": 2, "H1": np.eye(norb), "H2": eri}

    analysis = analyze_hk_from_fcidump(ctx, centers=((0,),), ligand_centers=((1,),))

    assert analysis["ligand_centers"] == [[1]]
    assert analysis["method_weight_fractions"]["hk_density_fe_s_onsite"] > analysis[
        "method_weight_fractions"
    ]["hk_density"]
    assert analysis["buckets"]["real_ligand_onsite_U"]["norm"] == 2.0
    assert analysis["buckets"]["fe_ligand_bridge"]["norm"] > 0.0
    assert (
        analysis["buckets"]["fe_ligand_bridge_real_extractable"][
            "relative_weight_fraction"
        ]
        == 0.0
    )
    assert (
        analysis["buckets"]["fe_ligand_bridge_complex_or_residual"][
            "relative_weight_fraction"
        ]
        == 1.0
    )


def test_build_model_fcidump_from_fcidump_writes_reduced_onsite_bridge_model(tmp_path):
    norb = 3
    pair = np.zeros((6, 6))
    fe_pp = 0
    lig_pp = 2
    other_pp = 5
    pair[fe_pp, fe_pp] = 4.0
    pair[lig_pp, lig_pp] = 2.0
    pair[other_pp, other_pp] = 1.0
    pair[fe_pp, lig_pp] = pair[lig_pp, fe_pp] = 4.0
    pair[fe_pp, other_pp] = pair[other_pp, fe_pp] = 0.25
    eri = ao2mo.restore(1, pair, norb)
    ctx = {"NORB": norb, "NELEC": 2, "MS2": 0, "H1": np.eye(norb), "H2": eri}
    out = tmp_path / "model.FCIDUMP"

    result = build_model_fcidump_from_fcidump(
        ctx,
        out,
        model_orbitals=(0, 1),
        fe_centers=((0,),),
        ligand_centers=((1,),),
        h2_model="onsite_bridge_density",
        nelec=2,
        ms2=0,
    )

    from pyscf.tools import fcidump

    model = fcidump.read(str(out))
    assert result["norb"] == 2
    assert model["NORB"] == 2
    assert model["NELEC"] == 2
    model_pair = ao2mo.restore(4, model["H2"], 2)
    expected = np.zeros((3, 3))
    expected[0, 0] = 4.0
    expected[2, 2] = 2.0
    expected[0, 2] = expected[2, 0] = 4.0
    np.testing.assert_allclose(model_pair, expected)
    spectrum = fcidump_pair_spectrum(out)
    assert spectrum["n_negative_eigenvalues"] == 1
    assert spectrum["min_eigenvalue"] < 0.0


def test_build_model_fcidump_can_fold_reference_fock_shift_into_h1(tmp_path):
    norb = 2
    pair = np.zeros((3, 3))
    pair[0, 0] = 4.0
    pair[0, 2] = pair[2, 0] = 0.5
    eri = ao2mo.restore(1, pair, norb)
    ctx = {"NORB": norb, "NELEC": 1, "MS2": 1, "H1": np.zeros((norb, norb)), "H2": eri}
    out = tmp_path / "model.FCIDUMP"

    result = build_model_fcidump_from_fcidump(
        ctx,
        out,
        model_orbitals=(0,),
        fe_centers=((0,),),
        h2_model="onsite",
        h1_correction="reference_fock",
        reference_occupations=(np.asarray([1.0, 1.0]), np.asarray([0.0, 1.0])),
        nelec=1,
        ms2=1,
    )

    from pyscf.tools import fcidump

    model = fcidump.read(str(out))
    np.testing.assert_allclose(model["H1"], np.asarray([[1.0]]), atol=1.0e-12)
    np.testing.assert_allclose(result["h1_correction_norm"], 1.0, atol=1.0e-12)


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


def test_generalized_auto_energy_uses_field_factor_quadratic_sign():
    sys = System(norb=1, nelec=(1, 1), walker_kind="generalized")
    trial_data = make_uhf_trial_data({"mo_a": np.eye(1), "mo_b": np.eye(1)}, sys)
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_auto_meas_ops(sys=sys, trial_ops_=trial_ops, eps=1.0e-2)
    walker = jnp.eye(2, dtype=jnp.complex128)

    def energy_for_factor(factor: complex):
        ham = HamChol(
            basis="generalized",
            h0=jnp.asarray(0.0),
            h1=jnp.zeros((2, 2)),
            chol=jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
            field_factors=jnp.asarray([factor]),
        )
        meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
        return meas_ops.require_kernel(k_energy)(walker, ham, meas_ctx, trial_data)

    np.testing.assert_allclose(
        np.asarray(energy_for_factor(1.0j)),
        1.0,
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(energy_for_factor(1.0 + 0.0j)),
        -1.0,
        rtol=1.0e-3,
        atol=1.0e-3,
    )


def test_unrestricted_auto_energy_uses_field_factor_quadratic_sign():
    sys = System(norb=1, nelec=(1, 1), walker_kind="unrestricted")
    trial_data = make_uhf_trial_data({"mo_a": np.eye(1), "mo_b": np.eye(1)}, sys)
    trial_ops = make_uhf_trial_ops(sys)
    meas_ops = make_auto_meas_ops(sys=sys, trial_ops_=trial_ops, eps=1.0e-2)
    walker = (jnp.eye(1, dtype=jnp.complex128), jnp.eye(1, dtype=jnp.complex128))

    def energy_for_factor(factor: complex):
        ham = HamChol(
            basis="unrestricted",
            h0=jnp.asarray(0.0),
            h1=jnp.zeros((2, 1, 1)),
            chol=jnp.asarray([[[[1.0]], [[1.0]]]]),
            field_factors=jnp.asarray([factor]),
        )
        meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)
        return meas_ops.require_kernel(k_energy)(walker, ham, meas_ctx, trial_data)

    np.testing.assert_allclose(
        np.asarray(energy_for_factor(1.0j)),
        1.0,
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(energy_for_factor(1.0 + 0.0j)),
        -1.0,
        rtol=1.0e-3,
        atol=1.0e-3,
    )


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


def test_trotter_ops_apply_unrestricted_ham_fields_to_unrestricted_walkers():
    chol = jnp.asarray([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.5]]]])
    ham = HamChol(
        basis="unrestricted",
        h0=jnp.asarray(0.0),
        h1=jnp.zeros((2, 2, 2)),
        chol=chol,
        field_factors=jnp.asarray([1.0 + 0.0j]),
    )
    rdm1 = jnp.stack([jnp.diag(jnp.asarray([1.0, 0.0])), jnp.diag(jnp.asarray([1.0, 0.0]))])
    ctx = _build_prop_ctx(ham, rdm1, 0.01)
    ops = make_trotter_ops("unrestricted", "unrestricted")
    walker = (jnp.eye(2, 1, dtype=jnp.complex128), jnp.eye(2, 1, dtype=jnp.complex128))

    out_a, out_b = ops.apply_trotter(walker, jnp.asarray([0.2]), ctx, 4)

    assert out_a.shape == walker[0].shape
    assert out_b.shape == walker[1].shape
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))
