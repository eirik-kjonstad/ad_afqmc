import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.runtime_layout import _build_restricted_prop_ctx_from_host
from trot.staging import (
    _build_hk_density_real_fields_from_eri,
    _normalize_real_field_centers,
)


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
    np.testing.assert_allclose(fit.h1_shift, np.diag([2.0, 0.0]))
    assert fit.metadata["extracted_terms"][0]["U"] == 4.0


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
