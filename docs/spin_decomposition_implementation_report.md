# Spin Decomposition Implementation Report

This report explains the spin-decomposition implementation changes, excluding the example script.

## Overview

The implementation adds an opt-in AFQMC Hubbard-Stratonovich decomposition mode:

```python
hs_decomposition="spin"
```

The existing charge decomposition remains the default. The spin decomposition is currently restricted to restricted-basis Cholesky Hamiltonians with unrestricted walkers and UHF-family trials: `uhf`, `ucisd`, `ucisdt`, and `ucisdtq`.

The core design is:

- Keep `_stage_ham_input` unchanged because staging should still produce the same spatial Cholesky factors.
- Apply spin decomposition at propagation-context construction time, after the trial/reference RDM1 is available for mean-field subtraction.
- Use a generic automatic-differentiation force-bias wrapper for spin decomposition, so correlated unrestricted trials do not need separately derived alpha, beta, and spin-channel analytic force-bias kernels.

## Propagation Context Changes

File: `trot/prop/chol_afqmc_ops.py`

### Added `force_bias_scales` to `CholAfqmcCtx`

`CholAfqmcCtx` now stores:

```python
force_bias_scales: jax.Array
```

Why:

The old charge path hard-coded the force-bias operator prefactor as `1.0j` inside `afqmc_step`. Spin decomposition has three channel-specific prefactors:

- alpha channel: `i * sqrt(2)`
- beta channel: `i * sqrt(2)`
- spin channel: `-1`

Storing these scales in the propagation context lets the charge and spin paths share the same importance-sampling code while preserving their different operator definitions.

### Added spin-resolved mean-field shifts

New helpers compute spin-resolved trial/reference contractions:

```python
l_a = Tr[L_gamma dm_alpha]
l_b = Tr[L_gamma dm_beta]
```

and assemble the spin-channel mean-field vector:

```python
[
    i * sqrt(2) * l_a,
    i * sqrt(2) * l_b,
    -(l_a - l_b),
]
```

Why:

Mean-field subtraction must be applied to the shifted operators used by the HS transformation. For unrestricted references, the scalar background is naturally split spin-resolved:

```text
m_gamma_alpha = <v_gamma_alpha>_T
m_gamma_beta  = <v_gamma_beta>_T
```

This gives zero trial expectation for all three shifted channels and keeps the completion-of-the-square correction exact.

### Kept `h1_eff` on the charge mean-field correction

Even in spin mode, `h1_eff` is built from the total charge mean-field shift:

```python
mf_charge = i * Tr[L_gamma (dm_alpha + dm_beta)]
h1_eff = _get_h1_eff(ham_data, mf_charge)
```

Why:

The one-body correction comes from completing the square for the original charge operator. The spin decomposition is an algebraic rewrite of the residual two-body operator after that subtraction, so the induced one-body term should remain the same as the exact charge mean-field subtraction.

### Added spin-mode Trotter application

For spin decomposition, the shifted field vector is split into:

```python
x_alpha, x_beta, x_spin
```

and separate one-body matrices are formed for alpha and beta walkers:

```python
V_alpha = (i * sqrt(2) * x_alpha - x_spin) . L
V_beta  = (i * sqrt(2) * x_beta  + x_spin) . L
```

Why:

The spin channel corresponds to `i(u_alpha - u_beta)`. With the existing propagation convention, this produces opposite signs for alpha and beta walkers. The unrestricted walker path must therefore apply different two-body propagators to the two spin sectors.

### Added `hs_decomposition` to `make_trotter_ops`

`make_trotter_ops` now accepts:

```python
hs_decomposition: str = "charge"
```

Why:

Trotter application is where the sampled auxiliary fields become one-body rotations. The spin path needs a different field interpretation and is only valid for restricted Hamiltonians with unrestricted walkers, so this is the natural dispatch point.

## AFQMC Step Changes

File: `trot/prop/afqmc.py`

### Sample fields from `mf_shifts.shape[0]`

The field count now comes from:

```python
prop_ctx.mf_shifts.shape[0]
```

instead of:

```python
prop_ctx.chol_flat.shape[0]
```

Why:

In charge mode, both are `nchol`. In spin mode, the number of auxiliary fields is `3 * nchol`, while `chol_flat` remains the original `nchol` spatial Cholesky matrix storage.

### Use `force_bias_scales` in field shifts

The force-bias shift now uses:

```python
prop_ctx.force_bias_scales * force_bias
```

Why:

The measurement kernel returns raw contractions with the underlying Cholesky operators. The propagation context owns the operator prefactors for the selected HS decomposition. This keeps charge mode unchanged and applies the correct alpha, beta, and spin prefactors in spin mode.

### Added `hs_decomposition` to `make_prop_ops`

`make_prop_ops` passes the decomposition choice to both:

- `make_trotter_ops`
- `_build_prop_ctx`

Why:

The decomposition affects both context construction and Trotter application, so both must be configured consistently.

## Generic AD Spin Force Bias

File: `trot/meas/spin_decomp.py`

### Added `SpinDecompMeasCtx`

This context wraps the original measurement context:

```python
SpinDecompMeasCtx(base_ctx=...)
```

Why:

Only the force-bias kernel changes in spin mode. Energy estimators and observables should continue using the existing trial-specific measurement context. Wrapping avoids duplicating those contexts.

### Added `_force_bias_spin_uw`

This computes spin-resolved force-bias components by differentiating the trial overlap with respect to independent alpha and beta Cholesky rotations:

```python
fb_alpha = d log overlap / d x_alpha
fb_beta  = d log overlap / d x_beta
fb_spin_raw = fb_alpha - fb_beta
```

It returns:

```python
[fb_alpha, fb_beta, fb_alpha - fb_beta]
```

Why:

The existing UCISD and UCISDT analytic force-bias kernels return only the total charge-channel contraction. Spin decomposition needs three channel contractions. Deriving and maintaining separate analytic alpha, beta, and spin formulas for each correlated trial would be large and error-prone. AD gives a single correct implementation for UHF, UCISD, UCISDT, and UCISDTQ.

### Added `wrap_spin_decomp_meas_ops`

This wrapper:

- validates unrestricted walkers and supported trial kinds,
- replaces only the `force_bias` kernel,
- forwards energy kernels and observables to the original measurement ops using the wrapped base context.

Why:

Spin decomposition should change propagation force-bias behavior without changing local energy evaluation. This keeps existing trial-specific energy kernels intact.

## Setup And Driver Plumbing

Files:

- `trot/setup.py`
- `trot/afqmc.py`

### Added public `hs_decomposition` option

`Afqmc` and `setup` now accept:

```python
hs_decomposition="charge"
hs_decomposition="spin"
```

Why:

The new path is experimental and changes AFQMC sampling behavior, so it should be explicitly opt-in. Existing scripts keep using charge decomposition by default.

### Added validation and measurement wrapping in setup

When `hs_decomposition == "spin"`, setup:

- checks that the Hamiltonian basis is restricted,
- wraps measurement ops with `wrap_spin_decomp_meas_ops`,
- passes the decomposition setting into propagation construction.

Why:

Setup is where the staged Hamiltonian, trial kind, walker kind, measurement ops, and propagation ops are all available. That makes it the correct place to validate compatibility and wire the spin force-bias wrapper.

### Stored `hs_decomposition` on `Job`

`Job` now records the selected decomposition.

Why:

The selected decomposition is part of the runtime configuration. It is printed in the AFQMC flags and included in cached-job reuse checks so an existing job built with charge decomposition is not accidentally reused for spin decomposition.

## Host Runtime Compatibility

File: `trot/runtime_layout.py`

The host-built restricted propagation context now also constructs `force_bias_scales`.

Why:

Some RHF/CISD host-runtime layouts build `CholAfqmcCtx` directly instead of calling `_build_prop_ctx`. Adding the new field there preserves compatibility with the updated context dataclass.

This host path still uses charge decomposition. Spin decomposition is routed through the default runtime layout for the supported unrestricted UHF-family trials.

## Tests Added Or Updated

Files:

- `tests/test_prop_chol_afqmc_ops.py`
- `tests/test_spin_decomp.py`

### Propagation context tests

Added checks that spin mode builds:

- `3 * nchol` mean-field shifts,
- `3 * nchol` force-bias scales,
- unchanged `nchol` Cholesky storage,
- the same mean-field scalar correction as charge mode.

Why:

These tests directly protect the exact shifted-operator algebra.

### Trotter sign test

Added a test that isolates the spin channel and verifies it applies opposite signs to alpha and beta walkers.

Why:

The spin-channel sign is the easiest place to introduce an algebraic bug.

### Force-bias test

Added a UHF determinant test comparing the AD spin force bias against analytic reference contractions.

Why:

This validates the raw `[alpha, beta, alpha - beta]` force-bias convention used by the spin wrapper.

### Setup smoke test

Added a setup-level test for:

```python
setup(..., walker_kind="unrestricted", hs_decomposition="spin")
```

Why:

This verifies that public API plumbing, measurement wrapping, and propagation context construction work together.

## Validation Performed

The targeted validation suite passed:

```text
pytest tests/test_prop_chol_afqmc_ops.py tests/test_prop_afqmc_step.py tests/test_spin_decomp.py tests/test_uhf.py tests/test_ucisd.py tests/test_ucisdt.py tests/test_ucisdtq.py -q
52 passed, 5 skipped
```

Additional host-runtime regression checks passed:

```text
pytest tests/test_rhf.py tests/test_cisd.py tests/test_spin_decomp.py -q
32 passed, 1 skipped
```

Formatting and linting passed:

```text
ruff check ...
black --check --target-version py310 ...
```

## Important Limitations

- Spin decomposition is opt-in and currently implemented only for unrestricted walkers.
- The supported trial kinds are `uhf`, `ucisd`, `ucisdt`, and `ucisdtq`.
- The spin-decomposition force bias uses AD. This is intentional for UCISDT because no analytic spin-channel UCISDT force-bias kernel exists in the current implementation.
- The existing charge decomposition remains the default and should be behaviorally unchanged.
