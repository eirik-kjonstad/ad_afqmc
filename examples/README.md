# Examples

These examples show how to run AFQMC with different trial wave functions and with both

- Phaseless AFQMC
- Free projection AFQMC

The examples include mean field and coupled cluster based trial wave functions. The `staging/` subdirectory contains examples for writing trial and Hamiltonian data to disk and running from staged data, including examples for multi-GPU parallelization.

The `heat/` subdirectory contains open-shell molecular examples, starting with CH using PySCF UHF references, the Fe2-style `uhf_charge_spin_unrham` Hamiltonian decomposition, and UHF/UCISD/UCISDT/UCISDTQ trial choices.
