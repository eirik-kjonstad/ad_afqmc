# HEAT Examples

Open-shell molecular examples using PySCF UHF references.

## CH

The CH example uses the geometry

```text
C   0.000000   0.000000   0.000000
H   1.116670   0.000000   0.000000
```

with `unit="Angstrom"`, `basis="6-31g"`, and `spin=1`.

Run the available trial choices with:

```bash
python examples/heat/ch.py --trial uhf
python examples/heat/ch.py --trial ucisd
python examples/heat/ch.py --trial ucisdt
python examples/heat/ch.py --trial ucisdtq
```

The CI trial options require `ccpy`. Staged inputs are cached next to the script as
`ch_<trial>_staged.h5`; pass `--overwrite-cache` to rebuild them.
