from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trot_fe2_real_fields_fit_hk as fe2


DEFAULT_DATA_DIR = Path("/Users/eirik/Documents/work/projects/compress-active-space")
DEFAULT_FCIDUMP = DEFAULT_DATA_DIR / "Fe2S2.COMPRESSED.FCIDUMP.1.97_18e_14o"
DEFAULT_CHKFILE = DEFAULT_DATA_DIR / "umf.chk.1.97.0.00"
TRIAL_TO_METHOD_ORDER = {
    "uhf": ("uhf", "2"),
    "ucisd": ("ccsd", "2"),
    "ucisdt": ("ccsdt", "3"),
    "ucisdtq": ("ccsdtq", "4"),
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compressed Fe2S2 18e/14o active-space model using the shared "
            "Fe2 real-field AFQMC example machinery. Additional arguments are passed "
            "through to trot_fe2_real_fields_fit_hk.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the compressed FCIDUMP and UHF checkpoint.",
    )
    parser.add_argument(
        "--fcidump",
        type=Path,
        default=None,
        help="Compressed FCIDUMP path. Defaults to Fe2S2.COMPRESSED.FCIDUMP.1.97_18e_14o.",
    )
    parser.add_argument(
        "--chkfile",
        type=Path,
        default=None,
        help="UHF checkpoint path. Defaults to umf.chk.1.97.0.00.",
    )
    parser.add_argument(
        "--trial",
        choices=tuple(TRIAL_TO_METHOD_ORDER),
        default="ucisd",
        help=(
            "Trial to stage. UHF uses the loaded determinant directly; UCISD, UCISDT, "
            "and UCISDTQ run the matching ccpy CC preparation and stage CI amplitudes."
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Staged HDF5 cache path.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    data_dir = args.data_dir.expanduser()
    fcidump = (args.fcidump or data_dir / DEFAULT_FCIDUMP.name).expanduser()
    chkfile = (args.chkfile or data_dir / DEFAULT_CHKFILE.name).expanduser()

    if not fcidump.exists():
        raise FileNotFoundError(f"Compressed FCIDUMP not found: {fcidump}")
    if not chkfile.exists():
        raise FileNotFoundError(f"UHF checkpoint not found: {chkfile}")

    method, order = TRIAL_TO_METHOD_ORDER[args.trial]
    cache = args.cache
    if cache is None:
        cache = Path(__file__).with_name(
            f"fe2_compressed_1p97_18e14o_uhf_charge_spin_{args.trial}_staged.h5"
        )

    forwarded = [
        str(fcidump),
        "--chkfile",
        str(chkfile),
        "--load-chkfile",
        "--cache",
        str(cache.expanduser()),
        "--method",
        method,
        "--order",
        order,
        "--real-field-method",
        "uhf_charge_spin",
        "--no-model-extraction",
    ]
    forwarded.extend(passthrough)

    sys.argv = [str(Path(fe2.__file__).resolve()), *forwarded]
    fe2.main()


if __name__ == "__main__":
    main()
