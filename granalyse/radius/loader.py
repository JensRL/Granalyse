"""Load the granulation-radius relation sets and pick the right one per star.

A relation set is a directory holding the reference star used for the
relative-space normalisation (reference_star.json) and the MCMC draws of the
relation coefficients per model (relation_coefficients_<M>.csv). Coefficients
and reference star are meaningless apart, so they are always loaded together.

Two sets are shipped, selected per star by numax:
  ms_sgb : main-sequence and subgiant stars (Larsen et al. 2026b), reference Perky (KIC 6106415)
  rgb    : red-giant-branch stars (Larsen et al. in prep.), reference KIC 7125569
"""

import json
from importlib import resources
from pathlib import Path

import pandas as pd

SHIPPED_RELATIONS = ("ms_sgb", "rgb")

# numax [muHz] separating the shipped sets: the Kepler long-cadence Nyquist
# frequency, where the MS/SGB (numax > 260) and RGB (numax < 271) calibration
# samples meet.
RELATION_NUMAX_SPLIT = 283.2

# Outside this numax range [muHz] the relations are extrapolated: below, the
# tip of the RGB (altered granulation, excluded from the RGB calibration);
# above, the granulation timescales decouple from numax (Larsen et al. 2026a).
CALIBRATED_NUMAX_RANGE = (2.5, 3000.0)

COEFF_KEYS = ("fac", "x", "y", "z", "z2", "sigma")


def _set_dir(source):
    """Directory of a shipped set name, or a user directory as given."""
    if source in SHIPPED_RELATIONS:
        return resources.files("granalyse.radius") / "relations" / source
    path = Path(source)
    if not (path / "reference_star.json").is_file():
        raise FileNotFoundError(
            f"Relation set '{source}' is neither a shipped set {SHIPPED_RELATIONS} "
            "nor a directory containing reference_star.json.")
    return path


def select_relations(numax, split=RELATION_NUMAX_SPLIT, warn=True):
    """Shipped relation set for a star of this numax [muHz]: 'rgb' below
    `split`, 'ms_sgb' above. Warns outside the calibrated numax range."""
    lo, hi = CALIBRATED_NUMAX_RANGE
    if warn and numax < lo:
        print(f"  WARNING: numax = {numax:.3g} muHz is below {lo} muHz (tip of the RGB); "
              "the radius relations are extrapolated here.")
    elif warn and numax > hi:
        print(f"  WARNING: numax = {numax:.0f} muHz is above {hi:.0f} muHz, where the "
              "granulation timescales decouple from numax (Larsen et al. 2026a); "
              "the radius relations are extrapolated here.")
    return "rgb" if numax < split else "ms_sgb"


def load_reference_star(source="ms_sgb"):
    """Reference star of a relation set: teff, R_cm, feh, per-model amp/charb."""
    with (_set_dir(source) / "reference_star.json").open() as f:
        return json.load(f)


def load_relations(source="ms_sgb", models=("H", "J", "T")):
    """Load one relation set.

    source : a shipped set name ('ms_sgb', 'rgb'), or a directory with the
        same layout, e.g. the output of calibration/derive_relations.py.

    Returns {"name", "reference", "coeffs": {model: {fac,x,y,z,z2,sigma: ndarray}},
    "n_samples": {model: int}}. Models without a coefficient file are skipped
    with a warning.
    """
    set_dir = _set_dir(source)
    reference = load_reference_star(source)
    coeffs = {}
    n_samples = {}
    for model in models:
        path = set_dir / f"relation_coefficients_{model}.csv"
        if not path.is_file():
            print(f"  WARNING: relation set '{source}' has no coefficients for model {model}")
            continue
        if model not in reference.get("models", {}):
            raise ValueError(f"Relation set '{source}': reference_star.json has no "
                             f"amp/charb for model {model}.")
        with path.open() as f:
            df = pd.read_csv(f)
        coeffs[model] = {key: df[key].to_numpy() for key in COEFF_KEYS}
        n_samples[model] = len(df)
    if not coeffs:
        raise RuntimeError(f"Relation set '{source}' has no coefficients for models {models}.")
    return {"name": str(source), "reference": reference,
            "coeffs": coeffs, "n_samples": n_samples}
