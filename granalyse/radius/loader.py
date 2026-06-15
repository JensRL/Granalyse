"""Load the shipped granulation-radius relation coefficients and the Perky
(KIC 6106415) reference parameters used for relative-space normalisation."""

import json
from importlib import resources

import numpy as np
import pandas as pd

DEFAULT_SEED = 42


def _relations_dir():
    return resources.files("granalyse.radius") / "relations"


def load_perky_params():
    """Perky reference parameters: teff, R_cm, feh, per-model amp/charb."""
    with (_relations_dir() / "perky_params.json").open() as f:
        return json.load(f)


def load_relation_coefficients(models=("H", "J", "T"), seed=DEFAULT_SEED):
    """MCMC draws of the scaling-relation coefficients per model.

    Returns (coeffs, n_samples): {model: {fac,x,y,z,z2,sigma: ndarray}},
    {model: int}. Models without a shipped coefficient file are skipped
    with a warning.
    """
    coeffs = {}
    n_samples = {}
    for model in models:
        path = _relations_dir() / f"relation_coefficients_{model}_seed{seed}.csv"
        try:
            with path.open() as f:
                df = pd.read_csv(f)
        except FileNotFoundError:
            print(f"  WARNING: no shipped relation coefficients for model {model}")
            continue
        coeffs[model] = {key: df[key].to_numpy() for key in
                         ("fac", "x", "y", "z", "z2", "sigma")}
        n_samples[model] = len(df)
    if not coeffs:
        raise RuntimeError(f"No relation coefficient files found for models {models}.")
    return coeffs, n_samples
