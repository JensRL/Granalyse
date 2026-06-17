"""Posterior summary statistics, amplitude normalisation, and result file output."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .models import get_model
from .scaling import gran_a, gran_b, gran_d

_GRAN_FUNCS = {"a": gran_a, "b": gran_b, "d": gran_d}


def pos_uncs(samples):
    """(upper, lower) 1-sigma uncertainties from the 16th/84th percentiles."""
    sort = sorted(samples)
    idx = int(len(sort) * 0.16)
    min_unc = np.median(samples) - sort[idx]
    max_unc = sort[len(sort) - idx] - np.median(samples)
    return max_unc, min_unc


def save_fit_params(fit, model=None):
    """Median and 1-sigma uncertainties for all fitted parameters; in
    correlated mode the physical granulation parameters are reconstructed
    from the sampled numax and scatter factors."""
    if model is None:
        model = get_model(fit.model)

    res_dict = {}
    fitdata = {k: jnp.asarray(v) for k, v in fit.fitdata.items()}

    for param in fit.fitparams:
        samples = fitdata[param]
        res_dict[param] = [
            jnp.median(samples).item(),
            *pos_uncs(np.asarray(samples)),
        ]

    if fit.corr_inference:
        numax = fitdata["numax_fit"]
        for param in ("a", "b", "d"):
            if param in model.corr_computed:
                scat = fitdata[f"sigma_{param}"]
                samples = _GRAN_FUNCS[param](numax, scat)
                res_dict[param] = [np.median(samples), *pos_uncs(np.asarray(samples))]

    return pd.DataFrame(res_dict)


def compute_normalised_amplitudes(fit, model=None):
    """Add normalised amplitudes (a_norm, c_norm, e_norm as applicable) and
    the total area under the granulation PDS to fit.fitresults."""
    if model is None:
        model = get_model(fit.model)

    norm = model.norm_factors(fit)
    res = fit.fitresults

    area = 0.0
    area_unc_sq = 0.0
    for p in model.amp_components:
        vals = res[p]
        p_norm = vals[0] / np.sqrt(norm[p])
        fit.fitresults[f"{p}_norm"] = [p_norm,
                                       vals[1] / np.sqrt(norm[p]),
                                       vals[2] / np.sqrt(norm[p])]
        area += p_norm**2
        area_unc_sq += (vals[1] / np.sqrt(norm[p])) ** 2

    area_unc = np.sqrt(area_unc_sq)
    fit.fitresults["area"] = [area, area_unc, area_unc]
    return fit


def save_run_output(fit, samples, cov, evidence, evidence_err, model=None):
    """Write FitData/Evidence/Covariances files, compute and write FitResults.

    Reproduces the output formats of the original framework:
      FitData_model_X.csv     equal-weight posterior samples (linear scale)
      Evidence_model_X.txt    log-evidence and error
      Covariances.csv         weighted posterior covariance matrix
      FitResults_model_X.csv  medians, uncertainties, normalised amplitudes
    """
    if model is None:
        model = get_model(fit.model)

    os.makedirs(fit.respath, exist_ok=True)

    fit_data = pd.DataFrame(samples)
    fit_data.columns = fit.fitparams
    for param in fit.logPars:  # back to linear space
        fit_data[param] = 10 ** fit_data[param]
    # gzip-compressed (inferred from the .csv.gz extension) and written at 6
    # significant figures: ~5x smaller than full-precision CSV, with a relative
    # effect on posterior medians/percentiles of ~1e-6, far below sampling noise.
    fit_data.to_csv(fit.fitdatafile, index=False, header=True, float_format="%.6g")
    fit.fitdata = fit_data

    with open(os.path.join(fit.respath, f"Evidence_model_{model.name}.txt"), "w") as f:
        f.write(f"{evidence:.3f} {evidence_err:.3f}")

    cov_df = pd.DataFrame(np.asarray(cov), index=fit.fitparams, columns=fit.fitparams)
    cov_df.to_csv(os.path.join(fit.respath, "Covariances.csv"))

    fit.fitresults = save_fit_params(fit, model=model)
    fit = compute_normalised_amplitudes(fit, model=model)
    fit.fitresults.to_csv(fit.fitresultsfile, index=False, header=True)
    return fit
