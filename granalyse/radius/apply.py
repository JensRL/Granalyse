"""Radius inference from granulation posteriors via the derived
granulation-radius scaling relations, with Bayesian Model Averaging.

The relation is applied in space relative to the reference star Perky
(scaling relations from arXiv:2605.26742):

  log10(R_rel) = fac + y*log10(charb_rel) + x*log10(A_gran_rel)
               + z*log10(Teff_rel) + z2*(feh - feh_Perky) + sigma*eps

Random draws use the global numpy RNG; seed with np.random.seed() for
reproducibility.
"""

import numpy as np

from ..models import get_model
from .loader import load_perky_params, load_relation_coefficients

R_SUN = 6.957e10  # cm

# Bolometric-correction parameters per mission: C_bol = (Teff/ref)**exp
BOLOMETRIC = {
    "Kepler": (5934.0, 0.80),
    "TESS": (4714.0, 0.81),
}


def parse_err(err_spec, default=None):
    """Normalise an error specification to (sigma_lo, sigma_hi).
    Accepts a float (symmetric), a (lo, hi) pair, or None -> default."""
    if err_spec is None:
        v = float(default) if default is not None else 0.0
        return (v, v)
    if isinstance(err_spec, (list, tuple)):
        if len(err_spec) != 2:
            raise ValueError(f"Asymmetric error must be (lo, hi); got {err_spec!r}")
        return (float(err_spec[0]), float(err_spec[1]))
    return (float(err_spec), float(err_spec))


def split_normal_samples(mu, sigma_lo, sigma_hi, n):
    """n draws from a split-normal centred on mu (plain normal if lo == hi)."""
    u = np.random.randn(n)
    return mu + np.where(u >= 0, sigma_hi * u, sigma_lo * u)


def transform_posterior_to_observables(posterior_df, teff, model,
                                       mission="Kepler", corr_inference=None):
    """(charb [muHz], A_gran [ppm]) per posterior sample, including the
    mission-specific bolometric correction. The granulation-amplitude
    composition is delegated to the model object."""
    model = get_model(model)
    if model.transform_observables is None:
        raise ValueError(f"Model {model.name} has no granulation-radius relation.")
    bol_ref, bol_exp = BOLOMETRIC[mission]
    C_bol = (float(teff) / bol_ref) ** bol_exp
    charb, agran = model.transform_observables(posterior_df, corr_inference)
    return charb, C_bol * agran


def predict_radius_direct(charbs_rel, amps_rel, teffs_rel, fehs_rel,
                          coeffs, perky_ref):
    """Apply the scaling relation in relative space; returns radii in R_sun."""
    lg_R_rel = (coeffs["fac"]
                + coeffs["y"] * np.log10(charbs_rel)
                + coeffs["x"] * np.log10(amps_rel)
                + coeffs["z"] * np.log10(teffs_rel)
                + coeffs["z2"] * fehs_rel
                + coeffs["sigma"] * np.random.randn(len(coeffs["sigma"])))
    R_cm = (10 ** lg_R_rel) * perky_ref["R_cm"]
    return R_cm / R_SUN


def compute_model_weights(evidences):
    """BMA weights from log-evidences: {model: [log_ev, log_ev_err]} -> {model: w}."""
    log_evs = {m: ev[0] for m, ev in evidences.items()}
    max_ev = max(log_evs.values())
    unnorm = {m: np.exp(lev - max_ev) for m, lev in log_evs.items()}
    total = sum(unnorm.values())
    return {m: w / total for m, w in unnorm.items()}


def combine_radius_posteriors(radius_samples, weights):
    """Concatenate per-model radius samples in proportion to their BMA weights."""
    first_model = next(iter(radius_samples))
    total_n = len(radius_samples[first_model])
    n_draw = {m: int(w * total_n) for m, w in weights.items()}
    remainder = total_n - sum(n_draw.values())
    if remainder > 0:
        n_draw[max(weights, key=weights.get)] += remainder

    pieces = []
    for m in ["H", "J", "T"]:
        if m in radius_samples and n_draw.get(m, 0) > 0:
            idx = np.random.choice(len(radius_samples[m]),
                                   size=n_draw[m], replace=True)
            pieces.append(radius_samples[m][idx])

    combined = np.concatenate(pieces)
    np.random.shuffle(combined)
    return combined


def infer_radius(star_name, posteriors, evidences, teff, teff_err,
                 feh=None, feh_err=None, mission="Kepler",
                 scaling_coeffs=None, n_samples=None, perky_ref=None,
                 feh_prior_mean=-0.10, feh_prior_std=0.40):
    """BMA radius inference for one star.

    Parameters
    ----------
    posteriors : dict {model letter: posterior DataFrame}
    evidences  : dict {model letter: (log_ev, log_ev_err)}
    teff, teff_err : float / float-or-(lo, hi)
    feh, feh_err   : optional [Fe/H] and uncertainty; without feh a
        population prior N(feh_prior_mean, feh_prior_std) is used.
    mission : 'Kepler' or 'TESS' (bolometric correction)
    scaling_coeffs, n_samples : as from load_relation_coefficients();
        loaded from the shipped seed-42 files when None.
    perky_ref : as from load_perky_params(); loaded when None.

    Returns a dict with combined and per-model radius summaries and samples.
    """
    if scaling_coeffs is None or n_samples is None:
        scaling_coeffs, n_samples = load_relation_coefficients()
    if perky_ref is None:
        perky_ref = load_perky_params()

    has_feh = (feh is not None) and not np.isnan(float(feh))
    teff_lo, teff_hi = parse_err(teff_err)
    if has_feh:
        feh_lo, feh_hi = parse_err(feh_err, default=0.2)

    radius_samples = {}
    evidences_used = {}
    model_results = {}

    for model in ["H", "J", "T"]:
        if model not in scaling_coeffs or model not in posteriors:
            continue
        post_df = posteriors[model]
        if post_df is None or len(post_df) == 0:
            print(f"  Model {model}: no posterior samples - skipping")
            continue
        ev = evidences.get(model)
        if ev is None:
            print(f"  Model {model}: no evidence - skipping")
            continue
        log_ev = ev[0]
        n_sr = n_samples[model]

        charb_post, agran_post = transform_posterior_to_observables(
            post_df, teff, model, mission=mission)

        n_post = len(charb_post)
        idx = np.random.choice(n_post, size=n_sr, replace=(n_post < n_sr))
        charbs = charb_post[idx]
        amps = agran_post[idx]

        # Split-normal Teff draws in log space
        lg_teff_lo = teff_lo / teff / np.log(10)
        lg_teff_hi = teff_hi / teff / np.log(10)
        teffs = 10 ** split_normal_samples(np.log10(teff), lg_teff_lo, lg_teff_hi, n_sr)

        if has_feh:
            fehs = split_normal_samples(float(feh), feh_lo, feh_hi, n_sr)
        else:
            fehs = np.clip(feh_prior_mean + feh_prior_std * np.random.randn(n_sr),
                           -4.0, 1.0)

        charbs_rel = charbs / perky_ref["models"][model]["charb"]
        amps_rel = amps / perky_ref["models"][model]["amp"]
        teffs_rel = teffs / perky_ref["teff"]
        fehs_rel = fehs - perky_ref["feh"]

        Rs = predict_radius_direct(charbs_rel, amps_rel, teffs_rel, fehs_rel,
                                   scaling_coeffs[model], perky_ref)

        radius_samples[model] = Rs
        evidences_used[model] = [log_ev, 0.0]
        R_med = float(np.median(Rs))
        R_16 = float(np.percentile(Rs, 16))
        R_84 = float(np.percentile(Rs, 84))
        model_results[model] = {
            "R_median": R_med,
            "R_16": R_16,
            "R_84": R_84,
            "R_std": float(np.std(Rs)),
            "samples": Rs,
            "log_evidence": log_ev,
        }
        print(f"  Model {model}: {R_med:.4f} "
              f"+{R_84 - R_med:.4f} -{R_med - R_16:.4f} Rsun")

    if not radius_samples:
        raise RuntimeError(f"No radius posteriors could be computed for {star_name}.")

    weights = compute_model_weights(evidences_used)
    combined = combine_radius_posteriors(radius_samples, weights)
    R_med = float(np.median(combined))
    R_16 = float(np.percentile(combined, 16))
    R_84 = float(np.percentile(combined, 84))

    print(f"\n  COMBINED: {R_med:.4f} +{R_84 - R_med:.4f} -{R_med - R_16:.4f} Rsun")
    print("  Weights: " + "  ".join(f"{m}={w:.3f}" for m, w in weights.items()))

    return {
        "star_name": star_name,
        "R_combined": R_med,
        "R_16_combined": R_16,
        "R_84_combined": R_84,
        "R_std_combined": float(np.std(combined)),
        "weights": weights,
        "model_results": model_results,
        "combined_samples": combined,
        "teff": teff,
        "teff_err": teff_err,
        "feh": float(feh) if has_feh else np.nan,
        "has_feh": has_feh,
    }
