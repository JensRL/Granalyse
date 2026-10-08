"""Prior construction: per-fit prior assembly (set_priors) and the
Beta-parameterisation helpers shared with the model definitions."""

from collections import OrderedDict

import numpy as np

from . import distributions as dist
from .scaling import gran_b


def beta_params_from_mode(lower_bound, upper_bound, numax, delta=0.1, k=10.0):
    """Beta (a, b, loc, scale) with mode slightly above numax inside the bounds."""
    loc = lower_bound
    end = upper_bound
    scale = end - loc
    # Clamp the target mode just inside the upper bound: for stars with numax
    # near the Nyquist frequency the (1 + delta) * numax mode can fall outside
    # [lower, upper] when the bound is capped at 0.9 * nyquist.
    target_mode = min((1 + delta) * numax, loc + 0.9 * scale)
    m = (target_mode - loc) / scale
    if not (0 < m < 1):
        raise ValueError(f"Mode fraction m={m:.4f} not in (0,1). Adjust eps/delta/eta.")
    a = 1 + k * m
    b = 1 + k * (1 - m)
    return a, b, loc, scale


def beta_mode_concentration(mode, concentration, lower, upper):
    """Beta parameters (dict with a, b, loc, scale) from a target mode and
    concentration (= alpha + beta, > 2 for a well-defined mode) on [lower, upper]."""
    m_norm = np.clip((mode - lower) / (upper - lower), 1e-6, 1 - 1e-6)
    alpha = m_norm * (concentration - 2.0) + 1.0
    beta = (1.0 - m_norm) * (concentration - 2.0) + 1.0
    alpha = max(alpha, 1e-3)
    beta = max(beta, 1e-3)
    return dict(a=alpha, b=beta, loc=lower, scale=upper - lower)


def determine_sig_prior(numax, siga):
    """(b, loc, scale) for a scaled Beta prior on the oscillation width sigma,
    with mode matching Mosser et al. 2012."""
    FWHM_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    mode_val = FWHM_to_sigma * (0.66 * numax**0.88)  # Mosser et al. 2012
    end = mode_val * 1.2
    loc = 0.8 * mode_val
    scale = end - loc
    m_scaled = (mode_val - loc) / scale
    a = siga
    b = (a - 1 - m_scaled * (a - 2)) / m_scaled
    return b, loc, scale


def set_priors(fit, model=None):
    """Assemble the full prior dictionary for one FitCase and draw prior
    predictive samples. Model-specific priors are delegated to the model
    object; the rest (activity, oscillation, noise, numax) is estimated
    from the data.

    Returns the priors OrderedDict keyed/ordered by fit.fitparams. Prior
    predictive samples are drawn separately (draw_prior_samples) so they are
    generated lazily at plot time rather than stored on every FitCase.
    """
    if model is None:
        from .models import get_model
        model = get_model(fit.model)

    priors = {}
    pris = fit.inits.copy()

    # ---------- Activity component: estimated from the low-frequency power
    low_freq = np.where(fit.freq < 0.15 * fit.numax)
    peak_power = np.max(fit.power[low_freq])
    b2_est = fit.freq[np.argmax(fit.power[low_freq])]
    a2_est = np.sqrt(2 * peak_power * b2_est)
    pris['a2'] = a2_est
    pris['b2'] = b2_est

    a2_min = 0.0
    a2_max = 3.0 * a2_est
    b2_min = 1e-6
    b2_max = (1 / 8) * gran_b(fit.numax, bscat=1)

    kappa_a2 = 6.0  # concentration; smaller => broader prior
    kappa_b2 = 6.0
    beta_a2 = beta_mode_concentration(a2_est, kappa_a2, a2_min, a2_max)
    beta_b2 = beta_mode_concentration(b2_est, kappa_b2, b2_min, b2_max)

    # ---------- Oscillation component
    if fit.pulsmod == "Gaussian":
        # Envelope width from dnu (pySYD-style), FWHM converted to sigma
        FWHM = fit.dnu * max(1, 4 * (fit.numax / 3090) ** 0.2)
        FWHM_to_sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        beta_sigma = beta_mode_concentration(FWHM_to_sigma, 8.0, 1e-4, 3 * FWHM_to_sigma)
        pris['sigma'] = FWHM_to_sigma

        # Envelope height from the median power around numax
        if fit.bin_data:
            freq_con = fit.freq
            power_con = fit.power
        else:
            freq_con = fit.freq_binned
            power_con = fit.power_binned
        numax_window = 2 * FWHM_to_sigma
        numax_region = np.where((freq_con > fit.numax - numax_window) &
                                (freq_con < fit.numax + numax_window))
        if numax_region[0].size == 0:
            # The visualization-binned grid can be coarser than the numax
            # window for low-numax stars; fall back to the raw spectrum.
            freq_con = fit.freq
            power_con = fit.power
            numax_region = np.where((freq_con > fit.numax - numax_window) &
                                    (freq_con < fit.numax + numax_window))
        median_pow = np.median(power_con[numax_region])
        if fit.numax < 1000:  # mode guess (RGB vs others)
            Posc_est = 0.4 * median_pow
        else:
            Posc_est = 0.1 * median_pow
        pris['Posc'] = Posc_est
        # Wide Beta prior allowing long tails and near-zero samples
        Posc_min = 1e-4
        Posc_max = 15 * Posc_est
        kappa = 6.0
        beta_posc = beta_mode_concentration(Posc_est, kappa, Posc_min, Posc_max)

    elif fit.pulsmod == "Peakbogging":
        pris['H'] = 0.5  # initial guess for the mixture height

    pris['numax_fit'] = fit.numax
    pris['W'] = np.median(fit.power[-100:])

    if fit.pulsmod == "Peakbogging":
        pris['beta'] = 2

    # Correlated inference: granulation params are replaced by scatter factors
    if fit.corr_inference:
        # Scatter prior on the characteristic frequencies b and d. Default:
        # Beta(6, 6) on [0.60, 1.40] around the Kallinger prediction. Above
        # numax = 3000 muHz the scaling relations are extrapolated, so the
        # lower bound is extended to 0.30 with the mode kept at 1.0; the
        # activity component stays below (its prior tops out at gran_b/8)
        # and b < d ordering is enforced in the likelihood.
        if fit.numax > 3000:
            scat_beta = beta_mode_concentration(1.0, 12.0, 0.30, 1.40)
        else:
            scat_beta = dict(a=6, b=6, loc=0.60, scale=0.8)
        if 'a' in pris and 'a' in model.corr_computed:
            pris.pop('a', None)
            pris['sigma_a'] = 0.1
            priors['sigma_a'] = dist.normal(loc=0.0, scale=0.1)  # log-normal, mean 1
        if 'b' in pris and 'b' in model.corr_computed:
            pris.pop('b', None)
            pris['sigma_b'] = 0.1
            priors['sigma_b'] = dist.beta(**scat_beta)
        if 'd' in pris and 'd' in model.corr_computed:
            pris.pop('d', None)
            pris['sigma_d'] = 0.1
            priors['sigma_d'] = dist.beta(**scat_beta)

    # ---------- Order initial estimates as fitparams, log-transform log params
    pris = OrderedDict((key, pris[key]) for key in fit.fitparams)
    for key in fit.logPars:
        pris[key] = np.log10(pris[key])

    # Model-specific priors from the model object
    priors_specific = model.specific_priors(fit)
    priors.update(priors_specific)

    # Non-model-specific priors
    if fit.pulsmod == "Gaussian":
        priors_general = {
            'Posc': dist.beta(**beta_posc),
            'sigma': dist.beta(**beta_sigma),
            'a2': dist.beta(**beta_a2),
            'b2': dist.beta(**beta_b2),
        }
    elif fit.pulsmod == "Peakbogging":
        priors_general = {
            'H': dist.beta(a=1, b=5, loc=0, scale=1),
            'a2': dist.beta(**beta_a2),
            'b2': dist.beta(**beta_b2),
            'beta': dist.beta(a=1, b=10, loc=1.0, scale=19),
        }
    else:
        raise ValueError(
            f"Unknown pulsmod '{fit.pulsmod}' (expected 'Gaussian' or 'Peakbogging')."
        )
    loc = 0.5 * pris['W']
    end = 3 * pris['W']
    numaxbot = 0.75 * fit.numax
    numaxtop = 1.25 * fit.numax
    common_generals = {'W': dist.beta(a=1.5, b=4, loc=loc, scale=end - loc),
                       'numax_fit': dist.beta(a=11, b=11, loc=numaxbot, scale=numaxtop - numaxbot)}
    priors_general.update(common_generals)
    priors.update(priors_general)

    # Combine and order according to fitparams
    all_priors = {**priors, **priors_general}
    priors = OrderedDict((key, all_priors[key]) for key in fit.fitparams)
    return priors


def draw_prior_samples(fit, size=15000):
    """Draw prior-predictive samples (linear scale) from fit.priors.

    Split out of set_priors so the samples — used only for the prior corner
    and prior-predictive PDS figures — are generated lazily at plot time
    instead of being stored in every pickled FitCase. Uses the global numpy
    RNG.
    """
    samples = np.array([fit.priors[key].rvs(size=size) for key in fit.fitparams]).T
    for key in fit.logPars:
        idx = fit.fitparams.index(key)
        samples[:, idx] = 10 ** samples[:, idx]
    return {key: samples[:, i] for i, key in enumerate(fit.fitparams)}
