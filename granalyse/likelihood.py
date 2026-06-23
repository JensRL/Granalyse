"""Likelihood construction for the nested-sampling fits.

make_loglike composes the log-likelihood from the model object and the fit
flags (Gaussian envelope vs Peakbogging, correlated vs free granulation parameters).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .models import get_model
from .scaling import gran_a, gran_b, gran_d

LOG10 = jnp.log(10)


def unpackParams(logPars, keys, theta):
    """Cast a sample vector theta as a dict, inverting log10 where needed."""
    thetaU = {key: theta[i] if key not in logPars else jnp.exp(theta[i] * LOG10)
              for i, key in enumerate(keys)}
    return thetaU


@jax.jit
def chi2_dist(P, dof, mod, term1):
    """Chi-squared log-likelihood for binned PDS data (Handberg & Campante 2014)."""
    term2 = (dof - 1) * jnp.log(P) - dof * jnp.log(mod)
    term3 = -dof * P / mod
    return jnp.sum(term1 + term2 + term3)


def peakbogging_dist(P, dof, mod_back, beta, term1):
    """Per-bin log-likelihoods of the background and the scaled (peak) background."""
    term2 = (dof - 1) * jnp.log(P) - dof * jnp.log(mod_back)
    term3 = -dof * P / mod_back
    background = term1 + term2 + term3

    term2peak = (dof - 1) * jnp.log(P) - dof * jnp.log(mod_back * beta)
    term3peak = -dof * P / (mod_back * beta)
    peakbog = term1 + term2peak + term3peak

    return background, peakbog


peakbogging_dist = jax.jit(peakbogging_dist, static_argnums=(1,))


def oscpeak(nu, numax, H, sigma):
    """Gaussian mixture-coefficient peak for peakbogging (sigma fixed per fit)."""
    return H * jnp.exp(-0.5 * ((nu - numax) / sigma) ** 2)


# ------------------- mixture-likelihood helpers (legacy/simulations) -------------------
def sigmoid_func(x, numax, a=1, b=0, width_factor=0.5, slope=0.8):
    """Sigmoid with plateau before numax; downweights high frequencies."""
    c = 2 * numax
    width = width_factor * numax
    y = b + (a - b) / (1 + np.exp(slope * (x - c) / width))
    return y


@jax.jit
def compute_static_chi2(P, bagmean, sigmoid):
    """Static chi2 term for the mixed-model likelihood."""
    bag = jax.scipy.special.logsumexp(-P / bagmean) - jnp.log(bagmean)
    chi2_sigmoid = jnp.log1p(-sigmoid) + bag
    return chi2_sigmoid


@jax.jit
def mixed_mod_likelihood(P, dof, mod, sigmoid, chi2_sigmoid):
    """Chi2 likelihood mixed with a sigmoid-weighted static term."""
    term1 = (dof - 1) * jnp.log(dof) - jax.scipy.special.gammaln(dof - 1)
    term2 = (dof - 1) * jnp.log(P) - dof * jnp.log(mod)
    term3 = -dof * P / mod
    chi2_standard = jnp.log(sigmoid) + jnp.sum((term1 + term2 + term3))
    chi2_standard_logsum = jax.scipy.special.logsumexp(chi2_standard)
    chi2_sigmoid_logsum = jax.scipy.special.logsumexp(chi2_sigmoid)
    return chi2_standard_logsum + chi2_sigmoid_logsum


# ------------------------------- loglike factory -------------------------------
def make_loglike(fit, model=None):
    """
    Build the jitted log-likelihood for one FitCase.
    """
    if model is None:
        model = get_model(fit.model)

    nu = jnp.array(fit.freq)
    power = jnp.array(fit.power)
    eta = jnp.array(fit.eta)
    dof = int(fit.dof)
    term1 = (dof - 1) * jnp.log(dof) - jax.scipy.special.gammaln(dof - 1)
    logPars = fit.logPars
    keys = list(fit.priors.keys())

    corr_computed = set(model.corr_computed) if fit.corr_inference else set()
    constraint = model.ordering_constraint

    def phys_params(thetaU):
        # In correlated mode the granulation amplitudes/timescales follow the
        # Kallinger 2014 relations at the sampled numax, times a scatter factor.
        out = dict(thetaU)
        if "a" in corr_computed:
            out["a"] = gran_a(thetaU["numax_fit"], thetaU["sigma_a"])
        if "b" in corr_computed:
            out["b"] = gran_b(thetaU["numax_fit"], thetaU["sigma_b"])
        if "d" in corr_computed:
            out["d"] = gran_d(thetaU["numax_fit"], thetaU["sigma_d"])
        return out

    if fit.pulsmod == "Peakbogging":
        func_back = model.func_back
        sigma = fit.sigwidth

        @jax.jit
        def loglike(theta):
            thetaU = unpackParams(logPars, keys, theta)
            thetaP = phys_params(thetaU)
            mod_back = func_back(nu=nu, eta=eta, **thetaP)
            mod_peak = oscpeak(nu, thetaU["numax_fit"], thetaU["H"], sigma)
            Lback, Lpeak = peakbogging_dist(power, dof, mod_back, thetaU["beta"], term1)
            term_back = jnp.log1p(-mod_peak) + Lback
            term_peak = jnp.log(mod_peak) + Lpeak
            log_mix = jnp.logaddexp(term_back, term_peak)
            L = jnp.sum(log_mix)
            L += constraint(**thetaP)
            return L

    elif fit.pulsmod == "Gaussian":
        func_real = model.func_real

        @jax.jit
        def loglike(theta):
            thetaU = unpackParams(logPars, keys, theta)
            thetaP = phys_params(thetaU)
            mod = func_real(nu=nu, eta=eta, **thetaP)
            L = chi2_dist(power, dof, mod, term1)
            L += constraint(**thetaP)
            return L

    else:
        raise ValueError(f"Unknown pulsation model '{fit.pulsmod}' "
                         "(expected 'Gaussian' or 'Peakbogging').")

    return loglike
