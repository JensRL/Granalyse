"""Nested-sampling driver (dynesty) for one FitCase."""

import time

import dynesty
import jax
import numpy as np
from dynesty import utils as dyfunc

from .likelihood import make_loglike

jax.config.update("jax_enable_x64", True)


def run_fit(fit, model=None):
    """Run dynesty nested sampling for a prepared FitCase.

    Returns (equal-weight posterior samples, weighted covariance matrix,
    log-evidence, log-evidence error).
    """
    priors = fit.priors
    ndims = len(priors)
    nlive = 4 * 50 * ndims
    if fit.nlive_enhance is not None:
        print(f"Changing nlive by a factor of {fit.nlive_enhance}")
        nlive = int(np.floor(nlive * fit.nlive_enhance))
    nliveMult = 4

    keys = list(priors.keys())

    @jax.jit
    def ptform(u):
        theta = jax.numpy.array([priors[key].ppf(u[i]) for i, key in enumerate(keys)])
        return theta

    loglike = make_loglike(fit, model=model)

    rng = np.random.default_rng(fit.seed)

    def getInitLive(ndims, nlive, nliveMult=4):
        # Sample live points from the prior, keeping only finite-likelihood ones
        u = rng.uniform(0, 1, size=(nliveMult * nlive, ndims))
        v = np.array([ptform(u[i, :]) for i in range(u.shape[0])])
        L = np.array([loglike(v[i, :]) for i in range(u.shape[0])])
        idx = np.isfinite(L)
        n_valid = idx.sum()
        if n_valid < nlive:
            raise RuntimeError(
                f"Only {n_valid}/{nlive} initial live points have finite "
                f"likelihood. Check priors and likelihood bounds."
            )
        return [u[idx, :][:nlive, :], v[idx, :][:nlive, :], L[idx][:nlive]]

    print(f"Number of live points: {nlive}")
    print(f"Number of dimensions in fit: {ndims}")
    skwargs = {
        "nlive": nlive,
        "sample": "rwalk",
        "bound": "multi",
        "live_points": getInitLive(ndims, nlive, nliveMult),
        "rstate": rng,
    }
    sampler = dynesty.NestedSampler(loglike, ptform, ndims, **skwargs)
    start_time = time.time()
    sampler.run_nested(print_progress=True, save_bounds=False, dlogz=1)
    end_time = time.time()
    elapsed = time.gmtime(round(end_time - start_time, 2))
    print(f"Nested sampling completed in {elapsed.tm_hour} h, "
          f"{elapsed.tm_min} min, {elapsed.tm_sec} s")

    result = sampler.results
    evidence = result.logz[-1]
    evidence_err = result.logzerr[-1]
    # Weighted covariance, then resample to equal weights
    unweighted_samples = result.samples
    weights = jax.numpy.exp(result.logwt - result.logz[-1])
    cov = jax.numpy.cov(unweighted_samples.T, aweights=weights, ddof=0)
    samples = dyfunc.resample_equal(unweighted_samples, np.asarray(weights))
    sampler.reset()
    del sampler
    return samples, cov, evidence, evidence_err
