"""The built-in background models D, J, H, T.

Model functions are the original NSGF jax implementations. Each model bundles
its parameters, scaling-law coefficients, initial guesses, model-specific
priors, ordering constraint, and amplitude normalisation.

Model forms (nu in muHz, amplitudes in ppm):
  D: single Harvey       (a^2/b) / (1 + (nu/b)^l)
  J: Kallinger hybrid    a^2 / (1 + (nu/b)^l + (nu/d)^k)
  H: two Harvey terms    (a^2/b)/(1+(nu/b)^l) + (c^2/d)/(1+(nu/d)^k)
  T: three Harvey terms  H + (e^2/f)/(1+(nu/f)^m)
All share white noise W, an activity Lorentzian (a2, b2), and (in the *_real
variants) a Gaussian oscillation envelope (Posc, numax_fit, sigma).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate

from .. import distributions as dist
from ..priors import beta_params_from_mode
from ..scaling import gran_a, gran_b, gran_d
from .base import BackgroundModel, xi_analytic, xi_clipped

# ============================ Model functions ============================
apo_act = (2 / jnp.pi) ** 2


@jax.jit
def modelJ_real(nu, eta, numax_fit, a, b, d, l, k, W, Posc, sigma, a2, b2, **kwargs):
    mod = a**2 / (1 + (nu / b) ** l + (nu / d) ** k)
    noise = W
    osc = Posc * jnp.exp(-0.5 * ((nu - numax_fit) / sigma) ** 2)
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + osc + activity) + noise


@jax.jit
def modelJ_back(nu, eta, a, b, d, l, k, W, a2, b2, **kwargs):
    mod = a**2 / (1 + (nu / b) ** l + (nu / d) ** k)
    noise = W
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + activity) + noise


@jax.jit
def modelH_real(nu, eta, numax_fit, a, c, b, d, l, k, W, Posc, sigma, a2, b2, **kwargs):
    p1 = (a**2 / b) / (1 + (nu / b) ** l)
    p2 = (c**2 / d) / (1 + (nu / d) ** k)
    mod = p1 + p2
    noise = W
    osc = Posc * jnp.exp(-0.5 * ((nu - numax_fit) / sigma) ** 2)
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + osc + activity) + noise


@jax.jit
def modelH_back(nu, eta, a, c, b, d, l, k, W, a2, b2, **kwargs):
    p1 = (a**2 / b) / (1 + (nu / b) ** l)
    p2 = (c**2 / d) / (1 + (nu / d) ** k)
    mod = p1 + p2
    noise = W
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + activity) + noise


@jax.jit
def modelT_real(nu, eta, numax_fit, a, c, b, d, e, f, l, k, m, W, Posc, sigma, a2, b2, **kwargs):
    p1 = (a**2 / b) / (1 + (nu / b) ** l)
    p2 = (c**2 / d) / (1 + (nu / d) ** k)
    p3 = (e**2 / f) / (1 + (nu / f) ** m)
    mod = p1 + p2 + p3
    noise = W
    osc = Posc * jnp.exp(-0.5 * ((nu - numax_fit) / sigma) ** 2)
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + osc + activity) + noise


@jax.jit
def modelT_back(nu, eta, a, c, b, d, e, f, l, k, m, W, a2, b2, **kwargs):
    p1 = (a**2 / b) / (1 + (nu / b) ** l)
    p2 = (c**2 / d) / (1 + (nu / d) ** k)
    p3 = (e**2 / f) / (1 + (nu / f) ** m)
    mod = p1 + p2 + p3
    noise = W
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + activity) + noise


@jax.jit
def modelD_real(nu, eta, numax_fit, a, b, l, W, Posc, sigma, a2, b2, **kwargs):
    mod = (a**2 / b) / (1 + (nu / b) ** l)
    noise = W
    osc = Posc * jnp.exp(-0.5 * ((nu - numax_fit) / sigma) ** 2)
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + osc + activity) + noise


@jax.jit
def modelD_back(nu, eta, a, b, l, W, a2, b2, **kwargs):
    mod = (a**2 / b) / (1 + (nu / b) ** l)
    noise = W
    activity = apo_act * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + activity) + noise


# ===================== Kallinger 2014 coefficient sets ====================
# 10**(k + e*log10(numax)) gives the scaling-law center on the linear scale.
K14 = {
    "a": (3.53, -0.609),
    "b": (-0.499, 0.970),
    "c": (3.477, -0.609),
    "d": (-0.02, 0.992),
}
K14_J = {**K14, "a": (3.555, -1.006)}  # J's single amplitude scales differently


def _scaling_log10(coeffs, param, numax):
    k, e = coeffs[param]
    return k + np.log10(numax) * e


# ============================ Initial guesses ============================
def _inits_J(numax):
    a = _scaling_log10(K14_J, "a", numax)
    b = _scaling_log10(K14_J, "b", numax)
    d = _scaling_log10(K14_J, "d", numax)
    return {"a": float(10**a), "b": float(10**b), "d": float(10**d),
            "l": float(5 / 3), "k": float(17 / 3)}


def _inits_H(numax):
    vals = {p: _scaling_log10(K14, p, numax) for p in ["a", "b", "c", "d"]}
    return {**{p: float(10**v) for p, v in vals.items()},
            "l": 4.0, "k": 4.0}


def _inits_T(numax):
    # a-d coefficients are the model-H ones (no T-specific calibration exists)
    vals = {p: _scaling_log10(K14, p, numax) for p in ["a", "b", "c", "d"]}
    e = 0.5 * 10 ** vals["c"]  # half the amplitude of the second component
    f = 1.8 * 10 ** vals["d"]  # beyond the frequency of the second component
    return {**{p: float(10**v) for p, v in vals.items()},
            "e": float(e), "f": float(f), "l": 4.0, "k": 4.0, "m": 6.0}


def _inits_D(numax):
    a = _scaling_log10(K14, "a", numax)
    b = _scaling_log10(K14, "b", numax)
    a *= 1.2  # inflate single-component amplitude (applied on the log10 scale)
    return {"a": float(10**a), "b": float(10**b), "l": 2.0}


# ========================= Model-specific priors =========================
# Shared hyperparameters: log10-normal widths and slope Beta setups
S_AMP = 0.1
BETA_L = dict(a=1.8, b=3.0, loc=1, scale=8)
BETA_K = dict(a=2.0, b=5.0, loc=1, scale=9)
BETA_M = dict(a=2.0, b=5.0, loc=1, scale=10)


def _lognormal_prior(coeffs, param, numax):
    return dist.normal(loc=_scaling_log10(coeffs, param, numax), scale=S_AMP)


def _priors_J(fit):
    if not fit.corr_inference:
        return {
            "a": _lognormal_prior(K14_J, "a", fit.numax),
            "b": _lognormal_prior(K14_J, "b", fit.numax),
            "d": _lognormal_prior(K14_J, "d", fit.numax),
            "l": dist.beta(**BETA_L),
            "k": dist.beta(**BETA_K),
        }
    return {
        "a": _lognormal_prior(K14_J, "a", fit.numax),
        "l": dist.beta(**BETA_L),
        "k": dist.beta(**BETA_K),
    }


def _priors_H(fit):
    if not fit.corr_inference:
        return {
            "a": _lognormal_prior(K14, "a", fit.numax),
            "b": _lognormal_prior(K14, "b", fit.numax),
            "c": _lognormal_prior(K14, "c", fit.numax),
            "d": _lognormal_prior(K14, "d", fit.numax),
            "l": dist.beta(**BETA_L),
            "k": dist.beta(**BETA_K),
        }
    return {
        "c": _lognormal_prior(K14, "c", fit.numax),
        "l": dist.beta(**BETA_L),
        "k": dist.beta(**BETA_K),
    }


def _priors_T(fit):
    lower = 0.9 * fit.numax
    upper = min(4 * fit.numax, 0.9 * fit.nyquist)
    af, bf, locf, scalef = beta_params_from_mode(lower, upper, fit.numax)
    mu_c = 10 ** _scaling_log10(K14, "c", fit.numax)
    extras = {
        "e": dist.normal(loc=np.log10(0.5 * mu_c), scale=0.10),
        "f": dist.beta(a=af, b=bf, loc=locf, scale=scalef),
        "m": dist.beta(**BETA_M),
    }
    if not fit.corr_inference:
        return {
            "a": _lognormal_prior(K14, "a", fit.numax),
            "b": _lognormal_prior(K14, "b", fit.numax),
            "c": _lognormal_prior(K14, "c", fit.numax),
            "d": _lognormal_prior(K14, "d", fit.numax),
            "l": dist.beta(**BETA_L),
            "k": dist.beta(**BETA_K),
            **extras,
        }
    return {
        "c": _lognormal_prior(K14, "c", fit.numax),
        "l": dist.beta(**BETA_L),
        "k": dist.beta(**BETA_K),
        **extras,
    }


def _priors_D(fit):
    # Not covered by the original build_common_priors (model D would crash
    # there); defined here consistently with the H-style amplitude priors.
    if not fit.corr_inference:
        return {
            "a": _lognormal_prior(K14, "a", fit.numax),
            "b": _lognormal_prior(K14, "b", fit.numax),
            "l": dist.beta(**BETA_L),
        }
    return {"l": dist.beta(**BETA_L)}


# ========================= Ordering constraints ==========================
# Reject samples where the Harvey components swap order: amplitude levels
# L1 > L2 (> L3), and characteristic frequencies b < d. The frequency
# ordering never triggers with the default scatter priors (their ranges are
# disjoint) but protects the extended priors used above numax = 3000 muHz.
@jax.jit
def _no_constraint(**kwargs):
    return 0.0


@jax.jit
def _ordering_J(b=None, d=None, **kwargs):
    return jnp.where(d > b, 0.0, -jnp.inf)


@jax.jit
def _ordering_H(a=None, b=None, c=None, d=None, **kwargs):
    L1 = a**2 / b
    L2 = c**2 / d
    ok = jnp.logical_and(L1 > L2, d > b)
    return jnp.where(ok, 0.0, -jnp.inf)


@jax.jit
def _ordering_T(a=None, b=None, c=None, d=None, e=None, f=None, **kwargs):
    L1 = a**2 / b
    L2 = c**2 / d
    L3 = e**2 / f
    ok = jnp.logical_and(jnp.logical_and(L1 > L2, L2 > L3), d > b)
    return jnp.where(ok, 0.0, -jnp.inf)


# ====================== Amplitude normalisation ==========================
# xi factors such that a_norm = a / sqrt(xi); analytic per Karoff et al. 2013
# for Harvey terms, numerical integration for J's joint denominator.
def _norm_H(fit):
    res = fit.fitresults
    return {"a": xi_analytic(res["l"][0]), "c": xi_analytic(res["k"][0])}


def _norm_T(fit):
    res = fit.fitresults
    return {"a": xi_analytic(res["l"][0]), "c": xi_analytic(res["k"][0]),
            "e": xi_analytic(res["m"][0])}


def _norm_D(fit):
    return {"a": xi_analytic(fit.fitresults["l"][0])}


def _norm_J(fit):
    res = fit.fitresults

    def integrand(nu, charfreq1, charfreq2, exponent1, exponent2):
        return 1 / (1 + (nu / charfreq1) ** exponent1 + (nu / charfreq2) ** exponent2)

    if fit.corr_inference:
        b = gran_b(res["numax_fit"][0], res["sigma_b"][0])
        d = gran_d(res["numax_fit"][0], res["sigma_d"][0])
    else:
        b = res["b"][0]
        d = res["d"][0]
    integral = integrate.quad(integrand, 0, np.inf, args=(b, d, res["l"][0], res["k"][0]))
    return {"a": 1 / integral[0]}


# ================== Granulation observables for the radius relations ==================
# Per-sample (charb, A_gran) from a posterior DataFrame; A_gran is the
# normalised total granulation amplitude before the bolometric correction.
# Ports of GranRadRels transform_posterior_to_observables, per model.
def norm_factors_J_vectorised(b, d, l, k, n_nodes=200, t_range=20.0):
    """Vectorised model-J normalisation via Gauss-Legendre quadrature with the
    substitution nu = exp(T*x), x in [-1, 1] (T=20 covers (0, inf) in practice)."""
    from numpy.polynomial.legendre import leggauss
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    l = np.asarray(l, dtype=float)
    k = np.asarray(k, dtype=float)

    nodes, weights = leggauss(n_nodes)
    nu = np.exp(t_range * nodes)
    jacobian = t_range * nu

    nu2 = nu[:, None]
    jac = jacobian[:, None]
    w = weights[:, None]
    intgrd = 1.0 / (1.0 + (nu2 / b[None, :]) ** l[None, :] + (nu2 / d[None, :]) ** k[None, :])
    integrals = (w * jac * intgrd).sum(axis=0)
    integrals = np.maximum(integrals, 1e-300)
    return 1.0 / integrals


def _detect_corr(posterior_df, corr_inference):
    if corr_inference is None:
        return "sigma_b" in posterior_df.columns
    return corr_inference


def _obs_H(df, corr_inference=None):
    l = df["l"].values
    k = df["k"].values
    if _detect_corr(df, corr_inference):
        numax = df["numax_fit"].values
        charb = gran_b(numax, df["sigma_b"].values)
        a = gran_a(numax, df["sigma_a"].values)
        c = df["c"].values
    else:
        charb = df["b"].values.copy()
        a = df["a"].values
        c = df["c"].values
    a_norm = a / np.sqrt(xi_clipped(l))
    c_norm = c / np.sqrt(xi_clipped(k))
    return charb, np.sqrt(a_norm**2 + c_norm**2)


def _obs_J(df, corr_inference=None):
    l = df["l"].values
    k = df["k"].values
    a = df["a"].values
    if _detect_corr(df, corr_inference):
        numax = df["numax_fit"].values
        b = gran_b(numax, df["sigma_b"].values)
        d = gran_d(numax, df["sigma_d"].values)
        charb = b
    else:
        b = df["b"].values
        d = df["d"].values
        charb = b.copy()
    norm = norm_factors_J_vectorised(b, d, l, k)
    return charb, a / np.sqrt(norm)


def _obs_T(df, corr_inference=None):
    l = df["l"].values
    k = df["k"].values
    m = df["m"].values
    c = df["c"].values
    e = df["e"].values
    if _detect_corr(df, corr_inference):
        numax = df["numax_fit"].values
        charb = gran_b(numax, df["sigma_b"].values)
        a = gran_a(numax, df["sigma_a"].values)
    else:
        charb = df["b"].values.copy()
        a = df["a"].values
    a_norm = a / np.sqrt(xi_clipped(l))
    c_norm = c / np.sqrt(xi_clipped(k))
    e_norm = e / np.sqrt(xi_clipped(m))
    return charb, np.sqrt(a_norm**2 + c_norm**2 + e_norm**2)


# ============================ Plotting overrides ============================
def _plot_components_J(ax, freq, eta, params, power_max):
    # J's joint denominator has no meaningful per-component profiles;
    # mark the two characteristic frequencies instead.
    for p in ("b", "d"):
        val = float(params[p])
        ax.vlines(val, 0, 0.8 * power_max, colors="g", linestyles="dashed",
                  linewidth=1, label=f"{p} = {val:.2f} " + r"$\mu$Hz")
    return []


# ============================ Model instances ============================
MODEL_D = BackgroundModel(
    name="D",
    aliases=("harvey1",),
    description="Single Harvey-like granulation component",
    params=("a", "b", "l"),
    log_params=frozenset({"a", "b"}),
    labels={"a": r"$a$ [ppm]", "b": r"$b$ [$\mu$Hz]", "l": r"$l$"},
    func_real=modelD_real,
    func_back=modelD_back,
    kallinger_coeffs=K14,
    inits=_inits_D,
    specific_priors=_priors_D,
    ordering_constraint=_no_constraint,
    norm_factors=_norm_D,
    amp_components=("a",),
    corr_computed=("a", "b"),
    has_radius_relation=False,
)

MODEL_J = BackgroundModel(
    name="J",
    aliases=("hybrid",),
    description="Single amplitude with two characteristic frequencies - Lundkvist et al. 2021 and Larsen et al. 2025",
    params=("a", "b", "d", "l", "k"),
    log_params=frozenset({"a", "b", "d"}),
    labels={"a": r"$a$ [ppm]", "b": r"$b$ [$\mu$Hz]", "d": r"$d$ [$\mu$Hz]",
            "l": r"$l$", "k": r"$k$"},
    func_real=modelJ_real,
    func_back=modelJ_back,
    kallinger_coeffs=K14_J,
    inits=_inits_J,
    specific_priors=_priors_J,
    ordering_constraint=_ordering_J,
    norm_factors=_norm_J,
    amp_components=("a",),
    corr_computed=("b", "d"),
    transform_observables=_obs_J,
    plot_components=_plot_components_J,
)

MODEL_H = BackgroundModel(
    name="H",
    aliases=("harvey2",),
    description="Two Harvey-like granulation components",
    params=("a", "b", "c", "d", "l", "k"),
    log_params=frozenset({"a", "b", "c", "d"}),
    labels={"a": r"$a$ [ppm]", "b": r"$b$ [$\mu$Hz]", "c": r"$c$ [ppm]",
            "d": r"$d$ [$\mu$Hz]", "l": r"$l$", "k": r"$k$"},
    func_real=modelH_real,
    func_back=modelH_back,
    kallinger_coeffs=K14,
    inits=_inits_H,
    specific_priors=_priors_H,
    ordering_constraint=_ordering_H,
    norm_factors=_norm_H,
    amp_components=("a", "c"),
    corr_computed=("a", "b", "d"),
    transform_observables=_obs_H,
)

MODEL_T = BackgroundModel(
    name="T",
    aliases=("harvey3",),
    description="Three Harvey-like granulation components",
    params=("a", "b", "c", "d", "e", "f", "l", "k", "m"),
    log_params=frozenset({"a", "b", "c", "d", "e"}),
    labels={"a": r"$a$ [ppm]", "b": r"$b$ [$\mu$Hz]", "c": r"$c$ [ppm]",
            "d": r"$d$ [$\mu$Hz]", "e": r"$e$ [ppm]", "f": r"$f$ [$\mu$Hz]",
            "l": r"$l$", "k": r"$k$", "m": r"$m$"},
    func_real=modelT_real,
    func_back=modelT_back,
    kallinger_coeffs=K14,
    inits=_inits_T,
    specific_priors=_priors_T,
    ordering_constraint=_ordering_T,
    norm_factors=_norm_T,
    amp_components=("a", "c", "e"),
    corr_computed=("a", "b", "d"),
    transform_observables=_obs_T,
)
