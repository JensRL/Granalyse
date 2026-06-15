"""BackgroundModel: everything one granulation background model needs to plug
into the pipeline.

This replaces the model-letter if-statements that were spread across the
original NSGF framework (parameter lists, initial guesses, prior blocks,
likelihood closures, normalisation, ordering constraints). A new model is
added by creating one BackgroundModel instance and registering it.
"""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# Common (non-model-specific) fit parameters appended after the granulation
# parameters, in the order used by the original framework.
COMMON_PARAMS = ["W", "Posc", "sigma", "a2", "b2"]
COMMON_LABELS = {
    "W": r"$W$ [ppm$^2/\mu$Hz]",
    "Posc": r"$P_{osc}$ [ppm]",
    "sigma": r"$\sigma$ [$\mu$Hz]",
    "a2": r"$a2$ [ppm]",
    "b2": r"$b2$ [$\mu$Hz]",
    "numax_fit": r"$\nu_{max}$ [$\mu$Hz]",
    "beta": r"$\beta$",
    "H": r"$H$",
    "sigma_a": r"$\sigma_a$ [ppm]",
    "sigma_b": r"$\sigma_b$ [$\mu$Hz]",
    "sigma_d": r"$\sigma_d$ [$\mu$Hz]",
}

# Granulation parameters replaced by scatter factors in correlated inference.
CORR_RENAMES = {"a": "sigma_a", "b": "sigma_b", "d": "sigma_d"}


def xi_analytic(exponent):
    """Analytic Harvey normalisation factor (Karoff et al. 2013)."""
    return 2 * exponent * np.sin(np.pi / exponent) / (2 * np.pi)


def xi_clipped(exponent):
    """xi_analytic with the exponent clipped to [1.1, 20], for vectorised use
    on posterior samples (as in the radius-relation framework)."""
    return xi_analytic(np.clip(exponent, 1.1, 20.0))


def harvey_component_profile(nu, amplitude, charfreq, exponent, eta):
    """Single Harvey-like component (apodised activity form for exponent 2)."""
    if exponent == 2:
        return eta**2 * ((2 / np.pi) ** 2 * (amplitude**2 / charfreq)
                         / (1 + (nu / charfreq) ** exponent))
    return eta**2 * (amplitude**2 / charfreq) / (1 + (nu / charfreq) ** exponent)


def amp_charfreq_exponent_triples(model, param_dict):
    """(amp, charfreq, exponent) value triples for each Harvey amplitude
    component, following the canonical pairing a/b/l, c/d/k, e/f/m."""
    _pairs = {"a": ("b", "l"), "c": ("d", "k"), "e": ("f", "m")}
    triples = []
    for amp in model.amp_components:
        cf_key, exp_key = _pairs[amp]
        if cf_key in param_dict and exp_key in param_dict:
            triples.append((
                float(param_dict[amp]),
                float(param_dict[cf_key]),
                float(param_dict[exp_key]),
            ))
    return triples


@dataclass(frozen=True)
class BackgroundModel:
    """Definition of one granulation background model.

    Hooks taking `fit` receive a FitCase (or any object with the same
    attributes: numax, nyquist, corr_inference, fitresults, ...).
    """

    name: str                       # canonical letter, e.g. 'H'
    aliases: tuple                  # descriptive names, e.g. ('harvey2',)
    description: str

    params: tuple                   # granulation parameters, canonical order
    log_params: frozenset           # subset fitted in log10 space
    labels: dict                    # LaTeX labels for the granulation parameters

    func_real: Callable             # jax model incl. oscillation envelope
    func_back: Callable             # jax model, background only

    # (k, e) per parameter for mu = 10**(k + e*log10(numax)), Kallinger 2014
    kallinger_coeffs: dict

    inits: Callable                 # numax -> dict of initial guesses
    specific_priors: Callable       # fit -> dict of model-specific priors
    ordering_constraint: Callable   # jax (**params) -> 0.0 or -inf
    norm_factors: Callable          # fit -> {amp param: xi}
    amp_components: tuple           # amplitude params entering the normalised area
    corr_computed: tuple            # of 'a','b','d': computed from scaling in corr mode

    has_radius_relation: bool = True
    # (posterior_df, corr_inference=None) -> (charb, A_gran) per-sample arrays,
    # A_gran before the bolometric correction; None if the model has no
    # granulation-radius relation.
    transform_observables: Callable | None = None
    # (ax, freq, eta, params, power_max) -> list of component floor levels;
    # draws the model's granulation components on a PDS axis. None uses the
    # generic per-component Harvey profiles.
    plot_components: Callable | None = None

    @property
    def n_components(self):
        return len(self.amp_components)

    def draw_pds_components(self, ax, freq, eta, params, power_max):
        """Draw this model's granulation components on a PDS axis.

        `params` holds physical median values keyed by parameter name.
        Returns a list of power levels (component floors) that the plot's
        lower y-limit should accommodate. Override per model by setting
        `plot_components`; the default draws one dashed profile per Harvey
        amplitude component.
        """
        if self.plot_components is not None:
            return self.plot_components(ax, freq, eta, params, power_max)
        levels = []
        for amp, cf, exp in amp_charfreq_exponent_triples(self, params):
            ax.plot(freq, harvey_component_profile(freq, amp, cf, exp, eta),
                    "g", linestyle="dashed", linewidth=1,
                    label="Individual components", rasterized=True)
            levels.append(amp**2 / cf)
        return levels

    def prior_centers(self, numax):
        """Model-neutral linear-scale centers for a, b, c, d from the
        Kallinger 2014 scaling laws (original common_centers_from_numax)."""
        return {
            p: 10 ** (k + np.log10(numax) * e)
            for p, (k, e) in self.kallinger_coeffs.items()
        }

    def configure_fit_parameters(self, fit):
        """Set fit.fitparams, fit.logPars, fit.paramlabels on a FitCase,
        applying the pulsation-model and correlated-inference adjustments.

        Reproduces the parameter-assembly logic of the original main scripts.
        """
        fitparams = list(self.params) + list(COMMON_PARAMS)
        logPars = set(self.log_params)
        labels = {**self.labels, **{p: COMMON_LABELS[p] for p in COMMON_PARAMS}}

        # numax is a free parameter when fitting a real star
        if fit.data_format == "STAR":
            fitparams.append("numax_fit")
            labels["numax_fit"] = COMMON_LABELS["numax_fit"]
            if fit.pulsmod == "Peakbogging":
                fitparams.append("beta")
                labels["beta"] = COMMON_LABELS["beta"]
                # The envelope is a scaled mixture: Posc becomes the mixture
                # height H, and sigma is fixed (Mosser et al. 2012 width).
                logPars.discard("Posc")
                fitparams[fitparams.index("Posc")] = "H"
                labels["H"] = COMMON_LABELS["H"]
                if "sigma" in fitparams:
                    fitparams.remove("sigma")
                    logPars.discard("sigma")
                    labels.pop("sigma", None)
                    fit.sigwidth = 1.5 * (
                        1 / (2 * np.sqrt(2 * np.log(2))) * (0.66 * fit.numax**0.88)
                    )

        if fit.corr_inference:
            for orig, scat in CORR_RENAMES.items():
                if orig in fitparams and orig in self.corr_computed:
                    fitparams[fitparams.index(orig)] = scat
                    labels[scat] = COMMON_LABELS[scat]
                    labels.pop(orig, None)
                    # only sigma_a is sampled in log space (log-normal scatter)
                    logPars.discard(orig)
                    if scat == "sigma_a":
                        logPars.add(scat)

        fit.fitparams = fitparams
        fit.logPars = logPars
        fit.paramlabels = {p: labels[p] for p in fitparams if p in labels}
        return fit
