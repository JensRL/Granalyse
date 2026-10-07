# Adding your own background model

Every background model in Granalyse is one `BackgroundModel` instance — the
pipeline, likelihood, results, radius inference, **and plotting** all interact
with models only through its fields and hooks. Adding a model means filling in
one definition and registering it; no other file should need editing.

## Minimal example

A single-Harvey model with a free white-noise level, as a user would write it:

```python
import jax
import jax.numpy as jnp
import numpy as np

from granalyse import BackgroundModel, register_model
from granalyse import distributions as dist
from granalyse.models.base import xi_analytic


@jax.jit
def mymodel_real(nu, eta, numax_fit, a, b, l, W, Posc, sigma, a2, b2, **kwargs):
    mod = (a**2 / b) / (1 + (nu / b) ** l)
    osc = Posc * jnp.exp(-0.5 * ((nu - numax_fit) / sigma) ** 2)
    activity = (2 / jnp.pi) ** 2 * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + osc + activity) + W


@jax.jit
def mymodel_back(nu, eta, a, b, l, W, a2, b2, **kwargs):
    mod = (a**2 / b) / (1 + (nu / b) ** l)
    activity = (2 / jnp.pi) ** 2 * (a2**2 / b2) / (1 + (nu / b2) ** 2)
    return eta**2 * (mod + activity) + W


K14 = {"a": (3.53, -0.609), "b": (-0.499, 0.970)}  # Kallinger 2014 log10 coeffs


def my_inits(numax):
    return {"a": float(10 ** (K14["a"][0] + np.log10(numax) * K14["a"][1])),
            "b": float(10 ** (K14["b"][0] + np.log10(numax) * K14["b"][1])),
            "l": 4.0}


def my_priors(fit):
    # called by set_priors; return priors for YOUR parameters only
    # (W, Posc, sigma, a2, b2, numax_fit are handled centrally)
    out = {"l": dist.beta(a=1.8, b=3.0, loc=1, scale=8)}
    if not fit.corr_inference:
        out["a"] = dist.normal(loc=K14["a"][0] + np.log10(fit.numax) * K14["a"][1], scale=0.1)
        out["b"] = dist.normal(loc=K14["b"][0] + np.log10(fit.numax) * K14["b"][1], scale=0.1)
    return out


MY_MODEL = BackgroundModel(
    name="M",
    aliases=("myharvey",),
    description="Example user model",
    params=("a", "b", "l"),
    log_params=frozenset({"a", "b"}),
    labels={"a": r"$a$ [ppm]", "b": r"$b$ [$\mu$Hz]", "l": r"$l$"},
    func_real=mymodel_real,
    func_back=mymodel_back,
    kallinger_coeffs=K14,
    inits=my_inits,
    specific_priors=my_priors,
    ordering_constraint=jax.jit(lambda **kw: 0.0),
    norm_factors=lambda fit: {"a": xi_analytic(fit.fitresults["l"][0])},
    amp_components=("a",),
    corr_computed=("a", "b"),       # which params follow the Kallinger scaling relations (used for correlated inference mode)
    has_radius_relation=False,      # no derived granulation-radius relation
)

register_model(MY_MODEL)
```

After `register_model`, the model is available everywhere by name or alias —
in YAML configs (`models: [M]`), the CLI, and the Python API.

## Field/hook reference

| Field | Used by | Purpose |
|---|---|---|
| `params`, `log_params`, `labels` | parameter assembly, corner plots | parameter names (canonical order), which are fitted in log10, LaTeX labels |
| `func_real`, `func_back` | likelihood, plotting | jax model with/without the oscillation envelope; must accept `**kwargs` |
| `kallinger_coeffs` | priors, inits | `{param: (k, e)}` for `10**(k + e*log10(numax))` scaling centers |
| `inits(numax)` | priors | initial parameter estimates |
| `specific_priors(fit)` | `set_priors` | model-specific prior distributions (both inference modes) |
| `ordering_constraint(**params)` | likelihood | jax function returning `0.0` or `-inf` (component ordering) |
| `norm_factors(fit)`, `amp_components` | results | amplitude normalisation factors and which amplitudes form the PDS area |
| `corr_computed` | likelihood, plotting, results | which of `a`, `b`, `d` are reconstructed from numax x scatter in correlated mode |
| `transform_observables(df, corr)` | radius inference | posterior → (charb, A_gran); `None` + `has_radius_relation=False` to opt out |
| `plot_components(ax, freq, eta, params, power_max)` | PDS plot | optional override of the component visualisation; returns floor levels for the y-limit. Default draws one dashed Harvey profile per `amp_components` entry (pairing a/b/l, c/d/k, e/f/m). Model J overrides this to draw vertical lines at b and d. |

## Conventions to respect

- Parameter names: amplitudes `a, c, e` pair with characteristic frequencies
  `b, d, f` and exponents `l, k, m`. The generic plotting, correlated-inference
  reconstruction, and results summaries rely on this pairing — models using
  different names should override `plot_components` and avoid `corr_computed`.
- The common parameters `W, Posc, sigma, a2, b2, numax_fit` (and Peakbogging's
  `H`, `beta`) are appended and given priors automatically.
- A new model has no granulation–radius relation unless you derive one
  (see `calibration/`) and provide `transform_observables`.
