# Granalyse

**Granulation Analysis with Nested Sampling for Stellar Radius Inference**

Granalyse fits granulation background models to stellar power density spectra
with Bayesian nested sampling, and infers stellar radii from the resulting
granulation properties through calibrated granulation–radius relations.

**Work in progress repository!**

## What it does

For each star you provide (one or many):

1. **Prepare** the power spectrum: instrumental-systematics masking, optional
   binning to a target resolution, low-frequency cut, apodisation...
2. **Define priors** from empirical scaling relations (Kallinger et al. 2014) given the star's νmax — supplied directly, or
   estimated from log g and Teff via the asteroseismic νmax scaling relation.
3. **Fit** the chosen background models with nested sampling
   ([dynesty](https://dynesty.readthedocs.io)), with JAX-accelerated
   likelihoods. Posterior samples, evidences, covariances, and diagnostic
   figures are produced per star and model.
4. **Infer the radius** from the granulation signals and atmospheric parameters
   using the shipped granulation-radius relations, combining the fitted models by their Bayesian evidence (BMA).

## Background models

| Name | Alias | Form | Radius relation |
|------|-------|------|-----------------|
| `D` | `harvey1` | single Harvey-like component | x |
| `J` | `hybrid`  | single amplitude, two characteristic frequencies | ✓ |
| `H` | `harvey2` | two Harvey-like components | ✓ |
| `T` | `harvey3` | three Harvey-like components | ✓ |

All models share a white-noise level, a low-frequency activity term, and (in
the default Gaussian mode) a Gaussian oscillation envelope. Models are
**pluggable**: each is a single `BackgroundModel` definition bundling its
parameters, priors, initial guesses, normalisation, constraints, and plot representation. 
See [docs/adding_models.md](docs/adding_models.md) for a worked example of registering your own model.

## Installation

```bash
git clone https://github.com/jensRL/Granalyse.git
cd Granalyse
pip install -e .               # core pipeline
pip install -e .[calibration]  # + optional relation re-derivation
```

Requires Python ≥ 3.11.

## Quick start

Create a configuration file (see [examples/example_config.yaml](examples/example_config.yaml)):

```yaml
output_dir: ./results
models: [J, H, T]
stars:
  - name: KIC8006161
    pds_file: data/kplr008006161_kasoc-psd_slc_v1.pow
    numax: 3575          # muHz; or give logg + teff instead
    teff: 5488
    teff_err: 77
    feh: 0.34
    feh_err: 0.1
    mission: Kepler       # or TESS (sets the bolometric correction)
```

then run:

```bash
granalyse run config.yaml
```

## Citing

Granalyse is the public merger product between two data analysis frameworks from previous research publications. 
If you use Granalyse, please cite the following papers: 
- [Larsen et al. 2025](https://doi.org/10.1051/0004-6361/202558683)
- [Larsen et al. 2026a](https://doi.org/10.1051/0004-6361/202555661)
- [Larsen et al. 2026b](https://doi.org/10.48550/arXiv.2605.26742)

## License
MIT — see [LICENSE](LICENSE).