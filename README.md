# Granalyse

**Granulation Analysis with Nested Sampling for Stellar Radius Inference**

Granalyse fits granulation background models to stellar power density spectra
with Bayesian nested sampling, and infers stellar radii from the resulting
granulation properties through calibrated granulation–radius relations.

Contributors: Jens R. Larsen, Mia S. Lundkvist, Guy R. Davies, Martin B. Nielsen, 

**Work in progress repository!**

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/JensRL/Granalyse.git
cd Granalyse
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .                # core pipeline
```

Optionally, to be able to re-derive the granulation–radius relations yourself, install the
calibration extra:

```bash
pip install -e .[calibration]   # optional relation re-derivation
```

## Quick start

Create a configuration file (see [examples/example_config.yaml](examples/example_config.yaml)):

```yaml
output_dir: ./results
models: [J, H, T]
stars:
  - name: KIC8006161
    pds_file: YOUR/PATH/TO/DATA/kplr008006161_kasoc-psd_slc_v1.pow
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

**Computation time**: note that the nested sampling is computationally expensive and, 
depending on the duration and quality of the PDS, the runtime per model may reach up to 1h.

**Output size**: the saved figures and posteriors can be quite large, meaning around 40-50 MB 
of data is generated for each chosen model per star.


## What it does

For each star you provide (one or many):

1. **Prepare the PDS data**: instrumental-systematics masking, optional
   binning to a target resolution, low-frequency cut, apodisation...
2. **Define priors**: empirical scaling relations given the star's νmax (Kallinger et al. 2014)
3. **Model the background**: run nested sampling ([dynesty](https://dynesty.readthedocs.io)) 
  for the chosen background models, with JAX-accelerated likelihoods. 
  Posterior samples, evidences, covariances, and diagnostic figures are produced per star and model.
4. **Infer the radius** from the granulation signals and atmospheric parameters
   using the shipped granulation-radius relations, marginalising over the chosen models by their Bayesian evidence (BMA).

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

The inferred background of KIC 8006161 using model H as presented in the resulting PDS_Fit.pdf looks like this: 
![PDS fit of KIC 8006161 with model H](docs/figures/KIC8006161_modelH_PDS_Fit.png)

The model-specific radius posteriors and resulting BMA radius posterior outputted as RadiusPosteriors.pdf looks like this:
![Radius posteriors for KIC 8006161](docs/figures/KIC8006161_RadiusPosteriors.png)



## Citations

Granalyse is the public merger product between two data analysis frameworks from previous research publications. 
If you use Granalyse, please cite the following papers: 
- [Larsen et al. 2025](https://doi.org/10.1051/0004-6361/202558683)
- [Larsen et al. 2026a](https://doi.org/10.1051/0004-6361/202555661)
- [Larsen et al. 2026b](https://doi.org/10.48550/arXiv.2605.26742)

## License
MIT — see [LICENSE](LICENSE).