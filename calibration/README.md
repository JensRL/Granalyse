# calibration/ — Re-derive granulation-radius scaling relations

This directory is a **self-contained optional component** that lets you
re-derive the granulation-radius relation coefficients from your own
sample of stars with known radii, instead of using the coefficients shipped
with `granalyse` (seed-42 relations from arXiv:2605.26742).

---

## What it does

For each star in your sample it:

1. Finds the granalyse nested-sampling posteriors under `<results-dir>/<name>/model_<MODEL>_*/FitData_model_<MODEL>.csv`.
2. Transforms the raw posteriors to physical `(charb [μHz], A_gran [ppm])` using `granalyse.radius.transform_posterior_to_observables`.
3. Computes the **per-star posterior median of log₁₀(A_gran) and log₁₀(charb)** and their **2×2 covariance matrix** via `np.cov`.
4. Normalises all quantities to **relative space** (divided by the Perky reference star KIC 6106415), exactly as `pystan_helpers.prepare_data_for_stan`.
5. Assembles the Stan data dict and runs a **hierarchical Stan regression** (4 chains × 10 000 iterations, adapt_delta=0.95, max_treedepth=12).
6. Saves the coefficient draws as `relation_coefficients_<MODEL>_seeduser.csv`.

The regression model may be seen in `relation_model.stan`

---

## Requirements

```bash
pip install granalyse[calibration]   # adds cmdstanpy + arviz
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

---

## Input: sample.csv

| Column       | Description                            |
|--------------|----------------------------------------|
| `name`       | Star identifier (= subdirectory name under `results-dir`) |
| `radius`     | Stellar radius [R☉]                   |
| `radius_err` | 1-sigma uncertainty [R☉]              |
| `teff`       | Effective temperature [K]             |
| `teff_err`   | 1-sigma uncertainty [K]               |
| `feh`        | [Fe/H] [dex]  (blank/NaN → prior)    |
| `feh_err`    | Not used by the Stan model (for your records only) |

---

## Usage

```bash
# Full run
python calibration/derive_relations.py sample.csv \
    --results-dir /path/to/granalyse/results \
    --model H \
    --output-dir ./my_calibration

# Dry run (validate data without sampling — no CmdStan required)
python calibration/derive_relations.py sample.csv \
    --results-dir /path/to/granalyse/results \
    --model H --dry-run
```

All flags: `--model {H,J,T}`, `--mission {Kepler,TESS}`, `--output-dir`,
`--chains`, `--iterations`, `--adapt-delta`, `--max-treedepth`, `--dry-run`.

---

## Output

| File | Description |
|------|-------------|
| `relation_coefficients_<MODEL>_seeduser.csv` | Coefficient draws (fac, x, y, z, z2, sigma) |
| `fit_data_<MODEL>_user.csv` | Full MCMC draws from cmdstanpy |
| `fit_<MODEL>_user.netcdf` | ArviZ InferenceData object |

### Loading user coefficients with granalyse

The output filename follows the loader convention
`relation_coefficients_{model}_seed{seed}.csv` with `seed='user'`.  Copy the
file into `granalyse/radius/relations/` and load with
`load_relation_coefficients(seed='user')`, or load it manually with pandas.

---

## Runtime

- ~10–30 min per model on a modern laptop (20–200 stars, 4 chains × 6 667 sampling draws).
- Use `--chains 2 --iterations 4000` for a quick exploratory run.

---

## Stan model variants

`relation_model.stan` contains the **Gaussian measurement-error** variant
(`lin_code_feh_gaussian` from GranRadRels/pystan_helpers.py), which was used
for the published seed-42 relations.

A **Student-t** variant (`lin_code_feh_studentt`) also exists in the original
file.  It adds a degrees-of-freedom parameter `nu ~ gamma(2, 0.1)` and uses
`student_t(nu, ...)` likelihoods for robustness to outliers.  It is not
ported here but can be added by copying the string from pystan_helpers.py.

---

## Differences from the published derivation

1. **No multi-seed drawing** — the published workflow uses 50 independent
   random draws of 50% of the calibration pool and averages the resulting
   coefficients (`run_seed_ensemble.py`).  This script performs a single fit
   on the entire sample you provide.

2. **No source stratification** — the published pool stratifies across Legacy,
   KAGES, APOKASC1, seismic, and interferometric sub-samples.  User-supplied
   samples are taken as-is.

3. **RGB exclusion left to the user** — stars with R > 5 R☉ are excluded in
   the published derivation.  Filter your sample before passing it in.

4. **Perky included if present** — KIC 6106415 is the normalisation anchor.
   If it appears in your sample it will be included (unlike the published
   workflow which excludes it).  Remove it manually if needed.

5. **Single mission** — use `--mission TESS` for TESS targets; the published
   calibration used Kepler `bol_ref=5934 K, bol_exp=0.80`.
