# calibration/ — Re-derive granulation-radius scaling relations

This directory is a **self-contained optional component** that lets you
re-derive the granulation-radius relation coefficients from your own
sample of stars with known radii, instead of using the coefficients shipped
with `granalyse`.

---

## What it does

For each star in your sample it:

1. Finds the granalyse nested-sampling posteriors under `<results-dir>/<name>/model_<MODEL>_*/FitData_model_<MODEL>.csv`.
2. Transforms the raw posteriors to physical `(charb [μHz], A_gran [ppm])`.
3. Computes the **per-star posterior median of log₁₀(A_gran) and log₁₀(charb)** and their **2×2 covariance matrix**`.
4. Normalises all quantities to **relative space** (divided by the Perky reference star KIC 6106415).
5. Assembles the Stan data dict and runs a **hierarchical Stan regression**.
6. Saves the coefficient draws.

The regression model may be seen in `relation_model.stan`

---

## Requirements

```bash
pip install granalyse[calibration]   # cmdstanpy, arviz, h5netcdf + h5py
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

`h5netcdf`/`h5py` are the netCDF backend arviz needs to write InferenceData;
without them the netCDF output is skipped (everything else still works).

---

## Input: sample.csv

| Column       | Description                            |
|--------------|----------------------------------------|
| `name`       | Star identifier                        |
| `radius`     | Stellar radius [R☉]                    |
| `radius_err` | 1-sigma uncertainty [R☉]               |
| `teff`       | Effective temperature [K]              |
| `teff_err`   | 1-sigma uncertainty [K]                |
| `feh`        | [Fe/H] [dex]  (blank/NaN → prior)      |
| `feh_err`    | Not currently used by the Stan model   |

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
`--chains`, `--iterations`, `--adapt-delta`, `--max-treedepth`, `--dry-run`,
`--reference-json`.

### Using your own reference star

Step 4 normalises everything to Perky (KIC 6106415), a main-sequence star.
A calibration sample far from it — giants, say — is better anchored on a
reference drawn from that sample. Pass one with `--reference-json`:

```json
{
  "name": "KIC 7125569",
  "teff": 4643.5527,
  "R_cm": 646799247000.0,
  "feh": -0.2206,
  "models": {"H": {"amp": 449.8155, "charb": 12.0463}}
}
```

`amp` and `charb` are the posterior medians of that star's own
mission-corrected `A_gran` [ppm] and `charb` [μHz] — the medians of
`transform_posterior_to_observables(post, teff, model, mission=...)` applied to
its `FitData_model_<MODEL>.csv`. Only the models you derive need entries.

The reference enters the regression as a fixed constant, not a marginalised
quantity, so choose a star whose own fit is well determined. **Whatever you
derive with you must also apply with**: pass the same file as `perky_ref` to
`granalyse.radius.infer_radius`. Coefficients and reference star are
meaningless apart.

Omit the flag and the shipped Perky parameters are used, unchanged.

---

## Output

| File | Description |
|------|-------------|
| `relation_coefficients_<MODEL>_seeduser.csv` | Coefficient draws (fac, x, y, z, z2, sigma) |
| `fit_data_<MODEL>_user.csv` | The same draws + chain/draw indices + per-draw sampler diagnostics |
| `fit_summary_<MODEL>_user.csv` | Coefficient mean, sd, intervals, MCSE, ESS, R-hat |
| `fit_diagnostics_<MODEL>_user.txt` | CmdStan `diagnose()` report |
| `fit_<MODEL>_user.netcdf` | ArviZ InferenceData (coefficients + sample stats) |
| `latent_parameters_<MODEL>_user.csv.gz` | Only with `--save-latent` (see below) |

The model samples **5N+6** parameters — five latent parameters per star plus the
six coefficients. At N = 4 000 that is 20 006, so the latent parameters dominate
anything written and are excluded by default; `--save-latent` includes them.
Nothing downstream needs them: the radius inference reads only
`relation_coefficients_<MODEL>_seeduser.csv`.

Writing the netCDF needs a netCDF backend, which arviz does not declare;
`h5netcdf` and `h5py` are therefore part of the `calibration` extra. If they are
missing the file is skipped with a warning rather than failing the run —
`fit_data_<MODEL>_user.csv` carries the same draws.

### Loading user coefficients with granalyse

The output filename follows the loader convention
`relation_coefficients_{model}_seed{seed}.csv` with `seed='user'`.  Copy the
file into `granalyse/radius/relations/` and load with
`load_relation_coefficients(seed='user')`, or load it manually with pandas.

---

## Runtime

- ~10–30 min per model on a modern laptop (20–200 stars, 4 chains × 6 667 sampling draws).
- Use `--chains 2 --iterations 4000` for a quick exploratory run.

Cost grows with sample size: the model gives **every star five latent
parameters**, so N stars means 5N+6 parameters. Measured on an M2 Pro at
N = 4 000 (20 006 parameters), model H: ~3 s per iteration with chains running
in parallel, i.e. ~2.5 h at `--iterations 3000` and ~8.5 h at the 10 000
default, plus ~8 min to read and transform the posteriors before sampling
starts. Only the six coefficients are of interest and they mix quickly, so for
large samples prefer a lower `--iterations` over the default.

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

4. **The reference star is included if present** — the normalisation anchor
   (Perky by default, or whatever `--reference-json` supplies) will be used as
   a data point if it appears in your sample, unlike the published workflow
   which excludes it.  Remove it manually if needed.

5. **Single mission** — use `--mission TESS` for TESS targets; the published
   calibration used Kepler `bol_ref=5934 K, bol_exp=0.80`.
