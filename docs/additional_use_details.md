## Generating a config for many stars

For large samples, use `make-config` to scaffold the YAML from a CSV/TSV
table of stellar parameters and/or a directory of PDS files:

```bash
# From a star table (delimiter is sniffed automatically):
granalyse make-config --stars my_stars.csv --output config.yaml

# From a directory of spectra (names extracted from filenames):
granalyse make-config --pds-dir data/spectra/ --no-radius-inference

# Both — CSV rows matched to files by extracted star name:
granalyse make-config --stars my_stars.csv --pds-dir data/spectra/
```

Recognised CSV columns: `name`, `pds_file`, `numax`, `logg`, `teff`,
`teff_err`, `feh`, `feh_err`, `mission`, `kind`, `nyquist`.  Any star missing
required fields is written with `null  # FILL IN` placeholders and listed in a
summary; when every star is complete, `load_config` is called automatically to
confirm the file is valid.

Per star this produces `results/<name>/model_<X>_*/` (posterior samples,
evidence, fit summary, PDS/corner/diagnostic figures), a `summary.json` with
the BMA radius, and a top-level `overview.csv`. Completed fits are detected
and loaded on re-runs, so interrupted batches resume cheaply. Set
`skip_replot: true` to also reuse existing figures on resume instead of
regenerating them — the corner plots dominate runtime, so this makes
resuming a stopped run substantially faster.

The same steps are available as a Python API:

```python
from granalyse.config import load_config
from granalyse.pipeline import run_pipeline

run_pipeline(load_config("config.yaml"))
```

or piecewise (`granalyse.pipeline.prepare_fit_case`, `run_single_fit`,
`granalyse.radius.infer_radius`) for custom workflows.

Input spectra are two-column ASCII files (frequency [µHz], power density
[ppm²/µHz]), e.g. KASOC power spectra.

## Notes on methodology

- **Correlated inference** (default): granulation amplitudes and timescales
  follow the Kallinger et al. (2014) relations at the sampled νmax, with
  fitted scatter factors. Set `correlated_inference: false` to fit them freely.
- **Peakbogging mode**: a background-only likelihood with a mixture treatment
  of the oscillation region (`pulsation_model: Peakbogging`).
- **Radius inference** propagates the full coefficient posteriors of the
  scaling relations (shipped in `granalyse/radius/relations/`, derivation
  seed 42 of arXiv:2605.26742) including intrinsic scatter, and combines
  models via evidence-weighted Bayesian Model Averaging. It requires Teff
  (and ideally [Fe/H]) for each star.
- **Simulations** (`kind: simulation`, e.g. STAGGER/CO5BOLD spectra) are fit
  through the same pipeline with νmax from log g/Teff and no instrumental
  systematics masking.

## Advanced

- **HPC batch runs**: a SLURM array-job workflow for large samples lives in
  [hpc/](hpc/README.md). The general user does not need it.
- **Re-deriving the relations**: with your own calibration sample (stars with
  known radii), see [calibration/](calibration/README.md). Requires the
  `[calibration]` extra and a CmdStan toolchain.