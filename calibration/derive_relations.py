#!/usr/bin/env python3
"""derive_relations.py
======================
Re-derive granulation-radius scaling-relation coefficients from  a sample of stars with known radii.

Usage
-----
    python calibration/derive_relations.py sample.csv \\
        --results-dir /path/to/results \\
        --model H \\
        --output-dir ./my_calibration

    # Dry-run: assemble+validate the Stan data dict without sampling
    python calibration/derive_relations.py sample.csv \\
        --results-dir /path/to/results \\
        --model H \\
        --output-dir ./my_calibration \\
        --dry-run

    # Normalise against your own reference star instead of Perky
    python calibration/derive_relations.py sample.csv \\
        --results-dir /path/to/results \\
        --model H \\
        --reference-json my_reference.json

sample.csv columns (required)
------------------------------
    name        – star identifier (must match subdirectory name under results_dir)
    radius      – stellar radius [R_sun]
    radius_err  – 1-sigma uncertainty [R_sun]
    teff        – effective temperature [K]
    teff_err    – 1-sigma uncertainty [K]
    feh         – [Fe/H] [dex]  (may be empty/NaN -> draws from prior)
    feh_err     – not used in the Stan model but expected for completeness

Output
------
    <output-dir>/relation_coefficients_<MODEL>_seeduser.csv
        Columns: fac, x, y, z, z2, sigma
        Directly loadable by granalyse.radius.load_relation_coefficients(seed='user').

    <output-dir>/fit_data_<MODEL>_user.csv
        The same coefficient draws plus chain/draw indices and the per-draw
        sampler diagnostics.

    <output-dir>/fit_summary_<MODEL>_user.csv
        Coefficient posterior summary: mean, sd, intervals, MCSE, ESS, R-hat.

    <output-dir>/fit_diagnostics_<MODEL>_user.txt
        CmdStan diagnose() report.

    <output-dir>/fit_<MODEL>_user.netcdf
        ArviZ InferenceData (coefficients + sample stats); skipped with a
        warning if no netCDF backend is installed.

    <output-dir>/latent_parameters_<MODEL>_user.csv.gz
        Only with --save-latent: the 5N per-star latent parameters.

The model samples 5N+6 parameters, so on a large sample the latent per-star
parameters dominate everything written. They are excluded by default.
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bolometric-correction parameters per mission (keep in sync with granalyse/radius/apply.py)
# ---------------------------------------------------------------------------
_BOLOMETRIC = {
    "Kepler": (5934.0, 0.80),
    "TESS":   (4714.0, 0.81),
}

R_SUN_CM = 6.957e10  # cm


# ---------------------------------------------------------------------------
# Reference star
# ---------------------------------------------------------------------------
def load_reference_params(path=None, model=None):
    """Reference-star parameters used to normalise observables into relative space.

    With `path` given, read that JSON; otherwise fall back to the shipped Perky
    (KIC 6106415) parameters. The file must follow the same layout as
    granalyse/radius/relations/perky_params.json:

        {"name": ..., "teff": K, "R_cm": cm, "feh": dex,
         "models": {"H": {"amp": ppm, "charb": muHz}, ...}}

    `amp` must carry the same bolometric correction as the calibration sample,
    i.e. be the posterior median of the mission-corrected A_gran.
    """
    if path is None:
        from granalyse.radius import load_perky_params
        return load_perky_params()

    with open(path) as f:
        ref = json.load(f)

    missing = {"teff", "R_cm", "feh", "models"} - set(ref)
    if missing:
        raise ValueError(f"{path}: reference JSON missing key(s) {sorted(missing)}")
    if model is not None:
        if model not in ref["models"]:
            raise ValueError(f"{path}: reference JSON has no entry for model {model}")
        sub_missing = {"amp", "charb"} - set(ref["models"][model])
        if sub_missing:
            raise ValueError(
                f"{path}: model {model} missing key(s) {sorted(sub_missing)}"
            )
    return ref


# ---------------------------------------------------------------------------
# Posterior transformation
# ---------------------------------------------------------------------------
def _transform_posterior_to_observables(post_df, teff, model, mission="Kepler",
                                         corr_inference=None):
    """Wrapper around granalyse.radius.transform_posterior_to_observables.

    Returns (charb_samples [muHz], agran_samples [ppm]).
    """
    from granalyse.radius import transform_posterior_to_observables
    return transform_posterior_to_observables(
        post_df, float(teff), model,
        mission=mission, corr_inference=corr_inference,
    )


# ---------------------------------------------------------------------------
# Per-star data preparation
# ---------------------------------------------------------------------------
def _find_posterior_file(results_dir, star_name, model):
    """
    Glob for FitData_model_<model>.csv[.gz] under results_dir/<star_name>/model_<model>_*/.
    """
    base = os.path.join(
        results_dir, star_name,
        f"model_{model}_*",
        f"FitData_model_{model}.csv",
    )
    matches = sorted(glob.glob(base + ".gz")) or sorted(glob.glob(base))
    return matches[0] if matches else None


def _compute_per_star_stats(charb_s, agran_s):
    """
    Compute log10 medians and 2x2 covariance from posterior samples.

    Returns
    -------
    lg_amp_mean   : float   median of log10(A_gran) samples
    lg_charb_mean : float   median of log10(charb) samples
    cov_2x2       : ndarray shape (2, 2)   [[var_amp, cov], [cov, var_charb]]
    """
    lg_a = np.log10(agran_s)
    lg_c = np.log10(charb_s)
    cov_m = np.cov(np.vstack([lg_a, lg_c]))
    return float(np.median(lg_a)), float(np.median(lg_c)), cov_m


# ---------------------------------------------------------------------------
# Stan data assembly
# ---------------------------------------------------------------------------
def build_stan_data(sample_df, results_dir, model, mission, reference):
    """
    Load posteriors, transform, normalise to the reference star, and assemble
    the Stan dict.

    Parameters
    ----------
    sample_df   : pd.DataFrame  (from sample.csv)
    results_dir : str or Path
    model       : 'H', 'J', or 'T'
    mission     : 'Kepler' or 'TESS'
    reference   : dict from load_reference_params()

    Returns
    -------
    stan_data : dict  (ready for CmdStanModel.sample)
    df_rel    : pd.DataFrame  (relative log-space values used for Stan)
    skipped   : list[str]   stars that could not be processed
    """
    ref_amp   = reference["models"][model]["amp"]
    ref_charb = reference["models"][model]["charb"]
    ref_teff  = reference["teff"]
    ref_R_cm  = reference["R_cm"]
    ref_feh   = reference["feh"]

    rows = []
    skipped = []

    for _, row in sample_df.iterrows():
        name = str(row["name"])
        post_file = _find_posterior_file(results_dir, name, model)
        if post_file is None:
            print(f"  WARNING: no posteriors for '{name}' (model {model}), skipping")
            skipped.append(name)
            continue

        try:
            post_df = pd.read_csv(post_file)
        except Exception as exc:
            print(f"  WARNING: could not read posteriors for '{name}': {exc}, skipping")
            skipped.append(name)
            continue

        try:
            charb_s, agran_s = _transform_posterior_to_observables(
                post_df, row["teff"], model, mission=mission,
            )
        except Exception as exc:
            print(f"  WARNING: transform failed for '{name}': {exc}, skipping")
            skipped.append(name)
            continue

        lg_amp_mean, lg_charb_mean, cov_m = _compute_per_star_stats(charb_s, agran_s)

        # ── Normalise to relative space ──────────────────────────────────────
        R_cm   = float(row["radius"])    * R_SUN_CM
        R_cm_e = float(row["radius_err"]) * R_SUN_CM

        R_rel     = R_cm   / ref_R_cm
        R_rel_err = R_cm_e / ref_R_cm

        teff_rel     = float(row["teff"])     / ref_teff
        teff_err_rel = float(row["teff_err"]) / ref_teff

        # log-space observables (relative) for gran amplitude and charb
        lg_amp_rel   = lg_amp_mean   - np.log10(ref_amp)
        lg_charb_rel = lg_charb_mean - np.log10(ref_charb)

        lg_R_rel     = np.log10(R_rel)
        lg_R_rel_err = R_rel_err / (R_rel * np.log(10))

        lg_teff_rel     = np.log10(teff_rel)
        lg_teff_rel_err = teff_err_rel / (teff_rel * np.log(10))

        feh_raw = row.get("feh", np.nan)
        has_feh = 0
        feh_val = 0.0
        if not pd.isna(feh_raw):
            try:
                feh_val = float(feh_raw) - ref_feh
                has_feh = 1
            except (TypeError, ValueError):
                pass

        rows.append({
            "name":          name,
            "lg_amp_obs":    lg_amp_rel,
            "lg_charb_obs":  lg_charb_rel,
            "cov_2x2":       cov_m,
            "lg_teff_obs":   lg_teff_rel,
            "lg_teff_err":   lg_teff_rel_err,
            "lg_R_obs":      lg_R_rel,
            "lg_R_err":      lg_R_rel_err,
            "feh_obs":       feh_val,
            "has_feh_obs":   has_feh,
        })

    if not rows:
        raise RuntimeError(
            "No stars could be processed — check results_dir and star names."
        )

    df_rel = pd.DataFrame(rows)
    N = len(df_rel)

    gran_cov = np.stack(df_rel["cov_2x2"].values)  # (N, 2, 2)

    stan_data = {
        "N":             N,
        "lg_amp_obs":    df_rel["lg_amp_obs"].values.tolist(),
        "lg_charb_obs":  df_rel["lg_charb_obs"].values.tolist(),
        "gran_cov":      gran_cov.tolist(),
        "lg_teff_obs":   df_rel["lg_teff_obs"].values.tolist(),
        "lg_teff_err":   df_rel["lg_teff_err"].values.tolist(),
        "lg_R_obs":      df_rel["lg_R_obs"].values.tolist(),
        "lg_R_err":      df_rel["lg_R_err"].values.tolist(),
        "feh_obs":       df_rel["feh_obs"].values.tolist(),
        "has_feh_obs":   df_rel["has_feh_obs"].values.tolist(),
    }

    return stan_data, df_rel, skipped


# ---------------------------------------------------------------------------
# Validation of Stan data dict
# ---------------------------------------------------------------------------
def validate_stan_data(stan_data):
    """Check shapes and finiteness; raise ValueError on failure."""
    N = stan_data["N"]
    expected_shapes = {
        "lg_amp_obs":   (N,),
        "lg_charb_obs": (N,),
        "gran_cov":     (N, 2, 2),
        "lg_teff_obs":  (N,),
        "lg_teff_err":  (N,),
        "lg_R_obs":     (N,),
        "lg_R_err":     (N,),
        "feh_obs":      (N,),
        "has_feh_obs":  (N,),
    }
    errors = []
    for key, expected in expected_shapes.items():
        arr = np.asarray(stan_data[key])
        if arr.shape != expected:
            errors.append(f"  {key}: expected shape {expected}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            n_bad = np.sum(~np.isfinite(arr))
            errors.append(f"  {key}: {n_bad} non-finite value(s)")

    # lg_teff_err and lg_R_err must be positive
    for key in ("lg_teff_err", "lg_R_err"):
        arr = np.asarray(stan_data[key])
        if np.any(arr <= 0):
            errors.append(f"  {key}: contains non-positive values")

    # gran_cov must be symmetric positive-definite for each star
    gran_cov = np.asarray(stan_data["gran_cov"])
    for i in range(N):
        try:
            np.linalg.cholesky(gran_cov[i])
        except np.linalg.LinAlgError:
            errors.append(f"  gran_cov[{i}]: not positive-definite")

    if errors:
        raise ValueError("Stan data validation failed:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Initial values
# ---------------------------------------------------------------------------
#: Declared bounds of `feh_true` in relation_model.stan (parameters block).
#: Keep in step with that declaration if it ever changes.
_FEH_TRUE_BOUNDS = (-4.0, 1.0)

#: How far inside a bound an initial value is placed. Stan rejects an init that
#: sits exactly on a bound as well as one outside it — the unconstraining
#: transform diverges there — so the clip has to land strictly inside.
_INIT_MARGIN = 1e-4


def _get_initial_values(df_rel):
    lo, hi = _FEH_TRUE_BOUNDS
    feh_obs = np.asarray(df_rel["feh_obs"].values, dtype=float)
    feh_init = np.clip(feh_obs, lo + _INIT_MARGIN, hi - _INIT_MARGIN)
    clipped = feh_init != feh_obs
    if clipped.any():
        outside = feh_obs[clipped]

    return {
        "lg_teff_true": df_rel["lg_teff_obs"].values.tolist(),
        "lg_R_true":    df_rel["lg_R_obs"].values.tolist(),
        "lg_amp_true":  df_rel["lg_amp_obs"].values.tolist(),
        "lg_charb_true":df_rel["lg_charb_obs"].values.tolist(),
        "feh_true":     feh_init.tolist(),
        "fac":   0.01,
        "x":     0.5,
        "y":    -0.7,
        "z":     1.5,
        "z2":    0.1,
        "sigma": 0.03,
    }


# ---------------------------------------------------------------------------
# Stan fitting
# ---------------------------------------------------------------------------
def run_stan(stan_data, df_rel, output_dir, model,
             chains=4, iterations=10000,
             adapt_delta=0.95, max_treedepth=12, refresh=None, seed=None):
    """Compile and sample the Stan model; return (fit, diagnostics).

    Sampling setup:
      - iter_warmup  = iterations // 3
      - iter_sampling = 2 * iterations // 3
      - per-chain init with 5% random jitter on scalar coefficients (rng seed 42)
    """
    try:
        from cmdstanpy import CmdStanModel
    except ImportError as exc:
        raise ImportError(
            "calibration extras not installed; run: pip install granalyse[calibration]"
        ) from exc

    stan_file = Path(__file__).parent / "relation_model.stan"

    print(f"\nCompiling Stan model from {stan_file} ...")
    sm = CmdStanModel(stan_file=str(stan_file))

    initial_values = _get_initial_values(df_rel)
    rng = np.random.default_rng(42)
    inits_list = []
    for _ in range(chains):
        chain_inits = initial_values.copy()
        for key in ("fac", "x", "y", "z", "sigma"):
            chain_inits[key] = chain_inits[key] * (1 + rng.normal(0, 0.05))
        chain_inits["z2"] = chain_inits["z2"] * (1 + rng.normal(0, 0.05))
        inits_list.append(chain_inits)

    print(f"Sampling: {chains} chains x {int(2*iterations/3)} draws "
          f"({int(iterations/3)} warmup)  adapt_delta={adapt_delta}  "
          f"max_treedepth={max_treedepth}")

    fit = sm.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=int(iterations / 3),
        iter_sampling=int(2 * iterations / 3),
        inits=inits_list,
        show_progress=True,
        refresh=refresh,
        seed=seed,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )

    diagnostics = fit.diagnose()
    print("\n" + diagnostics)

    return fit, diagnostics


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(fit, model, output_dir, diagnostics=None, save_latent=False):
    """Save the coefficient draws and everything needed to audit the fit.

    Files written
    -------------
    relation_coefficients_<MODEL>_seeduser.csv
        fac, x, y, z, z2, sigma — one row per draw. 
    fit_data_<MODEL>_user.csv
        The same coefficient draws plus chain/draw indices and the per-draw
        sampler diagnostics (lp__, accept_stat__, stepsize__, treedepth__,
        n_leapfrog__, divergent__, energy__). 
    fit_summary_<MODEL>_user.csv
        Posterior summary of the coefficients: mean, sd, intervals, MCSE,
        ESS bulk/tail and R-hat.
    fit_diagnostics_<MODEL>_user.txt
        The CmdStan `diagnose()` report.
    fit_<MODEL>_user.netcdf
        ArviZ InferenceData (coefficients + sample stats). 
    latent_parameters_<MODEL>_user.csv.gz
        Only with `save_latent`: the 5N per-star latent parameters.
    """
    import arviz as az

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_cols = ["fac", "x", "y", "z", "z2", "sigma"]

    # ── 1. coefficient draws ───────────────────────────────────────────────
    coeffs_df = fit.draws_pd(vars=coeff_cols)
    coeff_path = output_dir / f"relation_coefficients_{model}_seeduser.csv"
    coeffs_df.to_csv(coeff_path, index=False)
    print(f"\nCoefficient draws saved to {coeff_path}  "
          f"({len(coeffs_df)} draws)")

    # ── 2. coefficients + per-draw sampler diagnostics ──────────────────────
    method_vars = fit.method_variables()
    n_draws, n_chains = next(iter(method_vars.values())).shape
    audit = pd.DataFrame({
        "chain__": np.repeat(np.arange(1, n_chains + 1), n_draws),
        "draw__":  np.tile(np.arange(1, n_draws + 1), n_chains),
    })
    for key, arr in method_vars.items():
        audit[key] = arr.flatten(order="F")
    audit = pd.concat([audit, coeffs_df.reset_index(drop=True)], axis=1)
    full_path = output_dir / f"fit_data_{model}_user.csv"
    audit.to_csv(full_path, index=False)
    print(f"Coefficient draws + sampler diagnostics saved to {full_path}")

    # ── 3. InferenceData, summary, diagnose report ──────────────────────────
    _AZ_STATS = {"lp__": "lp", "accept_stat__": "acceptance_rate",
                 "stepsize__": "step_size", "treedepth__": "tree_depth",
                 "n_leapfrog__": "n_steps", "divergent__": "diverging",
                 "energy__": "energy"}
    draws_xr = fit.draws_xr(vars=coeff_cols)
    idata = az.from_dict({
        "posterior": {v: draws_xr[v].transpose("chain", "draw").values
                      for v in coeff_cols},
        "sample_stats": {_AZ_STATS.get(k, k): v.T
                         for k, v in method_vars.items()},
    })

    summary = az.summary(idata)
    summary_path = output_dir / f"fit_summary_{model}_user.csv"
    summary.to_csv(summary_path)
    print(f"Posterior summary saved to {summary_path}")

    if diagnostics:
        diag_path = output_dir / f"fit_diagnostics_{model}_user.txt"
        diag_path.write_text(diagnostics)
        print(f"Sampler diagnostics saved to {diag_path}")

    nc_path = output_dir / f"fit_{model}_user.netcdf"
    try:
        idata.to_netcdf(str(nc_path))
        print(f"ArviZ InferenceData saved to {nc_path}")
    except (ValueError, ImportError) as exc:
        print(f"  WARNING: could not write {nc_path.name} ({exc}).\n"
              f"           Install a netCDF backend (pip install h5netcdf) if "
              f"you want it; {full_path.name} holds the same draws.")

    # ── 4. optional: the per-star latent parameters ─────────────────────────
    if save_latent:
        latent_vars = []
        for column in fit.column_names:
            if column.endswith("__"):
                continue
            base = column.split("[")[0]
            if base not in coeff_cols and base not in latent_vars:
                latent_vars.append(base)
        latent_path = output_dir / f"latent_parameters_{model}_user.csv.gz"
        latent = fit.draws_pd(vars=latent_vars)
        latent.to_csv(latent_path, index=False, compression="gzip")
        print(f"Latent per-star parameters saved to {latent_path} "
              f"({len(latent_vars)} variables, {latent.shape[1]} columns)")

    # Print coefficient summaries
    print(f"\nCoefficient medians (model {model}):")
    for col in coeff_cols:
        vals = coeffs_df[col].values
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        print(f"  {col:6s}: {p50:+.4f}  +{p84-p50:.4f}  -{p50-p16:.4f}")

    max_rhat = float(summary["r_hat"].max())
    n_div = int(audit["divergent__"].sum())
    print(f"\n  max R-hat = {max_rhat:.4f}   divergences = {n_div} / {len(audit)}")
    if max_rhat > 1.01 or n_div > 0:
        print("  WARNING: check fit_diagnostics before using these coefficients.")

    return coeff_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-derive granulation-radius scaling-relation coefficients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sample_csv",
        help="CSV with columns: name, radius, radius_err, teff, teff_err, feh, feh_err",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Root directory of granalyse results. Stars found as "
             "<results_dir>/<name>/model_<MODEL>_*/FitData_model_<MODEL>.csv",
    )
    parser.add_argument(
        "--model", default="H", choices=["H", "J", "T"],
        help="Granulation background model (H=two-component Harvey, "
             "J=hybrid, T=three-component).",
    )
    parser.add_argument(
        "--output-dir", default="calibration_output",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--mission", default="Kepler", choices=["Kepler", "TESS"],
        help="Photometric mission (sets bolometric correction).",
    )
    parser.add_argument(
        "--reference-json", default=None,
        help="JSON with the reference star used to normalise into relative "
             "space (same layout as granalyse/radius/relations/perky_params.json). "
             "Defaults to the shipped Perky (KIC 6106415) parameters. Whatever "
             "is used here must also be used when applying the relations.",
    )
    parser.add_argument(
        "--chains", type=int, default=4,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10000,
        help="Total MCMC iterations per chain (warmup=1/3, sampling=2/3).",
    )
    parser.add_argument(
        "--adapt-delta", type=float, default=0.95,
        help="Stan adapt_delta (target acceptance rate).",
    )
    parser.add_argument(
        "--max-treedepth", type=int, default=12,
        help="Stan max_treedepth.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="CmdStan RNG seed.",
    )
    parser.add_argument(
        "--refresh", type=int, default=None,
        help="Iterations between CmdStan progress reports.",
    )
    parser.add_argument(
        "--save-latent", action="store_true",
        help="Also write the 5N per-star latent parameters (lg_*_true, "
             "feh_true). Large: 5N of the 5N+6 sampled parameters.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Assemble and validate the Stan data dict, print it, then exit "
             "without sampling. Use this to check your data before committing "
             "to a potentially long Stan run.",
    )

    args = parser.parse_args(argv)

    # Load sample CSV
    print(f"Loading sample from {args.sample_csv}")
    sample_df = pd.read_csv(args.sample_csv)
    required_cols = {"name", "radius", "radius_err", "teff", "teff_err"}
    missing = required_cols - set(sample_df.columns)
    if missing:
        parser.error(f"sample_csv missing required columns: {missing}")

    # Ensure feh column exists (may be all NaN)
    if "feh" not in sample_df.columns:
        sample_df["feh"] = np.nan
    print(f"  {len(sample_df)} stars in sample")

    # Load the reference star used for relative-space normalisation
    reference = load_reference_params(args.reference_json, model=args.model)
    ref_name = reference.get("name", "Perky (KIC 6106415)")
    print(f"  Reference {ref_name}: Teff={reference['teff']} K  "
          f"R={reference['R_cm']/R_SUN_CM:.4f} R_sun  "
          f"[Fe/H]={reference['feh']}")
    print(f"  Model {args.model}: amp_ref={reference['models'][args.model]['amp']} ppm  "
          f"charb_ref={reference['models'][args.model]['charb']} uHz")

    # Build Stan data dict
    print(f"\nPreparing Stan data for model {args.model} ...")
    stan_data, df_rel, skipped = build_stan_data(
        sample_df, args.results_dir, args.model, args.mission, reference,
    )
    if skipped:
        print(f"\n{len(skipped)} stars skipped: {skipped}")

    N = stan_data["N"]
    print(f"\nStars in Stan data: N={N}")

    # Validate shapes and finiteness
    print("\nValidating Stan data dict ...")
    validate_stan_data(stan_data)
    print("  All keys present, shapes correct, values finite.")

    if args.dry_run:
        print("\n--- DRY RUN: Stan data dict summary ---")
        for key, val in stan_data.items():
            arr = np.asarray(val)
            print(f"  {key:16s}  shape={arr.shape}  "
                  f"dtype={arr.dtype}  "
                  f"range=[{arr.min():.4g}, {arr.max():.4g}]")
        print("\nDry run complete. Stan sampling skipped.")
        return 0

    # Run Stan
    fit, diagnostics = run_stan(
        stan_data, df_rel, args.output_dir, args.model,
        chains=args.chains,
        iterations=args.iterations,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        refresh=args.refresh,
        seed=args.seed,
    )

    # Save outputs
    save_outputs(fit, args.model, args.output_dir,
                 diagnostics=diagnostics, save_latent=args.save_latent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
