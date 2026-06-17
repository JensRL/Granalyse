#!/usr/bin/env python3
"""derive_relations.py
======================
Re-derive granulation-radius scaling-relation coefficients from a user
sample of stars with known radii.

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
        Full MCMC draws (cmdstanpy output).

    <output-dir>/fit_<MODEL>_user.netcdf
        ArviZ InferenceData object.
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bolometric-correction parameters per mission (matching granalyse/apply.py)
# ---------------------------------------------------------------------------
_BOLOMETRIC = {
    "Kepler": (5934.0, 0.80),
    "TESS":   (4714.0, 0.81),
}

R_SUN_CM = 6.957e10  # cm


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
# Per-star data preparation  (replicates calib_data + prepare_data_for_stan)
# ---------------------------------------------------------------------------
def _find_posterior_file(results_dir, star_name, model):
    """Glob for FitData_model_<model>.csv[.gz] under results_dir/<star_name>/model_<model>_*/.

    Prefers the gzip-compressed file written by the current pipeline and falls
    back to a legacy uncompressed .csv; pd.read_csv reads either transparently.
    """
    base = os.path.join(
        results_dir, star_name,
        f"model_{model}_*",
        f"FitData_model_{model}.csv",
    )
    matches = sorted(glob.glob(base + ".gz")) or sorted(glob.glob(base))
    return matches[0] if matches else None


def _compute_per_star_stats(charb_s, agran_s):
    """Compute log10 medians and 2x2 covariance from posterior samples.

    Replicates the approach in application_helpers.calib_data (lines 306-307):
        lg_c = np.log10(charb_s); lg_a = np.log10(agran_s)
        cov_m = np.cov(np.vstack([lg_a, lg_c]))

    Row/column order in cov_m is [lg_amp, lg_charb] (amp first, charb second),
    matching the Stan data layout lg_amp_obs / lg_charb_obs.

    Returns
    -------
    lg_amp_mean   : float   median of log10(A_gran) samples
    lg_charb_mean : float   median of log10(charb) samples
    cov_2x2       : ndarray shape (2, 2)   [[var_amp, cov], [cov, var_charb]]
    """
    lg_a = np.log10(agran_s)
    lg_c = np.log10(charb_s)
    # np.cov stacks as [lg_a, lg_c] -> row 0 = lg_amp, row 1 = lg_charb
    cov_m = np.cov(np.vstack([lg_a, lg_c]))
    return float(np.median(lg_a)), float(np.median(lg_c)), cov_m


# ---------------------------------------------------------------------------
# Stan data assembly  (replicates prepare_data_for_stan)
# ---------------------------------------------------------------------------
def build_stan_data(sample_df, results_dir, model, mission, perky):
    """Load posteriors, transform, normalise to Perky, and assemble Stan dict.

    Parameters
    ----------
    sample_df   : pd.DataFrame  (from sample.csv)
    results_dir : str or Path
    model       : 'H', 'J', or 'T'
    mission     : 'Kepler' or 'TESS'
    perky       : dict from granalyse.radius.load_perky_params()

    Returns
    -------
    stan_data : dict  (ready for CmdStanModel.sample)
    df_rel    : pd.DataFrame  (relative log-space values used for Stan)
    skipped   : list[str]   stars that could not be processed
    """
    perky_amp   = perky["models"][model]["amp"]
    perky_charb = perky["models"][model]["charb"]
    perky_teff  = perky["teff"]
    perky_R_cm  = perky["R_cm"]
    perky_feh   = perky["feh"]

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
        # Matches pystan_helpers.prepare_data_for_stan lines 313-325:
        #   R [R_sun] -> [cm] -> divide by perky_R_cm
        #   Teff divide by perky_teff
        #   amp_tot divide by perky_amp   (already carried via lg_amp_mean)
        #   charb divide by perky_charb   (already carried via lg_charb_mean)
        #   FeH subtract perky_feh

        R_cm   = float(row["radius"])    * R_SUN_CM
        R_cm_e = float(row["radius_err"]) * R_SUN_CM

        R_rel     = R_cm   / perky_R_cm
        R_rel_err = R_cm_e / perky_R_cm

        teff_rel     = float(row["teff"])     / perky_teff
        teff_err_rel = float(row["teff_err"]) / perky_teff

        # log-space observables (relative) for gran amplitude and charb
        # The per-star posterior median is already in absolute units [ppm, muHz];
        # dividing before log10 is equivalent to subtracting log10(perky_*):
        #   lg_amp_rel = log10(amp/perky_amp) = lg_amp_mean - log10(perky_amp)
        lg_amp_rel   = lg_amp_mean   - np.log10(perky_amp)
        lg_charb_rel = lg_charb_mean - np.log10(perky_charb)

        # The 2x2 covariance is in absolute log10 space; dividing by a constant
        # shifts the mean but does NOT change the variance (log10(x/k) = log10(x) - log10(k)).
        # So cov_m is already the correct covariance in relative log space.

        lg_R_rel     = np.log10(R_rel)
        lg_R_rel_err = R_rel_err / (R_rel * np.log(10))

        lg_teff_rel     = np.log10(teff_rel)
        lg_teff_rel_err = teff_err_rel / (teff_rel * np.log(10))

        feh_raw = row.get("feh", np.nan)
        has_feh = 0
        feh_val = 0.0
        if not pd.isna(feh_raw):
            try:
                feh_val = float(feh_raw) - perky_feh
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
def _get_initial_values(df_rel):
    """Deterministic starting point matching pystan_helpers.get_initial_values."""
    return {
        "lg_teff_true": df_rel["lg_teff_obs"].values.tolist(),
        "lg_R_true":    df_rel["lg_R_obs"].values.tolist(),
        "lg_amp_true":  df_rel["lg_amp_obs"].values.tolist(),
        "lg_charb_true":df_rel["lg_charb_obs"].values.tolist(),
        "feh_true":     df_rel["feh_obs"].values.tolist(),
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
             adapt_delta=0.95, max_treedepth=12):
    """Compile and sample the Stan model; return (fit_data, az_data).

    Follows pystan_helpers.run_stan_fit exactly:
      - iter_warmup  = iterations // 3
      - iter_sampling = 2 * iterations // 3
      - per-chain init with 5% random jitter on scalar coefficients (rng seed 42)
    """
    try:
        import arviz as az
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
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )

    fit_data = fit.draws_pd()
    az_data  = az.from_cmdstanpy(fit)
    print("\n" + fit.diagnose())

    return fit_data, az_data


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(fit_data, az_data, model, output_dir):
    """Save coefficient draws, full fit CSV, and netCDF.

    The coefficient file relation_coefficients_<MODEL>_seeduser.csv has columns
    fac, x, y, z, z2, sigma — directly loadable by
    granalyse.radius.load_relation_coefficients(seed='user').

    The file written here uses '_user' suffix. To use it directly with the
    shipped loader you would need to rename or copy it into the relations/
    package directory — see README.md for details.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_cols = ["fac", "x", "y", "z", "z2", "sigma"]
    coeffs_df = fit_data[coeff_cols].copy()
    coeff_path = output_dir / f"relation_coefficients_{model}_seeduser.csv"
    coeffs_df.to_csv(coeff_path, index=False)
    print(f"\nCoefficient draws saved to {coeff_path}")

    full_path = output_dir / f"fit_data_{model}_user.csv"
    fit_data.to_csv(full_path, index=False)
    print(f"Full fit draws saved to {full_path}")

    nc_path = output_dir / f"fit_{model}_user.netcdf"
    az_data.to_netcdf(str(nc_path))
    print(f"ArviZ InferenceData saved to {nc_path}")

    # Print coefficient summaries
    print(f"\nCoefficient medians (model {model}):")
    for col in coeff_cols:
        vals = coeffs_df[col].values
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        print(f"  {col:6s}: {p50:+.4f}  +{p84-p50:.4f}  -{p50-p16:.4f}")

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

    # Load Perky reference parameters
    from granalyse.radius import load_perky_params
    perky = load_perky_params()
    print(f"  Perky (KIC 6106415): Teff={perky['teff']} K  "
          f"R={perky['R_cm']/R_SUN_CM:.4f} R_sun  "
          f"[Fe/H]={perky['feh']}")
    print(f"  Model {args.model}: amp_perky={perky['models'][args.model]['amp']} ppm  "
          f"charb_perky={perky['models'][args.model]['charb']} uHz")

    # Build Stan data dict
    print(f"\nPreparing Stan data for model {args.model} ...")
    stan_data, df_rel, skipped = build_stan_data(
        sample_df, args.results_dir, args.model, args.mission, perky,
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
    fit_data, az_data = run_stan(
        stan_data, df_rel, args.output_dir, args.model,
        chains=args.chains,
        iterations=args.iterations,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
    )

    # Save outputs
    save_outputs(fit_data, az_data, args.model, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
