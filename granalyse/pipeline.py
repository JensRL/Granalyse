"""Per-star pipeline orchestration: data prep -> priors -> nested-sampling
fits for all chosen models -> radius inference (BMA) -> summary output."""

import contextlib
import json
import os
import sys

import numpy as np
import pandas as pd

from . import data as gdata
from .models import get_model
from .plotting import present_fit, prior_predictive_PDS, radius_posterior_plot
from .priors import set_priors
from .radius import infer_radius
from .results import save_run_output
from .sampling import run_fit
from .star import FitCase


class _Tee:
    """Duplicate a text stream into a log file (stdout tee for run logs)."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._log = logfile

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()


def _result_dirname(model_name, cfg):
    """Original result-directory naming: model_X_binned_gaussian_correlatedinf etc."""
    parts = [f"model_{model_name}",
             "binned" if cfg.bin_data else "unbinned",
             cfg.pulsation_model.lower()]
    if cfg.correlated_inference:
        parts.append("correlatedinf")
    return "_".join(parts)


def prepare_fit_case(star, model_name, cfg):
    """Build a fully prepared FitCase (data, parameters, priors, paths)."""
    model = get_model(model_name)
    kind = cfg.star_kinds.get(star.name, "star")

    numax = star.numax
    if numax is None:
        numax = gdata.calc_numax(star.logg, star.teff)
        print(f"{star.name}: numax = {numax:.1f} muHz from logg/teff scaling")

    fit = FitCase(name=star.name, model=model.name, seed=cfg.seed)
    fit.PDSfile = star.pds_file
    fit.numax = float(numax)
    fit.dnu = 0.26 * fit.numax**0.772
    fit.pulsmod = cfg.pulsation_model
    fit.corr_inference = cfg.correlated_inference
    fit.nlive_enhance = cfg.nlive_factor
    fit.bin_data = cfg.bin_data
    fit.teff = star.teff
    fit.teff_err = star.teff_err
    fit.feh = star.feh
    fit.feh_err = star.feh_err
    fit.mission = star.mission

    # Data preparation chain
    freq, power = gdata.read_data(fit)
    fit.nyquist = star.nyquist if star.nyquist is not None else freq[-1]
    if kind == "star":
        # Kepler/TESS instrumental systematics; not present in simulations
        power, _ = gdata.mask_systematics(freq, power, fit.numax)
    freq, power, _ = gdata.process_binning(freq, power, fit, cfg.bin_data,
                                           cfg.bin_resolution, cfg.bin_visuals)
    freq, power, _ = gdata.cut_low_frequencies(freq, power, fit,
                                               cfg.low_frequency_cut)
    fit.freq, fit.power = freq, power
    fit.eta = gdata.calc_eta(fit)

    # Parameters, initial guesses, priors
    model.configure_fit_parameters(fit)
    fit.inits = model.inits(fit.numax)
    fit.priors, fit.prior_samples = set_priors(fit, model)

    # Output paths (original naming scheme)
    respath = os.path.join(cfg.output_dir, star.name, _result_dirname(model.name, cfg))
    fit.respath = respath + os.sep
    fit.fitdatafile = os.path.join(respath, f"FitData_model_{model.name}.csv")
    fit.fitresultsfile = os.path.join(respath, f"FitResults_model_{model.name}.csv")
    fit.priorpredictivePDSfigname = os.path.join(respath, "PriorPredictivePDS.pdf")
    fit.PDSfigname = os.path.join(respath, "PDS_Fit.pdf")
    fit.Cornerfigname = os.path.join(respath, "CornerPlot.pdf")
    fit.priorCornerfigname = os.path.join(respath, "PriorCornerPlot.pdf")
    fit.PriorPostCornerfigname = os.path.join(respath, "PriorPostCornerPlot.pdf")
    return fit


def run_single_fit(fit, model=None, make_plots=True, skip_replot=False):
    """Fit one prepared FitCase (or load a previous run from disk) and write
    all result files and figures. Returns (fit, evidence, evidence_err).

    With skip_replot, figures are not regenerated for fits that already exist
    on disk (the fast resume path); a freshly run fit is always plotted.
    """
    if model is None:
        model = get_model(fit.model)
    os.makedirs(fit.respath, exist_ok=True)

    existing = os.path.exists(fit.fitdatafile)
    # Only suppress plotting when reusing an existing fit; new fits still plot.
    do_plots = make_plots and not (existing and skip_replot)
    if existing and skip_replot:
        print(f"{fit.name} / model {model.name}: skip_replot -- reusing existing figures")

    if do_plots:
        prior_predictive_PDS(fit, model)

    evidence_file = os.path.join(fit.respath, f"Evidence_model_{model.name}.txt")
    if existing:
        print(f"{fit.name} / model {model.name}: existing fit found, loading results")
        fit.fitdata = pd.read_csv(fit.fitdatafile)
        if not os.path.exists(evidence_file):
            raise RuntimeError(
                f"FitData found for {fit.name} / model {model.name} but evidence file "
                f"is missing (partial/corrupt run): {evidence_file}"
            )
        evidence, evidence_err = np.loadtxt(evidence_file)
        from .results import compute_normalised_amplitudes, save_fit_params
        fit.fitresults = save_fit_params(fit, model=model)
        fit = compute_normalised_amplitudes(fit, model=model)
        fit.fitresults.to_csv(fit.fitresultsfile, index=False, header=True)
    else:
        print(f"{fit.name} / model {model.name}: starting nested sampling...")
        samples, cov, evidence, evidence_err = run_fit(fit, model)
        fit = save_run_output(fit, samples, cov, evidence, evidence_err, model)

    if do_plots:
        present_fit(fit, model)
    return fit, evidence, evidence_err


def run_star(star, cfg):
    """Run all configured models for one star, then radius inference.
    Returns a summary dict (also written to the star's output directory).
    Terminal output is mirrored to <star_dir>/run.log."""
    star_dir = os.path.join(cfg.output_dir, star.name)
    os.makedirs(star_dir, exist_ok=True)

    with open(os.path.join(star_dir, "run.log"), "w") as logfile, \
            contextlib.redirect_stdout(_Tee(sys.stdout, logfile)):
        return _run_star_logged(star, cfg, star_dir)


def _run_star_logged(star, cfg, star_dir):
    print("=" * 70)
    print(f"  STAR: {star.name}")
    print("=" * 70)

    posteriors = {}
    evidences = {}
    summary = {"star": star.name, "models": {}}

    for model_name in cfg.models:
        model = get_model(model_name)
        print("-" * 70)
        print(f"  {star.name}  |  model {model.name}")
        print("-" * 70)
        fit = prepare_fit_case(star, model_name, cfg)
        fit, evidence, evidence_err = run_single_fit(
            fit, model, skip_replot=cfg.skip_replot)
        posteriors[model.name] = fit.fitdata
        evidences[model.name] = (evidence, evidence_err)
        med = {p: float(fit.fitresults[p][0]) for p in fit.fitresults.columns}
        summary["models"][model.name] = {
            "log_evidence": float(evidence),
            "log_evidence_err": float(evidence_err),
            "medians": med,
        }

    # Evidence summary and preferred model
    print("-" * 70)
    print(f"  Evidence summary for {star.name}")
    ranked = sorted(evidences.items(), key=lambda kv: kv[1][0], reverse=True)
    for m, (ev, err) in ranked:
        print(f"    model {m}: logZ = {ev:.3f} +/- {err:.3f}")
    preferred, (best_ev, _) = ranked[0]
    if len(ranked) > 1:
        dlogz = best_ev - ranked[1][1][0]
        print(f"  Preferred model: {preferred} (dlogZ = {dlogz:.2f} over {ranked[1][0]})")
    else:
        print(f"  Preferred model: {preferred}")
    summary["preferred_model"] = preferred

    if cfg.radius_inference:
        rel_models = {m: posteriors[m] for m in posteriors
                      if get_model(m).has_radius_relation}
        if rel_models:
            print("-" * 70)
            print(f"  Radius inference for {star.name}")
            radius = infer_radius(
                star.name,
                posteriors=rel_models,
                evidences={m: evidences[m] for m in rel_models},
                teff=star.teff, teff_err=star.teff_err,
                feh=star.feh, feh_err=star.feh_err,
                mission=star.mission,
            )
            summary["radius"] = {
                "R_combined": radius["R_combined"],
                "R_16": radius["R_16_combined"],
                "R_84": radius["R_84_combined"],
                "weights": radius["weights"],
                "per_model": {m: {k: r[k] for k in
                                  ("R_median", "R_16", "R_84", "log_evidence")}
                              for m, r in radius["model_results"].items()},
            }
            # Diagnostics: per-model + combined posterior samples and figure
            np.savez_compressed(
                os.path.join(star_dir, "radius_samples.npz"),
                combined=radius["combined_samples"],
                **{m: r["samples"] for m, r in radius["model_results"].items()},
            )
            radius_fig = os.path.join(star_dir, "RadiusPosteriors.pdf")
            if cfg.skip_replot and os.path.exists(radius_fig):
                print(f"{star.name}: skip_replot -- reusing existing RadiusPosteriors.pdf")
            else:
                radius_posterior_plot(radius, radius_fig)
        else:
            print(f"{star.name}: no fitted model has a radius relation; "
                  "skipping radius inference.")

    with open(os.path.join(star_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 70)
    return summary


def run_pipeline(cfg):
    """Run the full pipeline for all stars in the configuration; writes a
    per-star summary.json and an overview CSV in the output directory."""
    rows = []
    summaries = []
    for star in cfg.stars:
        summary = run_star(star, cfg)
        summaries.append(summary)
        row = {"star": star.name}
        for m, res in summary["models"].items():
            row[f"logZ_{m}"] = res["log_evidence"]
        if "radius" in summary:
            row["R_combined"] = summary["radius"]["R_combined"]
            row["R_16"] = summary["radius"]["R_16"]
            row["R_84"] = summary["radius"]["R_84"]
        rows.append(row)

    os.makedirs(cfg.output_dir, exist_ok=True)
    overview = pd.DataFrame(rows)
    overview.to_csv(os.path.join(cfg.output_dir, "overview.csv"), index=False)
    print("\nPipeline complete. Overview:")
    print(overview.to_string(index=False))
    return summaries
