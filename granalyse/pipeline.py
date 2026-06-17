"""Per-star pipeline orchestration: data prep -> priors -> nested-sampling
fits for all chosen models -> radius inference (BMA) -> summary output."""

import contextlib
import copy
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
from .results import compute_normalised_amplitudes, save_fit_params, save_run_output
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


def _star_numax(star):
    """numax given directly, or estimated from logg/teff."""
    if star.numax is not None:
        return float(star.numax)
    return float(gdata.calc_numax(star.logg, star.teff))


def _assign_paths(fit, star_name, model, cfg):
    """Set all output paths on a FitCase (original naming scheme). FitData is
    written gzip-compressed; the .csv.gz extension drives the compression."""
    respath = os.path.join(cfg.output_dir, star_name, _result_dirname(model.name, cfg))
    fit.respath = respath + os.sep
    fit.fitdatafile = os.path.join(respath, f"FitData_model_{model.name}.csv.gz")
    fit.fitresultsfile = os.path.join(respath, f"FitResults_model_{model.name}.csv")
    fit.priorpredictivePDSfigname = os.path.join(respath, "PriorPredictivePDS.pdf")
    fit.PDSfigname = os.path.join(respath, "PDS_Fit.pdf")
    fit.Cornerfigname = os.path.join(respath, "CornerPlot.pdf")
    fit.priorCornerfigname = os.path.join(respath, "PriorCornerPlot.pdf")
    fit.PriorPostCornerfigname = os.path.join(respath, "PriorPostCornerPlot.pdf")


def _resolve_fitdatafile(fit):
    """Point fit.fitdatafile at an existing FitData file, preferring the
    compressed name but falling back to a legacy uncompressed .csv from older
    runs. Returns True if a FitData file exists."""
    if not os.path.exists(fit.fitdatafile) and fit.fitdatafile.endswith(".csv.gz"):
        legacy = fit.fitdatafile[:-3]
        if os.path.exists(legacy):
            fit.fitdatafile = legacy
    return os.path.exists(fit.fitdatafile)


def prepare_star_data(star, cfg):
    """Prepare the star-level data products shared by every model: read the
    PDS, mask systematics, bin, cut low frequencies and compute eta.

    Returns a FitCase carrying the processed spectrum and stellar metadata,
    with no model-specific parameters, priors or output paths set. Reused
    across a star's models so the spectrum is read and processed only once.
    """
    kind = cfg.star_kinds.get(star.name, "star")
    numax = star.numax
    if numax is None:
        numax = gdata.calc_numax(star.logg, star.teff)
        print(f"{star.name}: numax = {numax:.1f} muHz from logg/teff scaling")

    fit = FitCase(name=star.name, model="", seed=cfg.seed)
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

    # Data preparation chain (identical across models)
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
    return fit


def prepare_fit_case(star, model_name, cfg, star_data=None):
    """Build a fully prepared FitCase (data, parameters, priors, paths).

    If star_data (a FitCase from prepare_star_data) is given, the prepared
    spectrum is reused, avoiding a redundant read/bin per model; otherwise the
    star-level data is prepared here. prior_samples are left unset and drawn
    lazily at plot time (keeps prepared pickles small).
    """
    model = get_model(model_name)
    if star_data is None:
        star_data = prepare_star_data(star, cfg)
    fit = copy.deepcopy(star_data)
    fit.model = model.name

    # Parameters, initial guesses, priors
    model.configure_fit_parameters(fit)
    fit.inits = model.inits(fit.numax)
    fit.priors = set_priors(fit, model)
    fit.prior_samples = None  # drawn lazily in plotting (saves prep time + pickle size)

    _assign_paths(fit, star.name, model, cfg)
    return fit


def prepare_fit_case_light(star, model_name, cfg):
    """Build a FitCase with metadata, fit-parameter configuration and output
    paths but WITHOUT reading or preparing the spectrum.

    Sufficient to detect and finalise an already-completed fit (load posteriors,
    recompute the result table) without the cost of reading/binning the PDS;
    not enough to run a new fit (no priors/data). Used by the skip_replot fast
    path. fit.fitdatafile is resolved to an existing file if one is present.
    """
    model = get_model(model_name)
    fit = FitCase(name=star.name, model=model.name, seed=cfg.seed)
    fit.numax = _star_numax(star)
    fit.dnu = 0.26 * fit.numax**0.772
    fit.pulsmod = cfg.pulsation_model
    fit.corr_inference = cfg.correlated_inference
    fit.bin_data = cfg.bin_data
    fit.teff = star.teff
    fit.teff_err = star.teff_err
    fit.feh = star.feh
    fit.feh_err = star.feh_err
    fit.mission = star.mission

    model.configure_fit_parameters(fit)
    _assign_paths(fit, star.name, model, cfg)
    _resolve_fitdatafile(fit)
    return fit


def finalize_existing_fit(fit, model):
    """Load a previously completed fit from disk and (re)compute its result
    table without re-running the sampler or preparing the spectrum.

    Requires only fit.fitparams (from configure_fit_parameters) and the output
    paths; the granulation normalisation reads fit.fitresults, never the raw
    data. Returns (fit, evidence, evidence_err).
    """
    evidence_file = os.path.join(fit.respath, f"Evidence_model_{model.name}.txt")
    fit.fitdata = pd.read_csv(fit.fitdatafile)
    if not os.path.exists(evidence_file):
        raise RuntimeError(
            f"FitData found for {fit.name} / model {model.name} but evidence file "
            f"is missing (partial/corrupt run): {evidence_file}"
        )
    evidence, evidence_err = np.loadtxt(evidence_file)
    fit.fitresults = save_fit_params(fit, model=model)
    fit = compute_normalised_amplitudes(fit, model=model)
    fit.fitresults.to_csv(fit.fitresultsfile, index=False, header=True)
    return fit, evidence, evidence_err


def run_single_fit(fit, model=None, make_plots=True, skip_replot=False):
    """Fit one prepared FitCase (or load a previous run from disk) and write
    all result files and figures. Returns (fit, evidence, evidence_err).

    With skip_replot, figures are not regenerated for fits that already exist
    on disk (the fast resume path); a freshly run fit is always plotted.
    """
    if model is None:
        model = get_model(fit.model)
    os.makedirs(fit.respath, exist_ok=True)

    # Resume from an existing run; fall back to a legacy uncompressed FitData
    # file so older result directories remain resumable (pandas reads both).
    existing = _resolve_fitdatafile(fit)
    # Only suppress plotting when reusing an existing fit; new fits still plot.
    do_plots = make_plots and not (existing and skip_replot)
    if existing and skip_replot:
        print(f"{fit.name} / model {model.name}: skip_replot -- reusing existing figures")

    if do_plots:
        prior_predictive_PDS(fit, model)

    if existing:
        print(f"{fit.name} / model {model.name}: existing fit found, loading results")
        fit, evidence, evidence_err = finalize_existing_fit(fit, model)
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

    # The spectrum is read/binned once per star and shared across its models;
    # prepared lazily so the skip_replot fast path can avoid it entirely.
    star_data = None
    for model_name in cfg.models:
        model = get_model(model_name)
        print("-" * 70)
        print(f"  {star.name}  |  model {model.name}")
        print("-" * 70)

        fit = None
        if cfg.skip_replot:
            # Fast path: finalise an already-completed fit without reading the
            # spectrum. Falls through to a full run if no results exist yet.
            light = prepare_fit_case_light(star, model_name, cfg)
            if os.path.exists(light.fitdatafile):
                print(f"{star.name} / model {model.name}: skip_replot -- existing "
                      "results, finalising without spectrum preparation")
                os.makedirs(light.respath, exist_ok=True)
                fit, evidence, evidence_err = finalize_existing_fit(light, model)

        if fit is None:
            if star_data is None:
                star_data = prepare_star_data(star, cfg)
            fit = prepare_fit_case(star, model_name, cfg, star_data=star_data)
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
