"""Stage 3 (HPC workflow): run one work unit (executed by the SLURM array job).

Two modes:

  Prep-on-node (recommended): build the FitCase(s) on the node from the config
  and the star's PDS, so only the config and raw PDS files live on the cluster.
  One or several models can be run for the star in a single process (the
  spectrum is prepared once and reused):
    python hpc/run_fitcase.py --config config.yaml --star <name> --model H,J,T \\
        [--pds-file <staged.dat>] [--results-dir Results]

  Pickle (pre-built FitCase, one star x one model):
    python hpc/run_fitcase.py Fitcases/<star>/model_<X>.pkl [--results-dir Results]

The result paths are remapped under --results-dir so the job can run from any
working directory (e.g. cluster scratch space).
"""

import argparse
import os
import pickle


def remap_paths(fit, results_dir):
    safe_name = os.path.basename(os.path.normpath(fit.name))
    respath = os.path.join(results_dir, safe_name,
                           os.path.basename(os.path.normpath(fit.respath)))
    fit.respath = respath + os.sep
    for attr in ("fitdatafile", "fitresultsfile", "priorpredictivePDSfigname",
                 "PDSfigname", "Cornerfigname", "priorCornerfigname",
                 "PriorPostCornerfigname"):
        old = getattr(fit, attr)
        if old:
            setattr(fit, attr, os.path.join(respath, os.path.basename(old)))
    return fit


def _run_config(args):
    """Prep on the node and run every requested model for one star."""
    from granalyse.config import load_config
    from granalyse.pipeline import prepare_star_data, prepare_fit_case, run_single_fit

    cfg = load_config(args.config)
    stars = {s.name: s for s in cfg.stars}
    if args.star not in stars:
        raise SystemExit(f"Star '{args.star}' not found in {args.config}")
    star = stars[args.star]
    if args.pds_file:
        star.pds_file = args.pds_file  # read the node-local staged copy

    models = [m.strip() for m in (args.model or "").split(",") if m.strip()]
    if not models:
        raise SystemExit("config mode requires --model (e.g. 'H' or 'H,J,T')")

    # Read and bin the spectrum once, reuse it across this star's models.
    star_data = prepare_star_data(star, cfg)
    for model_name in models:
        fit = prepare_fit_case(star, model_name, cfg, star_data=star_data)
        fit = remap_paths(fit, args.results_dir)
        print(f"Running fit: star {fit.name}, model {fit.model}")
        run_single_fit(fit, skip_replot=cfg.skip_replot)
        print(f"Done: results in {fit.respath}")


def _run_pickle(args):
    """Run one pre-built FitCase pickle."""
    from granalyse.pipeline import run_single_fit
    with open(args.fitcase, "rb") as f:
        fit = pickle.load(f)
    fit = remap_paths(fit, args.results_dir)
    print(f"Running fit: star {fit.name}, model {fit.model}")
    run_single_fit(fit)
    print(f"Done: results in {fit.respath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fitcase", nargs="?",
                        help="Path to a pickled FitCase (pickle mode)")
    parser.add_argument("--config", help="Granalyse YAML config (prep-on-node mode)")
    parser.add_argument("--star", help="Star name to fit (prep-on-node mode)")
    parser.add_argument("--model", help="Model letter(s), comma-separated "
                        "(prep-on-node mode; e.g. 'H' or 'H,J,T')")
    parser.add_argument("--pds-file", help="Override the star's PDS path "
                        "(prep-on-node mode; e.g. a node-local staged copy)")
    parser.add_argument("--results-dir", default="Results")
    args = parser.parse_args()

    if args.config:
        if not args.star:
            raise SystemExit("config mode requires --star")
        _run_config(args)
    elif args.fitcase:
        _run_pickle(args)
    else:
        raise SystemExit("provide a pickle path, or --config with --star/--model")


if __name__ == "__main__":
    main()
