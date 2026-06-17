"""Stage 2 (HPC workflow): run one star's models (executed by the SLURM array job).

Builds the FitCase(s) on the node from the config and the star's PDS, so only
the config and raw PDS files live on the cluster. One or several models can be
run for the star in a single process (the spectrum is prepared once and reused):

    python hpc/run_fitcase.py --config config.yaml --star <name> --model H,J,T \\
        [--pds-file <staged.dat>] [--results-dir Results]

The result paths are remapped under --results-dir so the job can run from any
working directory (e.g. cluster scratch space).
"""

import argparse
import os


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
        raise SystemExit("--model is required (e.g. 'H' or 'H,J,T')")

    # Read and bin the spectrum once, reuse it across this star's models.
    star_data = prepare_star_data(star, cfg)
    for model_name in models:
        fit = prepare_fit_case(star, model_name, cfg, star_data=star_data)
        fit = remap_paths(fit, args.results_dir)
        print(f"Running fit: star {fit.name}, model {fit.model}")
        run_single_fit(fit, skip_replot=cfg.skip_replot)
        print(f"Done: results in {fit.respath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Granalyse YAML config")
    parser.add_argument("--star", required=True, help="Star name to fit")
    parser.add_argument("--model", required=True,
                        help="Model letter(s), comma-separated (e.g. 'H' or 'H,J,T')")
    parser.add_argument("--pds-file", help="Override the star's PDS path "
                        "(e.g. a node-local staged copy)")
    parser.add_argument("--results-dir", default="Results")
    args = parser.parse_args()

    _run_config(args)


if __name__ == "__main__":
    main()
