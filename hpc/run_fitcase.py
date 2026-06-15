"""Stage 3 (HPC workflow): run one pickled FitCase (executed by the SLURM array job).

    python hpc/run_fitcase.py Fitcases/<star>/model_<X>.pkl [--results-dir Results]

Workflow stages:
    Stage 1 — hpc/prepare_fitcases.py : prepare and pickle all FitCases locally
    Stage 2 — hpc/create_jobs.py      : generate dispatch_fits.job (sbatch script)
    Stage 3 — hpc/run_fitcase.py      : run one fit case (this script, called by sbatch)

The result paths stored in the pickle are remapped under --results-dir so the
job can run from any working directory (e.g. cluster scratch space).
"""

import argparse
import os
import pickle

from granalyse.pipeline import run_single_fit


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fitcase", help="Path to a pickled FitCase")
    parser.add_argument("--results-dir", default="Results")
    args = parser.parse_args()

    with open(args.fitcase, "rb") as f:
        fit = pickle.load(f)
    fit = remap_paths(fit, args.results_dir)
    print(f"Running fit: star {fit.name}, model {fit.model}")
    run_single_fit(fit)
    print(f"Done: results in {fit.respath}")


if __name__ == "__main__":
    main()
