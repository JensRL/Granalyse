"""Stage 1 (HPC workflow): prepare and pickle all FitCases locally.

Builds one pickled FitCase per star x model from a standard Granalyse run
configuration, ready for batch execution on a cluster:

    python hpc/prepare_fitcases.py config.yaml --outdir Fitcases

Workflow stages:
    Stage 1 — hpc/prepare_fitcases.py : prepare and pickle all FitCases (this script)
    Stage 2 — hpc/create_jobs.py      : generate dispatch_fits.job (sbatch script)
    Stage 3 — hpc/run_fitcase.py      : run one fit case (called by sbatch array job)
"""

import argparse
import os
import pickle

from granalyse.config import load_config
from granalyse.pipeline import prepare_fit_case, prepare_star_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Granalyse YAML run configuration")
    parser.add_argument("--outdir", default="Fitcases",
                        help="Directory for the pickled fit cases")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-prepare fit cases whose pickle already exists "
                             "(default: skip them, so a cancelled prep run can "
                             "be resumed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    count = 0
    skipped = 0
    for star in cfg.stars:
        star_dir = os.path.join(args.outdir, star.name)
        os.makedirs(star_dir, exist_ok=True)
        # Read and process the spectrum once per star, reused across its models.
        star_data = None
        for model_name in cfg.models:
            path = os.path.join(star_dir, f"model_{model_name}.pkl")
            if os.path.exists(path) and not args.overwrite:
                print(f"exists, skipping {path}")
                skipped += 1
                continue
            if star_data is None:
                star_data = prepare_star_data(star, cfg)
            fit = prepare_fit_case(star, model_name, cfg, star_data=star_data)
            with open(path, "wb") as f:
                pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"prepared {path}")
            count += 1
    print(f"\n{count} fit cases prepared ({skipped} already existed) in {args.outdir}/")


if __name__ == "__main__":
    main()
