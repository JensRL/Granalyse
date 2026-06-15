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
from granalyse.pipeline import prepare_fit_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Granalyse YAML run configuration")
    parser.add_argument("--outdir", default="Fitcases",
                        help="Directory for the pickled fit cases")
    args = parser.parse_args()

    cfg = load_config(args.config)
    count = 0
    for star in cfg.stars:
        star_dir = os.path.join(args.outdir, star.name)
        os.makedirs(star_dir, exist_ok=True)
        for model_name in cfg.models:
            fit = prepare_fit_case(star, model_name, cfg)
            path = os.path.join(star_dir, f"model_{model_name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"prepared {path}")
            count += 1
    print(f"\n{count} fit cases prepared in {args.outdir}/")


if __name__ == "__main__":
    main()
