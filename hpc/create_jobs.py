"""Stage 2 (HPC workflow): generate a SLURM array job for the prepared
fit cases.

    python hpc/create_jobs.py --fitcases Fitcases [SLURM options]

Workflow stages:
    Stage 1 — hpc/prepare_fitcases.py : prepare and pickle all FitCases locally
    Stage 2 — hpc/create_jobs.py      : generate dispatch_fits.job (this script)
    Stage 3 — sbatch dispatch_fits.job : each array task runs hpc/run_fitcase.py

Writes dispatch_fits.job (sbatch script) and fitcase_list.txt. Each array
task copies one pickle to node-local scratch, runs hpc/run_fitcase.py in
the activated environment, and rsyncs the results back. Fit cases that
already have results are skipped.
"""

import argparse
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitcases", default="Fitcases",
                        help="Directory with pickled fit cases (from prepare_fitcases.py)")
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--job-name", default="granalyse_fits")
    parser.add_argument("--partition", default="normal",
                        help="Comma-separated SLURM partition(s)")
    parser.add_argument("--walltime", default="15:00:00")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--max-parallel", type=int, default=64,
                        help="Max simultaneously running array tasks")
    parser.add_argument("--venv", required=True,
                        help="Path to the activate script of a venv with granalyse installed")
    parser.add_argument("--scratch", default="/scratch/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID",
                        help="Node-local scratch directory pattern")
    args = parser.parse_args()

    fitcases = sorted(glob.glob(os.path.join(args.fitcases, "*", "model_*.pkl")))
    pending = []
    for case in fitcases:
        star = os.path.basename(os.path.dirname(case))
        modname = os.path.basename(case).replace(".pkl", "")
        done = glob.glob(os.path.join(args.results_dir, star,
                                      f"{modname}_*", f"FitResults_{modname}.csv"))
        if done:
            print(f"skipping {case} (results exist)")
        else:
            pending.append(case)

    if not pending:
        print("Nothing to do: all fit cases have results.")
        sys.exit(0)

    with open("fitcase_list.txt", "w") as f:
        f.write("\n".join(pending) + "\n")

    n = len(pending)
    limit = min(n, args.max_parallel)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    job = f"""#!/bin/bash

#SBATCH --job-name={args.job_name}
#SBATCH --output=Outputs/%x_%A_%a.out
#SBATCH --time={args.walltime}
#SBATCH --partition={args.partition}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --array=1-{n}%{limit}
#SBATCH --export=NONE

echo "========= Job started at `date` on `hostname -s` =========="
export OMP_NUM_THREADS={args.cpus}

source "{args.venv}"

fitcase=$(awk "NR == ${{SLURM_ARRAY_TASK_ID}}" {os.path.abspath('fitcase_list.txt')})
starname=$(basename $(dirname "$fitcase"))
modname=$(basename "$fitcase" .pkl)
echo "Fit case: $starname / $modname"

scratchdir="{args.scratch}"
mkdir -p "$scratchdir"
cp "$fitcase" "$scratchdir/"
cd "$scratchdir"

python "{os.path.join(repo_dir, 'hpc', 'run_fitcase.py')}" "$(basename $fitcase)" --results-dir Results

mkdir -p "$SLURM_SUBMIT_DIR/{args.results_dir}/$starname"
rsync -aqz "Results/$starname/" "$SLURM_SUBMIT_DIR/{args.results_dir}/$starname/"
echo "========= Job finished at `date` =========="
"""

    os.makedirs("Outputs", exist_ok=True)
    with open("dispatch_fits.job", "w") as f:
        f.write(job)
    print(f"\nWrote dispatch_fits.job for {n} fit cases "
          f"(max {limit} parallel). Submit with:  sbatch dispatch_fits.job")


if __name__ == "__main__":
    main()
