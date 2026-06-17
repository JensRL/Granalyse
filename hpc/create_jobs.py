"""Stage 2 (HPC workflow): generate a SLURM array job for the fit cases.

Two modes:

  Prep-on-node (recommended; smallest upload -- only config + raw PDS files):
    python hpc/create_jobs.py --config config.yaml --venv <activate> [options]
  Each work unit is one star (all its models run in one node process).

  Pickle (pre-built FitCases from prepare_fitcases.py):
    python hpc/create_jobs.py --fitcases Fitcases --venv <activate> [options]
  Each work unit is one pickle (one star x one model).

Pooling (to respect a maximum array size, default 10000):
  All models of a star always run together in one task. If there are still
  more stars than --max-array, several stars are pooled into each task (run
  sequentially on one node); the walltime is scaled up accordingly. Per-task
  walltime = --hours-per-fit x models-per-star x stars-per-task (override with
  --walltime).

Writes three files in the current directory:
    fitcase_list.txt   one work unit per line; the array index selects a slice
    run_node.sh        executable: stage to scratch -> run -> rsync results back
    dispatch_fits.job  sbatch script: SLURM directives + per-task dispatch

The fit runs entirely on node-local scratch (no shared-FS I/O during the fit);
only results are copied back. Scratch is purged when the job ends. Work units
whose results already exist are skipped.
"""

import argparse
import glob
import math
import os
import stat
import sys


# ---- run_node.sh templates (node-local staging) ---------------------------
# {runner}/{config}/{results_dir}/{models} are filled by str.format; ${{...}}
# survives as a literal ${...} shell variable. Each invocation handles one star
# (config) or one pickle, stages to scratch, runs, rsyncs results, and cleans
# its scratch so a pooled task does not accumulate data.

_NODE_SH_CONFIG = """#!/bin/bash
# Prep-on-node: stage one star's PDS to scratch, build + run all its models
# there, copy results back, then clear the star's scratch.
#   usage: run_node.sh <starname> <pds_path>
set -eo pipefail

starname="$1"
pds="$2"

scratchdir="/scratch/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$scratchdir/Results"

cp "$pds" "$scratchdir/"
cd "$scratchdir"
python "{runner}" --config "{config}" --star "$starname" --model "{models}" \\
    --pds-file "$scratchdir/$(basename "$pds")" --results-dir "$scratchdir/Results"

mkdir -p "$SLURM_SUBMIT_DIR/{results_dir}/$starname"
rsync -aqz "$scratchdir/Results/$starname/" "$SLURM_SUBMIT_DIR/{results_dir}/$starname/"
rm -rf "$scratchdir/$(basename "$pds")" "$scratchdir/Results/$starname"
"""

_NODE_SH_PICKLE = """#!/bin/bash
# Pickle mode: stage one pre-built FitCase to scratch, run it, copy results back.
#   usage: run_node.sh <fitcase.pkl>
set -eo pipefail

fitcase="$1"
starname=$(basename "$(dirname "$fitcase")")
modname=$(basename "$fitcase" .pkl)

scratchdir="/scratch/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$scratchdir/Fitcases/$starname" "$scratchdir/Results"

cp "$fitcase" "$scratchdir/Fitcases/$starname/"
cd "$scratchdir"
python "{runner}" "Fitcases/$starname/$modname.pkl" --results-dir "$scratchdir/Results"

mkdir -p "$SLURM_SUBMIT_DIR/{results_dir}/$starname"
rsync -aqz "$scratchdir/Results/$starname/" "$SLURM_SUBMIT_DIR/{results_dir}/$starname/"
rm -rf "$scratchdir/Fitcases/$starname" "$scratchdir/Results/$starname"
"""

# ---- per-task dispatch blocks (inside dispatch_fits.job) -------------------
# Each array task processes a slice of {units_per_task} consecutive list lines.
# Field 1 (star or pickle path) must be space-free; for config mode the PDS
# path (field 2, line remainder) may contain spaces.
_DISPATCH_CONFIG = """units_per_task={units_per_task}
start=$(( (SLURM_ARRAY_TASK_ID - 1) * units_per_task + 1 ))
end=$(( SLURM_ARRAY_TASK_ID * units_per_task ))
awk "NR >= $start && NR <= $end" {list_file} | while read -r starname pds; do
    [ -z "$starname" ] && continue
    echo "----- Star $starname -----"
    "$SLURM_SUBMIT_DIR/run_node.sh" "$starname" "$pds"
done
"""

_DISPATCH_PICKLE = """units_per_task={units_per_task}
start=$(( (SLURM_ARRAY_TASK_ID - 1) * units_per_task + 1 ))
end=$(( SLURM_ARRAY_TASK_ID * units_per_task ))
awk "NR >= $start && NR <= $end" {list_file} | while read -r fitcase; do
    [ -z "$fitcase" ] && continue
    echo "----- $fitcase -----"
    "$SLURM_SUBMIT_DIR/run_node.sh" "$fitcase"
done
"""


def _has_results(results_dir, star, modname):
    """True if FitResults for star/model already exist (skip on resume)."""
    return bool(glob.glob(os.path.join(results_dir, star,
                                       f"{modname}_*", f"FitResults_{modname}.csv")))


def _fmt_walltime(hours):
    """SLURM walltime string from a (possibly fractional) number of hours."""
    total = int(math.ceil(hours * 3600))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def _pending_config(args):
    """(units, node_sh, dispatch, fits_per_unit, models_str) for prep-on-node.

    One work unit per star; a star is skipped only when every model is done.
    """
    from granalyse.config import load_config
    cfg = load_config(args.config)
    modnames = [f"model_{m}" for m in cfg.models]
    units = []
    for star in cfg.stars:
        pds = os.path.abspath(star.pds_file)
        if all(_has_results(args.results_dir, star.name, mn) for mn in modnames):
            print(f"skipping {star.name} (all models have results)")
        else:
            units.append(f"{star.name} {pds}")
    return units, _NODE_SH_CONFIG, _DISPATCH_CONFIG, len(cfg.models), ",".join(cfg.models)


def _pending_pickles(args):
    """(units, node_sh, dispatch, fits_per_unit, models_str) for pickle mode."""
    fitcases = sorted(glob.glob(os.path.join(args.fitcases, "*", "model_*.pkl")))
    units = []
    for case in fitcases:
        star = os.path.basename(os.path.dirname(case))
        modname = os.path.basename(case).replace(".pkl", "")
        if _has_results(args.results_dir, star, modname):
            print(f"skipping {case} (results exist)")
        else:
            units.append(os.path.abspath(case))
    return units, _NODE_SH_PICKLE, _DISPATCH_PICKLE, 1, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Granalyse YAML config (prep-on-node mode)")
    src.add_argument("--fitcases",
                     help="Directory with pickled fit cases (pickle mode)")
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--job-name", default="granalyse_fits")
    parser.add_argument("--partition", default="normal",
                        help="Comma-separated SLURM partition(s)")
    parser.add_argument("--hours-per-fit", type=float, default=1.0,
                        help="Estimated walltime per single model fit, in hours "
                             "(used to size --walltime; default 1.0)")
    parser.add_argument("--walltime", default=None,
                        help="Override the auto-computed per-task walltime "
                             "(e.g. 12:00:00 or 1-00:00:00)")
    parser.add_argument("--max-array", type=int, default=10000,
                        help="Maximum SLURM array size; stars are pooled per task "
                             "until the array fits (default 10000)")
    parser.add_argument("--cpus", type=int, default=1,
                        help="CPUs per array task; one fit runs on a single core")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--max-parallel", type=int, default=256,
                        help="Max simultaneously running array tasks (SLURM array %% throttle)")
    parser.add_argument("--venv", required=True,
                        help="Path to the activate script of a venv with granalyse installed")
    args = parser.parse_args()

    units, node_sh, dispatch_block, fits_per_unit, models = (
        _pending_config(args) if args.config else _pending_pickles(args))

    if not units:
        print("Nothing to do: all fit cases have results.")
        sys.exit(0)

    list_file = os.path.abspath("fitcase_list.txt")
    with open(list_file, "w") as f:
        f.write("\n".join(units) + "\n")

    # Pool work units into tasks so the array stays within --max-array.
    n_units = len(units)
    units_per_task = max(1, math.ceil(n_units / args.max_array))
    n_tasks = math.ceil(n_units / units_per_task)
    limit = min(n_tasks, args.max_parallel)

    # Per-task walltime: hours_per_fit x fits-per-star x stars-per-task.
    fits_per_task = fits_per_unit * units_per_task
    walltime = args.walltime or _fmt_walltime(args.hours_per_fit * fits_per_task)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner = os.path.join(repo_dir, "hpc", "run_fitcase.py")
    config_abs = os.path.abspath(args.config) if args.config else ""

    node_sh = node_sh.format(runner=runner, config=config_abs,
                             results_dir=args.results_dir, models=models)
    with open("run_node.sh", "w") as f:
        f.write(node_sh)
    os.chmod("run_node.sh", os.stat("run_node.sh").st_mode | stat.S_IEXEC |
             stat.S_IXGRP | stat.S_IXOTH)

    dispatch_block = dispatch_block.format(list_file=list_file,
                                           units_per_task=units_per_task)

    job = f"""#!/bin/bash

#SBATCH --job-name={args.job_name}
#SBATCH --output=Outputs/%x_%A_%a.out
#SBATCH --time={walltime}
#SBATCH --partition={args.partition}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --array=1-{n_tasks}%{limit}
#SBATCH --export=NONE

echo "========= Job started at `date` on `hostname -s` =========="
export OMP_NUM_THREADS={args.cpus}

source "{args.venv}"
cd "$SLURM_SUBMIT_DIR"

{dispatch_block}
echo "========= Job finished at `date` =========="
"""

    os.makedirs("Outputs", exist_ok=True)
    with open("dispatch_fits.job", "w") as f:
        f.write(job)

    mode = "prep-on-node" if args.config else "pickle"
    star_word = "stars" if args.config else "pickles"
    print(f"\nWrote dispatch_fits.job + run_node.sh ({mode} mode).")
    print(f"  {n_units} {star_word} -> {n_tasks} array tasks "
          f"({units_per_task} per task x {fits_per_unit} fit(s) each), "
          f"max {limit} parallel.")
    print(f"  Walltime/task: {walltime}  (={args.hours_per_fit} h x {fits_per_task} fits)")
    print(f"Submit with:  sbatch dispatch_fits.job")


if __name__ == "__main__":
    main()
