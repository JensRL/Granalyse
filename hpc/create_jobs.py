"""
Usage:

    python hpc/create_jobs.py --config config.yaml --hours-per-fit 3

Cluster defaults (partition, memory, etc.) are the constants below
for the generated ``dispatch_fits.job``.

Already-finished fits are detected and skipped: finished stars are dropped and
partly-finished stars keep only their missing models, so re-running regenerates
a job for exactly the work left.

Writes four files in the current directory:
    star_list.txt          one star per line: name <tab> models <tab> pds path
    granalyse_src.tar.gz   the granalyse package + runner, staged to scratch
    run_node.sh            executable: stage to scratch -> fit -> rsync back
    dispatch_fits.job      sbatch script: SLURM directives + per-task dispatch
"""

import argparse
import glob
import math
import os
import stat
import sys
import tarfile

# ---- cluster defaults (edit here, or tweak the generated dispatch_fits.job) -
RESULTS_DIR = "Results"
JOB_NAME = "granalyse_fits"
PARTITION = "q24,q28,q36,q40"        # comma-separated SLURM partition(s)
MAX_ARRAY = 10000           # cap on array size; stars are pooled per task to fit
MAX_PARALLEL = 256          # max simultaneously running array tasks
CPUS = 1                    # one fit runs on a single core
MEM_PER_CPU = "8G"


# ---- run_node.sh template  ----------------------------
_NODE_SH = """#!/bin/bash
set -eo pipefail

starname="$1"
models="$2"
pds="$3"

scratchdir="/scratch/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$scratchdir/Results"
cp "$pds" "$scratchdir/"
cd "$scratchdir"

dest="$SLURM_SUBMIT_DIR/{results_dir}/$starname"
mkdir -p "$dest"

IFS=',' read -ra modellist <<< "$models"
for model in "${{modellist[@]}}"; do
    python "$GRAN_RUNNER" --config "{config}" --star "$starname" --model "$model" \\
        --pds-file "$scratchdir/$(basename "$pds")" --results-dir "$scratchdir/Results"
    rsync -aqz "$scratchdir/Results/$starname/" "$dest/"
done
"""

# ---- per-task dispatch block (inside dispatch_fits.job) --------------------
# Each array task processes a slice of {units_per_task} consecutive list lines.
_DISPATCH = """units_per_task={units_per_task}
start=$(( (SLURM_ARRAY_TASK_ID - 1) * units_per_task + 1 ))
end=$(( SLURM_ARRAY_TASK_ID * units_per_task ))
awk -F'\\t' "NR >= $start && NR <= $end" {list_file} | while IFS=$'\\t' read -r starname models pds; do
    [ -z "$starname" ] && continue
    echo "----- Star $starname (models: $models) -----"
    "$SLURM_SUBMIT_DIR/run_node.sh" "$starname" "$models" "$pds"
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


def _scan(config, results_dir):
    """Inspect existing results for every star/model in the config.

    Returns ``(units, max_models, status)`` where:
      * ``units``  -- tab-separated 'name<TAB>models<TAB>pds' lines for the stars
        that still have work left.
      * ``max_models`` -- largest number of pending models on any one star
        (drives the per-task walltime).
      * ``status`` -- list of ``(name, n_models, done_list, pending_list)`` for
        every star, used to print the diagnostic summary.
    """
    from granalyse.config import load_config
    cfg = load_config(config)
    units = []
    status = []
    for star in cfg.stars:
        done, pending = [], []
        for m in cfg.models:
            (done if _has_results(results_dir, star.name, f"model_{m}")
             else pending).append(m)
        status.append((star.name, len(cfg.models), done, pending))
        if pending:
            pds = os.path.abspath(star.pds_file)
            units.append("\t".join((star.name, ",".join(pending), pds)))
    max_models = max((len(s[3]) for s in status), default=0)
    return units, max_models, status


def _print_diagnostics(status, max_list=20):
    """Print a succeeded/missing summary of existing results.

    Returns ``(done_fits, pending_fits)`` so the caller can tell a fresh run
    (no results yet) from a re-run over a directory that already has some.
    """
    n_stars = len(status)
    total_fits = sum(s[1] for s in status)
    done_fits = sum(len(s[2]) for s in status)
    pending_fits = sum(len(s[3]) for s in status)
    stars_complete = sum(1 for s in status if not s[3])
    stars_partial = sum(1 for s in status if s[2] and s[3])
    stars_untouched = sum(1 for s in status if not s[2])

    print("Existing results status:")
    print(f"  Stars : {n_stars} total -- {stars_complete} complete, "
          f"{stars_partial} partial, {stars_untouched} not started")
    print(f"  Fits  : {total_fits} total -- {done_fits} succeeded, "
          f"{pending_fits} missing")

    # Only worth listing per-star detail once some results exist (a re-run).
    incomplete = [s for s in status if s[3]]
    if done_fits and incomplete:
        print("  Stars with work remaining:")
        for name, n_models, done, pending in incomplete[:max_list]:
            print(f"    {name}: {len(done)}/{n_models} done, "
                  f"missing {','.join(pending)}")
        if len(incomplete) > max_list:
            print(f"    ... and {len(incomplete) - max_list} more")
    return done_fits, pending_fits


def _read_previous_job(job_file="dispatch_fits.job"):
    """Parse a previously generated dispatch job for context on a re-run.

    Returns ``(hours_per_fit, walltime_str)`` -- the per-fit hours baked into
    the job by an earlier run (via the marker comment below) and its SLURM
    ``--time``. Either may be ``None`` if not found.
    """
    hours = wall = None
    if os.path.exists(job_file):
        with open(job_file) as f:
            for line in f:
                if "hours-per-fit=" in line:
                    try:
                        hours = float(line.split("hours-per-fit=")[1].split()[0])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("#SBATCH --time="):
                    wall = line.split("=", 1)[1].strip()
    return hours, wall


def _resolve_hours(cli_hours, prev_hours, prev_wall, has_results, parser):
    """Decide the per-fit walltime (hours) for this job.

    Fresh run: ``--hours-per-fit`` is required. Re-run over a directory with
    existing results: report what the previous job used and prompt for a new,
    usually higher, value.
    """
    if not has_results:
        if cli_hours is None:
            parser.error("--hours-per-fit is required for the first run "
                         "(e.g. --hours-per-fit 3).")
        return cli_hours

    ref = cli_hours if cli_hours is not None else prev_hours
    if prev_hours is not None:
        print(f"\nPrevious job: {prev_hours:g} h per model fit "
              f"({_fmt_walltime(prev_hours)}).")
    elif prev_wall is not None:
        # Older job written before the per-fit marker existed: only the pooled
        # per-task --time was recorded, which is not directly the per-fit time.
        print(f"\nPrevious job: per-task --time was {prev_wall} "
              "(per-model-fit time not recorded).")
    print("Fits commonly fail because the per-fit walltime was set too low; "
          "consider raising it for the rerun.")

    if not sys.stdin.isatty():
        chosen = cli_hours if cli_hours is not None else prev_hours
        if chosen is None:
            parser.error("--hours-per-fit is required (stdin is not interactive "
                         "and no previous job was found to reuse its value).")
        print(f"(non-interactive: using {chosen:g} h per fit)")
        return chosen

    # The example illustrates that the input is decimal hours per model fit.
    prompt = "Enter hours-per-fit for the rerun (decimal hours, e.g. 1.5 = 01:30:00)"
    prompt += f" [{ref:g}]: " if ref is not None else ": "
    while True:
        resp = input(prompt).strip()
        if not resp:
            if ref is not None:
                return ref
            print("  Please enter a value (decimal hours).")
            continue
        try:
            val = float(resp)
        except ValueError:
            print("  Enter a positive number of hours (e.g. 4 or 4.5).")
            continue
        if val <= 0:
            print("  Enter a positive number of hours (e.g. 4 or 4.5).")
            continue
        return val


def _detect_venv(repo_dir):
    """Activate script of the venv to source on the node.

    Prefers the venv this script is being run from (so the generated job uses
    the same environment), falling back to the .gvenv in the repo checkout.
    """
    candidates = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(os.path.join(os.environ["VIRTUAL_ENV"], "bin", "activate"))
    candidates.append(os.path.join(sys.prefix, "bin", "activate"))
    candidates.append(os.path.join(repo_dir, ".gvenv", "bin", "activate"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise SystemExit(
        "Could not find a venv to activate. Run this from the Granalyse venv, "
        f"or create one at {os.path.join(repo_dir, '.gvenv')}.")


def _build_source_tarball(repo_dir, dest):
    """
    Package the granalyse source + runner for staging to node scratch.
    """
    def _no_cache(ti):
        parts = ti.name.split("/")
        if "__pycache__" in parts or ti.name.endswith(".pyc"):
            return None
        return ti

    with tarfile.open(dest, "w:gz") as tar:
        tar.add(os.path.join(repo_dir, "granalyse"), arcname="granalyse",
                filter=_no_cache)
        tar.add(os.path.join(repo_dir, "hpc", "run_fitcase.py"),
                arcname="run_fitcase.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Granalyse YAML config")
    parser.add_argument("--hours-per-fit", type=float, default=None,
                        help="Estimated walltime per single model fit, in hours "
                             "(used to size the per-task --time). Required on the "
                             "first run; on a re-run it pre-fills the prompt.")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    units, max_models, status = _scan(args.config, RESULTS_DIR)
    done_fits, pending_fits = _print_diagnostics(status)
    if not units:
        print("\nNothing to do: all fits already have results.")
        sys.exit(0)

    has_results = done_fits > 0
    prev_hours, prev_wall = _read_previous_job()
    hours_per_fit = _resolve_hours(args.hours_per_fit, prev_hours, prev_wall,
                                   has_results, parser)

    n_units = len(units)
    verb = "Recreating" if has_results else "Writing"
    print(f"\n{verb} star_list.txt for {n_units} star(s) with work remaining "
          f"({pending_fits} model fit(s) to run).")

    list_file = os.path.abspath("star_list.txt")
    with open(list_file, "w") as f:
        f.write("\n".join(units) + "\n")

    _build_source_tarball(repo_dir, "granalyse_src.tar.gz")

    # Pool stars into tasks so the array stays within MAX_ARRAY.
    units_per_task = max(1, math.ceil(n_units / MAX_ARRAY))
    n_tasks = math.ceil(n_units / units_per_task)
    limit = min(n_tasks, MAX_PARALLEL)

    # Per-task walltime: hours_per_fit x (models per star) x (stars per task).
    walltime = _fmt_walltime(hours_per_fit * max_models * units_per_task)

    venv = _detect_venv(repo_dir)
    config_abs = os.path.abspath(args.config)

    node_sh = _NODE_SH.format(config=config_abs, results_dir=RESULTS_DIR)
    with open("run_node.sh", "w") as f:
        f.write(node_sh)
    os.chmod("run_node.sh", os.stat("run_node.sh").st_mode | stat.S_IEXEC |
             stat.S_IXGRP | stat.S_IXOTH)

    dispatch_block = _DISPATCH.format(list_file=list_file,
                                      units_per_task=units_per_task)

    job = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --output=Outputs/%x_%A_%a.out
#SBATCH --time={walltime}
#SBATCH --partition={PARTITION}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --array=1-{n_tasks}%{limit}
#SBATCH --export=NONE

echo "========= Job started at `date` on `hostname -s` =========="
export OMP_NUM_THREADS={CPUS}

source "{venv}"
cd "$SLURM_SUBMIT_DIR"

# Ship the granalyse source to scratch
src_scratch="/scratch/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/granalyse_src"
mkdir -p "$src_scratch"
tar -xzf "$SLURM_SUBMIT_DIR/granalyse_src.tar.gz" -C "$src_scratch"
export PYTHONPATH="$src_scratch${{PYTHONPATH:+:$PYTHONPATH}}"
export GRAN_RUNNER="$src_scratch/run_fitcase.py"

{dispatch_block}
echo "========= Job finished at `date` =========="
"""

    os.makedirs("Outputs", exist_ok=True)
    with open("dispatch_fits.job", "w") as f:
        f.write(job)

    print(f"Wrote dispatch_fits.job ({n_units} stars -> {n_tasks} array tasks).")
    print("Submit with:  sbatch dispatch_fits.job")


if __name__ == "__main__":
    main()
